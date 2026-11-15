"""R7.38 causal quality gate: exact parent parity and optimizer continuity."""
import copy
import unittest

import torch

from fa1_r728_geometric_memory import candidate_priority, passes_gate
from fa1_r736_deep_wiring import gradient_path_health
from fa1_r737_feedback_closure import attach_feedback_closure
from fa1_r738_reliability import (ROUND, TriangleReliability,
                                   attach_reliability, reliability_disabled)
from test_fa1_r737_feedback_closure import ClosedFeedbackTests


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(738)
        torch.set_num_threads(2)

    def scene(self):
        parent,opt,inputs,kwargs=ClosedFeedbackTests.scene()
        attach_feedback_closure(parent,opt)
        return parent,opt,inputs,kwargs

    def test_identity_optimizer_moments_and_strict_restore(self):
        parent,opt,inputs,kwargs=self.scene()
        first=next(parent.parameters())
        first.grad=torch.ones_like(first)
        opt._adamw.step()
        moments=copy.deepcopy(opt._adamw.state[first])
        prior_ids={id(p) for p in parent.parameters()}
        with torch.no_grad():
            baseline=parent.diamond(*inputs,**kwargs,return_trace=True)
        attach_reliability(parent,opt)
        with torch.no_grad():
            migrated=parent.diamond(*inputs,**kwargs,return_trace=True)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(baseline[:3],migrated[:3])))
        self.assertEqual(len(opt.param_groups),4)
        self.assertEqual(opt._source_group_counts,[4])
        self.assertTrue(prior_ids.issubset({id(p) for p in parent.parameters()}))
        self.assertEqual(len({id(p) for g in opt.param_groups for p in g['params']}),
                         len(list(parent.parameters())))
        for key in moments:
            self.assertTrue(torch.equal(moments[key],opt._adamw.state[first][key]))
        clone=copy.deepcopy(parent)
        clone.load_state_dict(parent.state_dict(),strict=True)
        with torch.no_grad():
            restored=clone.diamond(*inputs,**kwargs)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(migrated[:3],restored)))
        with self.assertRaisesRegex(ValueError,'already attached'):
            attach_reliability(parent,opt)

    def test_causal_mask_bounded_gain_and_ablations(self):
        model,opt,inputs,kwargs=self.scene()
        attach_reliability(model,opt)
        writer=model.diamond.geometry.history_writer
        keys=torch.randn(2,8,32)
        query=torch.randn(2,32)
        far=torch.randn(2,4,32)
        mask=torch.tensor([[True,False,True,False],[False,False,False,False]])
        with torch.no_grad():
            writer.reliability.gain.fill_(.5)
            first=writer(keys,query,far,mask)
            masked_changed=far.clone()
            masked_changed[:,1] += 1000
            second=writer(keys,query,masked_changed,mask)
            self.assertTrue(torch.equal(first,second))
            self.assertTrue(torch.equal(first[1],torch.zeros_like(first[1])))
            changed=far.clone()
            changed[0,0] += 4
            self.assertGreater(float((first[0]-writer(keys,query,changed,mask)[0]).abs().sum()),0)
            bounded=writer.reliability(keys,query,far,mask)
            self.assertTrue(bool((bounded>=0).all() and (bounded<=2).all()))
            with reliability_disabled(model):
                self.assertEqual(float(writer.reliability.gain),0.)
            self.assertAlmostEqual(float(writer.reliability.gain),.5,places=6)
            baseline=model.diamond(*inputs,**kwargs)
            with reliability_disabled(model):
                without=model.diamond(*inputs,**kwargs)
        self.assertGreater(float((baseline[0]-without[0]).abs().sum()),0.)
        with self.assertRaisesRegex(RuntimeError,'test restore'):
            with reliability_disabled(model):
                raise RuntimeError('test restore')
        self.assertAlmostEqual(float(writer.reliability.gain),.5,places=6)

    def test_quality_path_has_word_and_byte_gradients(self):
        model,opt,inputs,kwargs=self.scene()
        attach_reliability(model,opt)
        with torch.no_grad():
            model.diamond.geometry.history_writer.reliability.gain.fill_(.5)
        model.zero_grad(set_to_none=True)
        hidden,byte_logits,_,trace=model.diamond(*inputs,**kwargs,return_trace=True)
        labels=torch.tensor([[97,98,256],[99,100,256]])
        byte_ar=model.diamond.hierarchy.autoregressive(
            labels,trace['layered_byte_condition'])
        (hidden.square().mean()+byte_logits.square().mean()+byte_ar.square().mean()).backward()
        health=gradient_path_health(model)
        self.assertTrue(health['r738_reliability']['active'])
        self.assertTrue(health['r738_reliability']['finite'])
        self.assertTrue(health['r738_reliability_gate']['active'])

    def test_empty_far_mask_and_invalid_shapes(self):
        module=TriangleReliability()
        keys=torch.randn(2,8,32)
        query=torch.randn(2,32)
        values=torch.randn(2,4,32)
        mask=torch.zeros(2,4,dtype=torch.bool)
        weights=module(keys,query,values,mask)
        self.assertTrue(bool(torch.isfinite(weights).all()))
        self.assertTrue(torch.equal(weights,torch.ones_like(weights)))
        with self.assertRaisesRegex(ValueError,'causal'):
            module(keys,query,values,mask[:,:3])

    def test_selection_requires_positive_quality_causal_effect(self):
        # Even a lower full loss cannot obtain a geometry-ready tier from a
        # reliability control that does nothing or damages predictions.
        ref={'full_word_nll':5.,'conditional_spelling_bits_per_byte':5.5}
        cand={'full_word_nll':4.9,'conditional_spelling_bits_per_byte':5.4,
              'geometry_word_gain':.1,'branch_geometry_gain':.1,
              'correct_book_advantage_nats_per_word':.1,'apex_word_gain':.1,
              'book_geometry_gain':{'book_bootstrap_ci95':[.01,.2]},
              'book_correct_history':{'book_bootstrap_ci95':[.01,.2]},
              'pair_word_gain':.1,'r738_reliability_word_gain':0.,
              'r738_reliability_branch_gain':0.,
              'r738_reliability_byte_gain_nats_per_byte':0.,
              'r738_reliability_word_book_bootstrap':{'book_bootstrap_ci95':[0.,0.]},
              'r738_reliability_byte_book_bootstrap':{'book_bootstrap_ci95':[0.,0.]}}
        self.assertEqual(candidate_priority(ref,cand,.0001)[0],0)


if __name__=='__main__':
    unittest.main()
