"""R7.39 in-place migration, signed antisymmetry and causal writer controls."""
import copy
import unittest

import torch

from fa1_r728_geometric_memory import candidate_priority, passes_gate
from fa1_r738_reliability import attach_reliability
from fa1_r739_signed_rails import (SignedRailCross, antisymmetric_cross,
                                   attach_signed_cross_rail, signed_cross_disabled)
import test_fa1_r738_reliability as reliability_fixture


class SignedRailsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(739)
        torch.set_num_threads(2)

    def scene(self):
        model,optimizer,inputs,kwargs=reliability_fixture.ReliabilityTests().scene()
        attach_reliability(model,optimizer)
        return model,optimizer,inputs,kwargs

    def test_identity_continuity_strict_reload_and_reject_duplicate(self):
        model,optimizer,inputs,kwargs=self.scene()
        first=next(model.parameters())
        first.grad=torch.ones_like(first)
        optimizer._adamw.step()
        before_state=copy.deepcopy(optimizer._adamw.state[first])
        old_ids={id(p) for p in model.parameters()}
        with torch.no_grad():
            before=model.diamond(*inputs,**kwargs,return_trace=True)
        attach_signed_cross_rail(model,optimizer)
        with torch.no_grad():
            after=model.diamond(*inputs,**kwargs,return_trace=True)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(before[:3],after[:3])))
        self.assertEqual(len(optimizer.param_groups),5)
        self.assertEqual(optimizer._source_group_counts,[5])
        ids=[id(p) for group in optimizer.param_groups for p in group['params']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertTrue(old_ids.issubset(set(ids)))
        for key,value in before_state.items():
            self.assertTrue(torch.equal(value,optimizer._adamw.state[first][key]))
        clone=copy.deepcopy(model)
        clone.load_state_dict(model.state_dict(),strict=True)
        with torch.no_grad():
            restored=clone.diamond(*inputs,**kwargs,return_trace=True)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(after[:3],restored[:3])))
        with self.assertRaisesRegex(ValueError,'already attached'):
            attach_signed_cross_rail(model,optimizer)

    def test_antisymmetry_and_masked_future_cannot_influence_writer(self):
        p=torch.randn(2,8,32)
        m=torch.randn(2,8,32)
        self.assertTrue(torch.equal(antisymmetric_cross(p,m),
                                    -antisymmetric_cross(m,p)))
        rails=SignedRailCross()
        rails.gain.data.fill_(.4)
        far=torch.randn(2,4,32)
        mask=torch.tensor([[True,False,True,False],[False,False,False,False]])
        plus=torch.randn(2,8)
        minus=torch.randn(2,8)
        first=rails(far,mask,plus,minus)
        changed=far.clone()
        changed[0,1]+=1000
        changed[1]+=1000
        second=rails(changed,mask,plus,minus)
        self.assertTrue(torch.equal(first,second))
        self.assertTrue(torch.equal(first[1],torch.zeros_like(first[1])))
        with self.assertRaisesRegex(ValueError,'causal'):
            rails(far,mask[:,:3],plus,minus)

    def test_live_word_byte_path_and_exception_safe_ablation(self):
        model,optimizer,inputs,kwargs=self.scene()
        attach_signed_cross_rail(model,optimizer)
        with torch.no_grad():
            model.diamond.geometry.history_writer.gain.fill_(.3)
            model.diamond.geometry.rail_cross.gain.fill_(.5)
            active=model.diamond(*inputs,**kwargs)
            with signed_cross_disabled(model):
                off=model.diamond(*inputs,**kwargs)
        self.assertGreater(float((active[0]-off[0]).abs().max()),0.)
        self.assertGreater(float((active[1]-off[1]).abs().max()),0.)
        model.zero_grad(set_to_none=True)
        hidden,byte_logits,_,trace=model.diamond(*inputs,**kwargs,return_trace=True)
        labels=torch.tensor([[97,98,256],[99,100,256]])
        byte_ar=model.diamond.hierarchy.autoregressive(
            labels,trace['layered_byte_condition'])
        (hidden.square().mean()+byte_logits.square().mean()+byte_ar.square().mean()).backward()
        grads=[p.grad for p in model.diamond.geometry.rail_cross.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads),0.)
        with self.assertRaisesRegex(RuntimeError,'restore'):
            with signed_cross_disabled(model):
                raise RuntimeError('restore')
        self.assertAlmostEqual(float(model.diamond.geometry.rail_cross.gain),.5,places=6)

    def test_promotion_gate_rejects_uninformative_cross_rails(self):
        reference={'full_word_nll':5.,'conditional_spelling_bits_per_byte':5.5}
        candidate={'full_word_nll':4.9,'conditional_spelling_bits_per_byte':5.4,
                   'geometry_word_gain':.1,'branch_geometry_gain':.1,
                   'correct_book_advantage_nats_per_word':.1,'apex_word_gain':.1,
                   'pair_word_gain':.1,
                   'book_geometry_gain':{'book_bootstrap_ci95':[.01,.2]},
                   'book_correct_history':{'book_bootstrap_ci95':[.01,.2]},
                   'r739_cross_rail_word_gain':0.,
                   'r739_cross_rail_branch_gain':0.,
                   'r739_cross_rail_byte_gain_nats_per_byte':0.,
                   'r739_cross_rail_specificity_nats_per_word':0.,
                   'r739_cross_rail_byte_specificity_nats_per_byte':0.}
        self.assertEqual(candidate_priority(reference,candidate,.0001)[0],0)


if __name__ == '__main__':
    unittest.main()
