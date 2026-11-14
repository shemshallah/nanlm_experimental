"""R7.36: migrated identity, normalized sparse values, causal gates, donors."""
import copy
import unittest

import numpy as np
import torch
from torch import nn

from fa1_r731_layered_decoder import LayeredDiamondDecoder, CausalApexFeedback
from fa1_r735_history_values import FarHistoryValueWriter, different_book_writer_values
from fa1_r736_deep_wiring import (attach_deep_wiring, gradient_path_health,
                                  normalized_sparse_history_read,
                                  require_eligible_minibatch,control_disabled)
from fa1_r728_geometric_memory import passes_gate
from test_fa1_r735_history_values import HistoryValueTests


class DeepWiringTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(736)
        torch.set_num_threads(2)

    @staticmethod
    def scene():
        diamond, inputs, kwargs=HistoryValueTests.diamond_scene()
        diamond.geometry.history_writer=FarHistoryValueWriter()
        diamond.hierarchy=LayeredDiamondDecoder()
        diamond.hierarchy.apex_feedback=CausalApexFeedback()
        diamond.hierarchy.apex_feedback_gain=nn.Parameter(torch.tensor(.35))
        with torch.no_grad():
            diamond.geometry.history_writer.gain.fill_(.3)
            diamond.hierarchy.word_gain.fill_(.3)
            diamond.hierarchy.byte_geometry_gain.fill_(.3)
        parent=nn.Module()
        parent.diamond=diamond
        optimizer=type('HybridShell',(),{})()
        old=list(parent.parameters())
        optimizer._name_by_id={id(p):n for n,p in parent.named_parameters()}
        optimizer._group_names=[{'algorithm':'adamw',
                                 'names':list(optimizer._name_by_id.values())}]
        optimizer._adamw=torch.optim.AdamW(old,lr=.0001)
        optimizer._legacy_groups=[]
        optimizer._source_group_counts=[1]
        optimizer.param_groups=optimizer._adamw.param_groups
        return parent, optimizer, inputs, kwargs

    def test_sparse_read_does_not_depend_on_unwritten_triangle_mass(self):
        weights=torch.tensor([[.2,.3,.5],[.1,.15,.75]],requires_grad=True)
        writes=torch.tensor([[[1.,2.],[2.,3.],[0.,0.]],
                             [[1.,2.],[2.,3.],[0.,0.]]],requires_grad=True)
        indices=torch.tensor([[0,1],[0,1]])
        read=normalized_sparse_history_read(weights,writes,indices)
        self.assertTrue(torch.allclose(read[0],read[1],atol=1e-6))
        self.assertTrue(torch.allclose(read[0],torch.tensor([1.6,2.6])))
        read.square().sum().backward()
        self.assertGreater(float(weights.grad.abs().sum()),0.)
        self.assertGreater(float(writes.grad.abs().sum()),0.)
        with self.assertRaises(ValueError):
            normalized_sparse_history_read(weights,writes,indices[:,:0])

    def test_migration_preserves_outputs_moments_and_checkpoint(self):
        parent,opt,inputs,kwargs=self.scene()
        with torch.no_grad():
            original=parent.diamond(*inputs,**kwargs,return_trace=True)
        previous_ids={id(p) for p in parent.parameters()}
        old_param=next(parent.parameters())
        old_param.grad=torch.ones_like(old_param)
        opt._adamw.step()
        old_moments=copy.deepcopy(opt._adamw.state[old_param])
        # Baseline is recomputed after the harmless moment-initialization step.
        with torch.no_grad():
            inherited=parent.diamond(*inputs,**kwargs,return_trace=True)
        attach_deep_wiring(parent,opt)
        with torch.no_grad():
            extended=parent.diamond(*inputs,**kwargs,return_trace=True)
        self.assertEqual(len(opt.param_groups),2)
        self.assertEqual(opt._source_group_counts,[2])
        self.assertEqual({id(p) for p in parent.parameters()}-previous_ids,
                         {id(parent.diamond.geometry.history_read_mix),
                          id(parent.diamond.hierarchy.apex_feedback_center_gain)})
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(inherited[:3],extended[:3])))
        self.assertEqual(set(old_moments),set(opt._adamw.state[old_param]))
        for key in old_moments:
            self.assertTrue(torch.equal(old_moments[key],opt._adamw.state[old_param][key]))
        self.assertEqual(len({id(p) for g in opt.param_groups for p in g['params']}),
                         len(list(parent.parameters())))
        clone=copy.deepcopy(parent)
        clone.load_state_dict(parent.state_dict(),strict=True)
        with torch.no_grad():
            clone_out=clone.diamond(*inputs,**kwargs)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(clone_out,extended[:3])))
        with self.assertRaisesRegex(ValueError,'already attached'):
            attach_deep_wiring(parent,opt)

    def test_new_gates_affect_predictions_and_receive_gradients(self):
        parent,opt,inputs,kwargs=self.scene()
        attach_deep_wiring(parent,opt)
        with torch.no_grad():
            before=parent.diamond(*inputs,**kwargs)
            parent.diamond.geometry.history_read_mix.fill_(.75)
            parent.diamond.hierarchy.apex_feedback_center_gain.fill_(.75)
            after=parent.diamond(*inputs,**kwargs)
        self.assertGreater(float((before[0]-after[0]).abs().sum()),0.)
        parent.zero_grad(set_to_none=True)
        hidden,byte_logits,_,trace=parent.diamond(*inputs,**kwargs,return_trace=True)
        labels=torch.tensor([[97,98,256],[99,100,256]])
        ar=parent.diamond.hierarchy.autoregressive(labels,
                                                    trace['layered_byte_condition'])
        (hidden.square().mean()+byte_logits.square().mean()+ar.square().mean()).backward()
        health=gradient_path_health(parent)
        for name in ('history_query','history_key','history_value','history_write',
                     'apex_feedback','byte_decoder','sparse_read_gate',
                     'feedback_center_gate'):
            self.assertTrue(health[name]['finite'],name)
            self.assertTrue(health[name]['active'],name)
        with torch.no_grad():
            # A writer-only donor cannot affect the output when the apex is
            # removed, regardless of nonzero new controls.
            off=parent.diamond(*inputs,**kwargs,without_apex=True)
            foreign=parent.diamond(*inputs,**kwargs,without_apex=True,
                history_modal_ids=kwargs['modal_ids'][:,8:].flip(0),
                history_modal_mask=kwargs['modal_mask'][:,8:].flip(0))
        self.assertTrue(torch.equal(off[0],foreign[0]))

    def test_wrong_book_must_have_actual_far_history(self):
        ids=torch.arange(36).reshape(3,12)%256
        mask=torch.zeros(3,12,dtype=torch.bool)
        mask[0,8]=True
        mask[0,10]=True
        mask[1,9]=True
        data={'book_ids':torch.tensor([0,1,2]),'memory_modal_ids':ids,
              'memory_modal_mask':mask}
        wrong,donors=different_book_writer_values(data,736)
        self.assertEqual(int(donors[0]),1)
        self.assertEqual(int(donors[1]),0)
        self.assertTrue(wrong['diamond_history_value_mask'][0].any())
        self.assertTrue(wrong['diamond_history_value_mask'][1].any())
        matched,again=different_book_writer_values(
            data,736,preserve_receiver_mask=True)
        self.assertTrue(torch.equal(donors,again))
        self.assertTrue(torch.equal(matched['diamond_history_value_mask'],mask[:,8:]))
        self.assertEqual(int(matched['diamond_history_value_ids'][0,0]),int(ids[1,9]))
        self.assertEqual(int(matched['diamond_history_value_ids'][0,2]),int(ids[1,9]))
        self.assertTrue(torch.equal(matched['memory_modal_ids'],data['memory_modal_ids']))
        mask[1,9]=False
        with self.assertRaisesRegex(ValueError,'two distinct far-eligible books'):
            different_book_writer_values(data,736)

    def test_minibatch_forces_two_eligible_books(self):
        masks=torch.zeros(6,12,dtype=torch.bool)
        masks[1,8]=True
        masks[5,10]=True
        data={'book_ids':torch.tensor([0,0,0,1,1,1]),
              'memory_modal_mask':masks}
        indices=require_eligible_minibatch(data,torch.tensor([0,2,3,4]),
                                           np.random.default_rng(736))
        self.assertTrue(torch.equal(indices[:2],torch.tensor([1,5])))
        masks[5,10]=False
        with self.assertRaisesRegex(ValueError,'two far-eligible training books'):
            require_eligible_minibatch(data,indices,np.random.default_rng(736))

    def test_ablation_restores_gate_after_exception(self):
        control=nn.Parameter(torch.tensor(.17))
        with self.assertRaisesRegex(RuntimeError,'probe'):
            with control_disabled(control):
                self.assertEqual(float(control.detach()),0.)
                raise RuntimeError('probe')
        self.assertAlmostEqual(float(control.detach()),.17,places=6)

    def test_promotion_rejects_nonpositive_new_wiring_gain(self):
        # Even strong inherited gates cannot waive new path attribution.
        positive={'book_bootstrap_ci95':[.001,.01]}
        parent={'full_word_nll':5.0}
        candidate={'full_word_nll':4.9,'geometry_word_gain':.1,
                   'branch_geometry_gain':.1,'correct_book_advantage_nats_per_word':.1,
                   'apex_word_gain':.1,'pair_word_gain':.1,
                   'book_geometry_gain':positive,'book_correct_history':positive}
        for part in ('sparse','center'):
            for metric in ('word_gain','branch_gain','byte_gain_nats_per_byte'):
                candidate['r736_'+part+'_'+metric]=.01
            for metric in ('word_book_bootstrap','byte_book_bootstrap'):
                candidate['r736_'+part+'_'+metric]=positive
        self.assertTrue(passes_gate(parent,candidate,.0001))
        candidate['r736_center_byte_gain_nats_per_byte']=-.001
        self.assertFalse(passes_gate(parent,candidate,.0001))


if __name__=='__main__':
    unittest.main()
