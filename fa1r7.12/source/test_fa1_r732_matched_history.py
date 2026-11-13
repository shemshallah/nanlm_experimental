"""R7.32 controls: isolate older geometric evidence without target leakage."""
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from fa1_r732_matched_history import (matched_history_context,
                                      geometry_specific_history_gain,
                                      context_specific_margin)
from fa1_r728_geometric_memory import warm_start_only_if_migrated


class MatchedHistoryTests(unittest.TestCase):
    @staticmethod
    def batch():
        n=6
        g=torch.Generator().manual_seed(732)
        return {'book_ids':torch.tensor([1,1,2,2,3,3]),
                'target':torch.arange(n)+20,
                'hidden':torch.randn(n,96,generator=g),
                'word_ids':torch.arange(n*32).reshape(n,32),
                'byte_ids':torch.arange(n*128).reshape(n,128)%256,
                'sentence_ids':torch.arange(n*8).reshape(n,8),
                'vertex_ids':torch.arange(8).repeat(n,1),
                'near_features':torch.randn(n,60,generator=g),
                'far_features':torch.randn(n,32,generator=g),
                'memory_modal_ids':torch.arange(n*12).reshape(n,12),
                'memory_modal_mask':torch.ones(n,12,dtype=torch.bool),
                'byte_target_sequence':torch.arange(n*65).reshape(n,65)%256,
                'first_byte_target':torch.arange(n)}

    def test_matched_recent_and_targets_are_bit_identical(self):
        data=self.batch()
        donor,ix=matched_history_context(data,732)
        self.assertTrue(torch.all(data['book_ids'][ix]!=data['book_ids']))
        for key, keep in (('word_ids',8),('byte_ids',32),('sentence_ids',2)):
            self.assertTrue(torch.equal(donor[key][:,-keep:],data[key][:,-keep:]))
            self.assertTrue(torch.equal(donor[key][:,:-keep],data[key][ix,:-keep]))
        for key in ('target','hidden','book_ids','vertex_ids',
                    'byte_target_sequence','first_byte_target','near_features'):
            self.assertTrue(torch.equal(donor[key],data[key]))
        self.assertTrue(torch.equal(donor['diamond_memory_features'][:,:60],data['near_features']))
        self.assertTrue(torch.equal(donor['diamond_memory_features'][:,60:],data['far_features'][ix]))
        self.assertTrue(torch.equal(donor['diamond_modal_ids'][:,:8],data['memory_modal_ids'][:,:8]))
        self.assertTrue(torch.equal(donor['diamond_modal_ids'][:,8:],data['memory_modal_ids'][ix,8:]))
        self.assertTrue(torch.equal(data['word_ids'],self.batch()['word_ids']))

    def test_reject_invalid_windows_and_shapes(self):
        data=self.batch()
        with self.assertRaises(ValueError):matched_history_context(data,732,recent_bytes=129)
        with self.assertRaises(ValueError):matched_history_context(data,732,recent_words=-1)
        with self.assertRaises(ValueError):matched_history_context({**data,'word_ids':data['word_ids'][:,:4]})
        with self.assertRaises(ValueError):matched_history_context({**data,'book_ids':torch.ones(6)})

    def test_difference_of_differences_and_live_gradients(self):
        correct=torch.tensor([1.0,2.0],requires_grad=True)
        wrong=torch.tensor([1.4,2.4],requires_grad=True)
        off_correct=torch.tensor([1.3,2.3])
        off_wrong=torch.tensor([1.5,2.5])
        gain=geometry_specific_history_gain(correct,wrong,off_correct,off_wrong)
        self.assertTrue(torch.allclose(gain,torch.tensor([.2,.2]),atol=1e-6))
        loss=context_specific_margin(gain,margin=.005)
        loss.backward()
        self.assertIsNotNone(correct.grad)
        self.assertIsNotNone(wrong.grad)
        self.assertTrue(bool((correct.grad>0).all()))  # raising correct NLL worsens contrast
        self.assertTrue(bool((wrong.grad<0).all()))    # raising wrong NLL helps contrast
        with self.assertRaises(ValueError):
            geometry_specific_history_gain(correct,wrong[:1],off_correct,off_wrong)

    def test_geometry_off_history_has_no_gain(self):
        true=torch.tensor([2.,2.1])
        wrong=torch.tensor([2.2,2.3])
        gain=geometry_specific_history_gain(true,wrong,true,wrong)
        self.assertTrue(torch.equal(gain,torch.zeros_like(gain)))
        self.assertGreater(float(context_specific_margin(gain)),0)

    def test_candidate_resume_does_not_reset_learned_gates(self):
        hierarchy=nn.Module()
        names=('context_gain','point_gain','pair_gain','triangle_gain',
               'word_gain','byte_geometry_gain')
        for name in names:
            hierarchy.register_parameter(name,nn.Parameter(torch.tensor(.31)))
        geometry=nn.Module()
        for name in ('meta_gate','context_read_gain','factor_up_gain','factor_decoder_gain'):
            geometry.register_parameter(name,nn.Parameter(torch.tensor(.27)))
        diamond=nn.Module()
        diamond.register_parameter('gate',nn.Parameter(torch.tensor(.44)))
        diamond.geometry=geometry
        diamond.hierarchy=hierarchy
        model=SimpleNamespace(diamond=diamond)
        args=SimpleNamespace(layered=True,bridge=.35,meta=.15,contextual_init=.15,
                             factor_up_init=.20,factor_decoder_init=.35,layered_gate_init=.12)
        old={k:v.detach().clone() for k,v in diamond.named_parameters()}
        warmed=warm_start_only_if_migrated(model,args,{
            'architecture_round':'FA1_R7_31_LAYERED_DIAMOND_V1'})
        self.assertFalse(warmed)
        self.assertTrue(all(torch.equal(v,dict(diamond.named_parameters())[k])
                            for k,v in old.items()))
        warmed=warm_start_only_if_migrated(model,args,{
            'architecture_round':'FA1_R7_32_CALIBRATED_DIAMOND_V1'})
        self.assertFalse(warmed)
        self.assertTrue(all(torch.equal(v,dict(diamond.named_parameters())[k])
                            for k,v in old.items()))
        warmed=warm_start_only_if_migrated(model,args,{
            'architecture_round':'FA1_R7_30_FACTORIZED_GEOMETRY_V1'})
        self.assertTrue(warmed)
        self.assertNotEqual(float(diamond.gate),float(old['gate']))


if __name__=='__main__':
    unittest.main()
