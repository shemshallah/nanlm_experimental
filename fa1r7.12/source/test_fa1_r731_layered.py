"""R7.31 regression: exact migration parity, byte causality, geometry use."""
import copy
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from fa1_r730_factorized_geometry import FactorizedContextualGeometry
from fa1_r731_layered_decoder import (
    EOW, LayeredDiamondDecoder, attach_layered)
from fa1_r728_geometric_memory import layered_byte_metrics


class LayeredTests(unittest.TestCase):
    @staticmethod
    def decoder():
        torch.manual_seed(731)
        m=LayeredDiamondDecoder()
        m.eval()
        return m

    def test_identity_and_sparse_level_shapes(self):
        decoder=self.decoder()
        for level,count in (('point',8),('pair',28),('triangle',56)):
            x=torch.randn(2,count,32)
            self.assertTrue(torch.equal(x,decoder.concept_update(level,x)))
        point=torch.randn(2,32)
        apex=torch.randn(2,32)
        query=torch.randn(2,32)
        word,condition=decoder.descend(point,apex,query)
        self.assertTrue(torch.equal(word,torch.zeros_like(word)))
        self.assertEqual(tuple(condition.shape),(2,32))
        trunk=nn.Module()
        trunk.E=nn.Parameter(torch.randn(130,96))
        delta=decoder.context_update(torch.randint(0,256,(2,128)),
                     torch.randint(0,130,(2,32)),torch.randint(0,130,(2,8)),trunk)
        self.assertTrue(torch.equal(delta,torch.zeros_like(delta)))

    def test_autoregression_excludes_future_target_bytes(self):
        decoder=self.decoder()
        cond=torch.randn(2,32)
        labels=torch.tensor([[97,98,99,EOW,-100],
                             [109,110,111,EOW,-100]])
        altered=labels.clone()
        altered[:,2]=120
        with torch.no_grad():
            before=decoder.autoregressive(labels,cond)
            after=decoder.autoregressive(altered,cond)
        self.assertTrue(torch.equal(before[:,:3],after[:,:3]))
        self.assertEqual(tuple(before.shape),(2,5,257))
        self.assertTrue(torch.allclose(before.softmax(-1).sum(-1),
                                       torch.ones(2,5),atol=1.e-6))

    def test_signed_geometric_feedback_and_gradients(self):
        decoder=self.decoder()
        with torch.no_grad():
            for name in ('point_gain','pair_gain','triangle_gain',
                         'word_gain','byte_geometry_gain'):
                getattr(decoder,name).fill_(.2)
        values={level:torch.randn(2,n,32) for level,n in
                (('point',8),('pair',28),('triangle',56))}
        loops=[decoder.concept_update(level,x).square().mean()
               for level,x in values.items()]
        point,apex,query=(torch.randn(2,32) for _ in range(3))
        word,condition=decoder.descend(point,apex,query)
        self.assertGreater(float(word.abs().sum()),0.)
        labels=torch.tensor([[97,98,EOW],[99,100,EOW]])
        logits=decoder.autoregressive(labels,condition)
        loss=sum(loops)+word.square().mean()+F.cross_entropy(
            logits.transpose(1,2),labels)
        loss.backward()
        for parameter in (decoder.point_loop.attn.in_proj_weight,
                          decoder.pair_loop.attn.in_proj_weight,
                          decoder.triangle_loop.attn.in_proj_weight,
                          decoder.word_readout[0].weight,
                          decoder.byte_head.weight,decoder.byte_geometry_gain):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()),0.)
        with torch.no_grad():
            baseline=decoder.descend(torch.zeros_like(point),apex,query)[1]
            no_apex=decoder.descend(point,torch.zeros_like(apex),query)[1]
            no_both=decoder.descend(torch.zeros_like(point),torch.zeros_like(apex),query)[1]
        self.assertTrue(torch.equal(baseline,no_apex))
        self.assertTrue(torch.equal(baseline,no_both))

    def test_byte_metrics_are_word_conditioned_not_raw_bpb(self):
        decoder=self.decoder()
        labels=torch.tensor([[97,98,EOW,-100], [99,100,101,EOW]])
        logits=decoder.autoregressive(labels,torch.randn(2,32))
        metrics=layered_byte_metrics({'byte_ar_logits':logits},
                                    {'byte_target_sequence':labels})
        self.assertEqual(metrics['conditional_spelling_bytes'],5)
        self.assertEqual(metrics['conditional_spelling_words'],2)
        self.assertIsNone(metrics['raw_contiguous_text_bits_per_byte'])
        for key in ('conditional_spelling_top1','conditional_spelling_top5'):
            self.assertGreaterEqual(metrics[key],0)
            self.assertLessEqual(metrics[key],1)

    def test_optimizer_append_keeps_old_moments(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.diamond=nn.Module()
                self.diamond.d=32
                self.diamond.geometry=FactorizedContextualGeometry()
        model=Model()
        old=list(model.parameters())
        opt=type('Optimizer',(),{})()
        opt._name_by_id={id(p):n for n,p in model.named_parameters()}
        opt._group_names=[{'algorithm':'adamw','names':list(opt._name_by_id.values())}]
        opt._adamw=torch.optim.AdamW(old,lr=1.e-3)
        opt._legacy_groups=[]
        opt._source_group_counts=[1]
        opt.param_groups=opt._adamw.param_groups
        old[0].grad=torch.ones_like(old[0])
        opt._adamw.step()
        moment=copy.deepcopy(opt._adamw.state[old[0]])
        attach_layered(model,opt)
        self.assertEqual(len(opt.param_groups),2)
        self.assertEqual(opt._source_group_counts,[2])
        self.assertTrue(all(id(p) in {id(q) for q in model.parameters()} for p in old))
        for key in moment:
            self.assertTrue(torch.equal(moment[key],opt._adamw.state[old[0]][key]))
        with self.assertRaises(ValueError):attach_layered(model,opt)


if __name__=='__main__':
    unittest.main()
