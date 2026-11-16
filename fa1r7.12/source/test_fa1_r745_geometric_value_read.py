"""R7.45 normalized sparse value attention and causal ablations."""
import unittest

import torch
from torch import nn

from fa1_r744_causal_word_backoff import CausalWordBackoff
from fa1_r745_geometric_value_read import GeometricWordBackoff
from test_fa1_r744_causal_word_backoff import CausalWordBackoffTests


class GeometricValueReadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(745)
        torch.set_num_threads(2)
        fixture=CausalWordBackoffTests()
        fixture.setUp()
        self.data=fixture.data
        self.trace=fixture.trace
        self.trace['descended_point_summary']=torch.randn(2,32)
        self.embedding=F=nn.Embedding(256,96).weight.detach()
        self.old=CausalWordBackoff()
        self.new=GeometricWordBackoff()
        self.new.load_state_dict(self.old.state_dict(),strict=False)
        self.new.geometric_query=nn.Linear(32,96,bias=False)
        self.new.geometric_gain=nn.Parameter(torch.zeros(()))

    def test_exact_zero_gated_parent_and_normalization(self):
        with torch.no_grad():
            self.old.gain.fill_(.4)
            self.new.gain.copy_(self.old.gain)
        labels=torch.tensor([3,9])
        a,p,_=self.old(self.data,self.trace,labels,self.embedding)
        b,q,meta=self.new(self.data,self.trace,labels,self.embedding)
        self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.equal(p,q))
        self.assertTrue(torch.equal(meta['r745_geometric_beta'],torch.zeros(2)))
        with torch.no_grad():self.new.geometric_gain.fill_(.4)
        allq=[]
        for y in range(10):
            _,value,_=self.new(self.data,self.trace,torch.tensor([y,y]),self.embedding)
            allq.append(value)
        self.assertAlmostEqual(float(torch.stack(allq).sum(0)[0]),1.,places=5)
        self.assertEqual(float(torch.stack(allq).sum(0)[1]),0.)

    def test_geometric_query_changes_value_not_gate_or_masked_tokens(self):
        with torch.no_grad():
            self.new.gain.fill_(.4)
            self.new.geometric_gain.fill_(.5)
        label=torch.tensor([3,9])
        original_alpha,original_q,_=self.new(self.data,self.trace,label,self.embedding)
        changed_trace=dict(self.trace)
        changed_trace['descended_point_summary']=torch.randn(2,32)
        moved_alpha,moved_q,_=self.new(self.data,changed_trace,label,self.embedding)
        self.assertTrue(torch.equal(original_alpha,moved_alpha))
        self.assertGreater(float((original_q-moved_q).abs().max()),0.)
        changed=dict(self.data)
        changed['memory_modal_ids']=self.data['memory_modal_ids'].clone()
        changed['memory_modal_ids'][1]=torch.tensor([199,200])
        masked_alpha,masked_q,_=self.new(changed,self.trace,label,self.embedding)
        self.assertTrue(torch.equal(original_alpha,masked_alpha))
        self.assertTrue(torch.equal(original_q,masked_q))

    def test_geometry_has_live_trainable_gradient(self):
        with torch.no_grad():
            self.new.gain.fill_(.5)
            self.new.geometric_gain.fill_(.5)
        alpha,q,_=self.new(self.data,self.trace,torch.tensor([3,9]),self.embedding)
        loss=-torch.logaddexp(torch.log1p(-alpha)-torch.tensor([4.,4.]),
                             (alpha*q).clamp_min(1.e-38).log()).mean()
        loss.backward()
        self.assertIsNotNone(self.new.geometric_query.weight.grad)
        self.assertTrue(torch.isfinite(self.new.geometric_query.weight.grad).all())
        self.assertGreater(float(self.new.geometric_query.weight.grad.abs().sum()),0.)
        self.assertIsNotNone(self.new.geometric_gain.grad)


if __name__=='__main__':
    unittest.main()
