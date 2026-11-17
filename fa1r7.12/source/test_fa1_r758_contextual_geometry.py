#!/usr/bin/env python3
"""Causal R7.58 geometry, mirror and target-independent gate regressions."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from fa1_r758_contextual_geometry import (
    ADDRESSES, VOCAB, BANK_FIELDS, ContextualGeometry,
    sparse_probability, temporal_address, build_context_banks)


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.heads = torch.zeros((VOCAB, 2), dtype=torch.long)
        self.heads[2] = torch.tensor([1, 7])
        self.heads[5] = torch.tensor([1, 9])
        self.heads[3] = torch.tensor([11, 13])
        self.g0, self.g1 = 7*256+11, 9*256+11
        self.banks = dict(geo_keys=torch.tensor([self.g0*VOCAB+4, self.g1*VOCAB+6]),
                          geo_counts=torch.tensor([2, 2]),
                          geo_totals=torch.zeros(ADDRESSES, dtype=torch.long),
                          exact_keys=torch.tensor([(2*VOCAB+3)*VOCAB+4,
                                                   (5*VOCAB+3)*VOCAB+6]),
                          exact_counts=torch.tensor([2, 2]),
                          exact_addresses=torch.tensor([2*VOCAB+3, 5*VOCAB+3]),
                          exact_totals=torch.tensor([2, 2]))
        self.banks['geo_totals'][self.g0] = 2
        self.banks['geo_totals'][self.g1] = 2

    def test_conditioned_on_two_distinct_words(self):
        a = temporal_address(torch.tensor([2, 5]), torch.tensor([3, 3]), self.heads)
        self.assertEqual(a.tolist(), [self.g0, self.g1])

    def test_mirror_fixed_points(self):
        addr = self.g0
        mirror = (addr%256)*256+addr//256
        self.assertNotEqual(addr, mirror)
        self.assertEqual((mirror%256)*256+mirror//256, addr)
        self.heads[2,1] = 11
        self.assertEqual(temporal_address(torch.tensor([2]),torch.tensor([3]),self.heads).item(), 11*257)

    def test_probability_lookup_and_leave_one_out(self):
        router = ContextualGeometry(self.banks)
        out = router(torch.tensor([2,5]), torch.tensor([3,3]),
                     torch.tensor([4,4]), self.heads, torch.zeros((2,32)))
        self.assertEqual(out[0].tolist(), [1.,0.])
        self.assertEqual(out[1].tolist(), [1.,0.])
        loo = router(torch.tensor([2,5]), torch.tensor([3,3]),
                     torch.tensor([4,6]), self.heads, torch.zeros((2,32)),
                     leave_one_out=True, valid=torch.tensor([True,False]))
        self.assertEqual(loo[0].tolist(), [1.,0.])
        self.assertEqual(loo[1].tolist(), [1.,0.])

    def test_singleton_no_self_credit(self):
        totals = torch.zeros(VOCAB,dtype=torch.long)
        totals[2] = 1
        q, d = sparse_probability(torch.tensor([2*VOCAB+4]),torch.tensor([1]),
                                  torch.tensor([2]),torch.tensor([4]),totals,
                                  leave_one_out=torch.tensor([True]))
        self.assertEqual((q.item(),d.item()),(0.,0))

    def test_zero_gate_exact_parity_and_gradients(self):
        r = ContextualGeometry(self.banks)
        self.assertEqual(set(dict(r.named_buffers())),set(BANK_FIELDS))
        args = (torch.tensor([2]),torch.tensor([3]),torch.tensor([4]),
                self.heads,torch.ones((1,32)))
        out = r(*args)
        self.assertEqual((out[2].item(),out[3].item()),(0.,0.))
        with torch.no_grad(): r.strength.fill_(.5)
        eq,gq,ew,gw,*_ = r(*args)
        (-torch.log((eq*ew+gq*gw).clamp_min(1e-9))).backward()
        self.assertIsNotNone(r.logits.weight.grad)
        wrong = r(*args,control='wrong_geo_values')
        self.assertTrue(torch.equal(eq,wrong[0]))
        self.assertEqual(wrong[1].item(),0.)

    def test_no_cross_book_training_triples(self):
        class CorpusFixture:
            def __init__(self,*args,**kwargs):
                self.seq=np.array([2,3,4,5,3,6],dtype=np.int64)
                self.doc_start=np.array([0,0,0,3,3,3])
        with patch('fa1_r758_contextual_geometry.Corpus',CorpusFixture):
            banks=build_context_banks('/unused','/unused',self.heads)
        self.assertEqual(banks['geo_counts'].sum().item(),2)
        self.assertEqual(banks['exact_counts'].sum().item(),2)


if __name__=='__main__': unittest.main()
