#!/usr/bin/env python3
"""Matched lexical/geometric Bayesian smoothing, causal LOO and parity."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from fa1_r759_geometric_prior import (
    GeometricPrior, conditioned_address, build_conditioned_bank, BANK_FIELDS)
from fa1_r758_contextual_geometry import ContextualGeometry, VOCAB, ADDRESSES
from fa1_r757_geometric_transition_router import TransitionRouter

class PriorTests(unittest.TestCase):
    def setUp(self):
        self.heads=torch.zeros((VOCAB,2),dtype=torch.long)
        self.heads[2,1]=7
        self.heads[5,1]=7
        self.heads[3,0]=11
        self.addr=3*256+7
        self.exact=2*VOCAB+3
        self.geo=self.heads[2,1]*256+self.heads[3,0]
        self.exact_bank=ContextualGeometry(dict(
            geo_keys=torch.tensor([int(self.geo)*VOCAB+4]),geo_counts=torch.tensor([2]),
            geo_totals=torch.zeros(ADDRESSES,dtype=torch.long),
            exact_keys=torch.tensor([self.exact*VOCAB+4]),exact_counts=torch.tensor([2]),
            exact_addresses=torch.tensor([self.exact]),exact_totals=torch.tensor([2])))
        self.exact_bank.geo_totals[int(self.geo)]=2
        totals=torch.zeros(VOCAB,dtype=torch.long)
        totals[3]=5
        self.words=TransitionRouter(dict(
            word_keys=torch.tensor([3*VOCAB+4,3*VOCAB+6]),
            word_counts=torch.tensor([2,3]),word_totals=totals,
            signed_keys=torch.tensor([1*VOCAB+4]),signed_counts=torch.tensor([1]),
            signed_totals=torch.zeros(2*32640,dtype=torch.long)))
        self.banks=dict(conditional_keys=torch.tensor([self.addr*VOCAB+4]),
                        conditional_counts=torch.tensor([3]),
                        conditional_addresses=torch.tensor([self.addr]),
                        conditional_totals=torch.tensor([3]))

    def test_address_changes_with_exact_recent_word(self):
        self.assertEqual(conditioned_address(torch.tensor([2]),torch.tensor([3]),
                                             self.heads).item(),self.addr)
        self.assertNotEqual(conditioned_address(torch.tensor([2]),torch.tensor([4]),
                                                 self.heads).item(),self.addr)

    def test_probability_and_lexical_matched_control(self):
        model=GeometricPrior(self.banks)
        args=(torch.tensor([2]),torch.tensor([3]),torch.tensor([4]),
              self.heads,torch.ones((1,32)),self.exact_bank,self.words)
        geo,w,ec,gc,wc,k=model(*args)
        lex,*_=model(*args,control='lexical_matched')
        self.assertEqual(w.item(),0.)
        self.assertGreater(geo.item(),lex.item())
        self.assertAlmostEqual(geo.item(),1.)
        with torch.no_grad():model.gain.fill_(.5)
        geo,w,*_=model(*args)
        self.assertGreater(w.item(),0.)
        (-torch.log(geo*w)).backward()
        self.assertIsNotNone(model.route.weight.grad)

    def test_train_leave_one_out_and_wrong_values(self):
        model=GeometricPrior(self.banks)
        args=(torch.tensor([2]),torch.tensor([3]),torch.tensor([4]),
              self.heads,torch.zeros((1,32)),self.exact_bank,self.words)
        good,*_=model(*args,leave_one_out=True)
        bad,*_=model(*args,control='wrong_geometry_values',leave_one_out=True)
        self.assertLess(bad.item(),good.item())
        self.assertEqual(set(dict(model.named_buffers())),set(BANK_FIELDS))

    def test_train_bank_excludes_cross_book(self):
        class CorpusFixture:
            def __init__(self,*args,**kwargs):
                self.seq=np.array([2,3,4,5,3,6],dtype=np.int64)
                self.doc_start=np.array([0,0,0,3,3,3])
        with patch('fa1_r759_geometric_prior.Corpus',CorpusFixture):
            banks=build_conditioned_bank('/unused','/unused',self.heads)
        self.assertEqual(int(banks['conditional_counts'].sum()),2)

if __name__=='__main__':unittest.main()
