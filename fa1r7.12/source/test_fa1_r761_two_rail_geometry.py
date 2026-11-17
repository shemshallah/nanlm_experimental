#!/usr/bin/env python3
"""R7.61 two-rail causal boundaries, parity, normalization and gradients."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from test_fa1_r759_geometric_prior import PriorTests
from fa1_r759_geometric_prior import GeometricPrior
from fa1_r760_geometric_calibration import CalibratedGeometricPrior
from fa1_r761_two_rail_geometry import (
    TwoRailGeometricPrior,build_two_rail_bank,BANK_FIELDS)
from fa1_r757_geometric_transition_router import VOCAB


class RailTests(PriorTests):
    def make_model(self):
        model=GeometricPrior(self.banks)
        model.__class__=CalibratedGeometricPrior
        model.r760_shrinkage=nn.Parameter(torch.zeros(()))
        model.r760_selector=nn.Linear(37,1)
        nn.init.zeros_(model.r760_selector.weight)
        nn.init.zeros_(model.r760_selector.bias)
        return model

    def grow(self,model):
        model.__class__=TwoRailGeometricPrior
        first=int(3*256+self.heads[2,0])
        pair=int(3*65536+self.heads[2,0]*256+self.heads[2,1])
        for prefix,addr in (('rail0',first),('cross',pair)):
            model.register_buffer(prefix+'_keys',torch.tensor([addr*VOCAB+4]))
            model.register_buffer(prefix+'_counts',torch.tensor([3]))
            model.register_buffer(prefix+'_addresses',torch.tensor([addr]))
            model.register_buffer(prefix+'_totals',torch.tensor([3]))
        model.r761_strength=nn.Parameter(torch.zeros(()))
        model.r761_log_tau=nn.Parameter(torch.tensor(np.log(32.),dtype=torch.float32))
        model.r761_selector=nn.Linear(5,1)
        nn.init.zeros_(model.r761_selector.weight)
        nn.init.zeros_(model.r761_selector.bias)
        return model

    def args(self):
        return (torch.tensor([2]),torch.tensor([3]),torch.tensor([4]),
                self.heads,torch.ones((1,32)),self.exact_bank,self.words)

    def test_parity_and_gradient(self):
        model=self.make_model()
        with torch.no_grad():prior=model(*self.args())[0]
        self.grow(model)
        with torch.no_grad():after=model(*self.args())[0]
        self.assertTrue(torch.equal(prior,after))
        self.assertEqual(len(BANK_FIELDS),8)
        with torch.no_grad():model.r761_strength.fill_(.7)
        score=model(*self.args())[0]
        self.assertTrue(torch.isfinite(score).all())
        self.assertGreaterEqual(score.item(),0.)
        self.assertLessEqual(score.item(),1.)
        score.backward()
        self.assertTrue(torch.isfinite(model.r761_strength.grad))

    def test_bank_document_boundary(self):
        class Dummy:
            def __init__(self,*args,**kwargs):
                self.seq=np.array([2,3,4,5,3,6],dtype=np.int64)
                self.doc_start=np.array([0,0,0,3,3,3])
        with patch('fa1_r761_two_rail_geometry.Corpus',Dummy):
            banks=build_two_rail_bank('/unused','/unused',self.heads)
        self.assertEqual(int(banks['rail0_counts'].sum()),2)
        self.assertEqual(int(banks['cross_counts'].sum()),2)

if __name__=='__main__':unittest.main()
