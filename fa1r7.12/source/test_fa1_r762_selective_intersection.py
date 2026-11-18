#!/usr/bin/env python3
"""R7.62 learned two-rail selector: exact parity and causal semantics."""
import math
import unittest
import torch
from torch import nn
from test_fa1_r761_two_rail_geometry import RailTests as R761Tests
from fa1_r762_selective_intersection import (
    SelectiveIntersectionPrior,ROUND,BANK_FIELDS)
from fa1_r757_geometric_transition_router import VOCAB


class SelectorTests(R761Tests):
    def grow(self,model):
        model.__class__=SelectiveIntersectionPrior
        first=int(3*256+self.heads[2,0])
        cross=int(3*65536+self.heads[2,0]*256+self.heads[2,1])
        for prefix,address in (('rail0',first),('cross',cross)):
            model.register_buffer(prefix+'_keys',torch.tensor([address*VOCAB+4]))
            model.register_buffer(prefix+'_counts',torch.tensor([3]))
            model.register_buffer(prefix+'_addresses',torch.tensor([address]))
            model.register_buffer(prefix+'_totals',torch.tensor([3]))
        model.r762_strength=nn.Parameter(torch.zeros(()))
        model.r762_log_tau=nn.Parameter(torch.tensor(
            [math.log(32.),math.log(32.),math.log(64.)]))
        model.r762_selector=nn.Linear(5,4)
        nn.init.zeros_(model.r762_selector.weight)
        with torch.no_grad():
            model.r762_selector.bias.copy_(torch.tensor([0.,0.,0.,2.]))
        return model

    def test_parity_and_gradient(self):
        model=self.make_model()
        with torch.no_grad():base=model(*self.args())[0]
        self.grow(model)
        with torch.no_grad():attached=model(*self.args())[0]
        self.assertTrue(torch.equal(base,attached))
        self.assertEqual(len(BANK_FIELDS),8)
        with torch.no_grad():model.r762_strength.fill_(.6)
        p=model(*self.args())[0]
        self.assertTrue(torch.isfinite(p).all())
        self.assertGreaterEqual(p.item(),0.)
        self.assertLessEqual(p.item(),1.)
        p.backward()
        for name in ('r762_strength','r762_log_tau'):
            self.assertTrue(torch.isfinite(getattr(model,name).grad).all())
        self.assertTrue(torch.isfinite(model.r762_selector.weight.grad).all())

    def test_lexical_null_expert_and_geometry_ablation(self):
        model=self.grow(self.make_model())
        with torch.no_grad():
            model.r762_strength.fill_(.6)
            model.r762_selector.bias.copy_(torch.tensor([-20.,-20.,-20.,20.]))
            null=model(*self.args())[0]
            model._r762_enabled=False
            inherited=model(*self.args())[0]
            lexical=model(*self.args(),control='lexical_matched')[0]
            model._r762_enabled=True
            expected=inherited+torch.tanh(model.r762_strength)*(lexical-inherited)
            self.assertTrue(torch.allclose(expected,null,atol=1e-6))
            model.r762_selector.bias.copy_(torch.tensor([20.,-20.,-20.,-20.]))
            geom=model(*self.args())[0]
            self.assertTrue(torch.isfinite(geom).all())
            disabled=model(*self.args(),control='disabled')[0]
            self.assertTrue(torch.isfinite(disabled).all())
            self.assertEqual(ROUND,'FA1_R7_62_SELECTIVE_GEOMETRIC_INTERSECTION_V1')


if __name__=='__main__':unittest.main()
