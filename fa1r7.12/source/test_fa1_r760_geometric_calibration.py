#!/usr/bin/env python3
"""R7.60 calibration bounds, lexical parity and differentiability."""
import unittest
import torch
from torch import nn
from test_fa1_r759_geometric_prior import PriorTests
from fa1_r759_geometric_prior import GeometricPrior
from fa1_r760_geometric_calibration import CalibratedGeometricPrior


class CalibrationTests(PriorTests):
    def make_calibrator(self):
        model=GeometricPrior(self.banks)
        model.__class__=CalibratedGeometricPrior
        model.r760_shrinkage=nn.Parameter(torch.zeros(()))
        model.r760_selector=nn.Linear(37,1)
        nn.init.zeros_(model.r760_selector.weight)
        nn.init.zeros_(model.r760_selector.bias)
        return model

    def args(self):
        return (torch.tensor([2]),torch.tensor([3]),torch.tensor([4]),
                self.heads,torch.ones((1,32)),self.exact_bank,self.words)

    def test_exact_initial_r759_parity(self):
        model=self.make_calibrator()
        previous=GeometricPrior(self.banks)
        with torch.no_grad():
            for mode in ('full','lexical_matched','wrong_older','wrong_geometry_values'):
                out=model(*self.args(),control=mode)
                old=previous(*self.args(),control=mode)
                self.assertTrue(all(torch.equal(x,y) for x,y in zip(out,old)))

    def test_shrinkage_is_bounded_by_lexical(self):
        model=self.make_calibrator()
        with torch.no_grad():
            model.r760_shrinkage.fill_(2.)
            x=model(*self.args())[0]
            lex=model(*self.args(),control='lexical_matched')[0]
            self.assertTrue(torch.equal(x,lex))
            model.r760_shrinkage.fill_(1.)
            mid=model(*self.args())[0]
            model._r760_enabled=False
            geo=model(*self.args())[0]
            self.assertGreaterEqual(mid.item(),lex.item())
            self.assertLessEqual(mid.item(),geo.item())

    def test_shrinkage_gradient(self):
        model=self.make_calibrator()
        out=model(*self.args())[0]
        out.backward()
        self.assertIsNotNone(model.r760_shrinkage.grad)
        self.assertTrue(torch.isfinite(model.r760_shrinkage.grad))

if __name__=='__main__':unittest.main()
