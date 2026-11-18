#!/usr/bin/env python3
"""R7.63 proper four-word causal memory, opposing orientations and parity."""
import math
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from test_fa1_r762_selective_intersection import SelectorTests
from fa1_r757_geometric_transition_router import VOCAB
from fa1_r763_temporal_cross import (
    ROUND,HEADS,BANK_FIELDS,build_temporal_banks,TemporalCrossPrior)


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.fixture=SelectorTests('test_parity_and_gradient')
        self.fixture.setUp()
        self.fixture.heads[7,0]=12
        self.fixture.heads[7,1]=13
        self.model=self.fixture.grow(self.fixture.make_model())
        self.model.__class__=TemporalCrossPrior
        f=3*HEADS*HEADS+12*HEADS+7
        r=3*HEADS*HEADS+13*HEADS+0
        for prefix,address in (('forward',f),('reverse',r)):
            self.model.register_buffer(prefix+'_keys',torch.tensor([address*VOCAB+4]))
            self.model.register_buffer(prefix+'_counts',torch.tensor([3]))
            self.model.register_buffer(prefix+'_addresses',torch.tensor([address]))
            self.model.register_buffer(prefix+'_totals',torch.tensor([3]))
        self.model.r763_strength=nn.Parameter(torch.zeros(()))
        self.model.r763_log_tau=nn.Parameter(torch.tensor(
            [math.log(64.),math.log(64.)]))
        self.model.r763_selector=nn.Linear(6,3)
        nn.init.zeros_(self.model.r763_selector.weight)
        with torch.no_grad():
            self.model.r763_selector.bias.copy_(torch.tensor([0.,0.,2.]))
        self.args=self.fixture.args()

    def test_exact_parent_parity_and_missing_history(self):
        with torch.no_grad():
            self.model._r763_enabled=False
            parent=self.model(*self.args)[0]
            self.model._r763_enabled=True
            actual=self.model(*self.args,third_older=torch.tensor([7]))[0]
            self.assertTrue(torch.equal(parent,actual))
            self.model.r763_strength.fill_(.6)
            missing=self.model(*self.args,third_older=torch.tensor([7]),
                valid_third=torch.tensor([False]))[0]
            self.assertTrue(torch.equal(parent,missing))

    def test_temporal_gradient_and_controls(self):
        with torch.no_grad():self.model.r763_strength.fill_(.6)
        result=self.model(*self.args,third_older=torch.tensor([7]))[0]
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(((result>=0)&(result<=1)).all())
        result.backward()
        self.assertTrue(torch.isfinite(self.model.r763_strength.grad).all())
        self.assertTrue(torch.isfinite(self.model.r763_log_tau.grad).all())
        self.assertTrue(torch.isfinite(self.model.r763_selector.weight.grad).all())
        with torch.no_grad():
            for control in ('lexical_matched','wrong_older',
                            'wrong_geometry_values','disabled'):
                result=self.model(*self.args,third_older=torch.tensor([7]),
                                  control=control)[0]
                self.assertTrue(torch.isfinite(result).all())

    def test_train_bank_is_four_word_and_book_boundary_safe(self):
        class CorpusFixture:
            def __init__(self,*args,**kwargs):
                self.seq=np.array([7,2,3,4,7,2,3,4],dtype=np.int64)
                self.doc_start=np.array([0,0,0,0,4,4,4,4])
        with patch('fa1_r763_temporal_cross.Corpus',CorpusFixture):
            bank=build_temporal_banks('/unused','/unused',self.fixture.heads)
        self.assertEqual(len(BANK_FIELDS),8)
        self.assertEqual(int(bank['forward_counts'].sum()),2)
        self.assertEqual(int(bank['reverse_counts'].sum()),2)
        self.assertNotEqual(int(bank['forward_addresses'][0]),
                            int(bank['reverse_addresses'][0]))
        self.assertEqual(ROUND,'FA1_R7_63_TEMPORAL_CROSS_RAIL_GEOMETRY_V1')


if __name__=='__main__':unittest.main()
