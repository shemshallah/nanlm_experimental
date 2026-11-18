#!/usr/bin/env python3
"""R7.64 low/high-resolution rails, zero-parity, causal boundaries and LOO."""
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from fa1_r757_geometric_transition_router import VOCAB
from fa1_r764_multiresolution import (
    HEADS, ROUND, BANK_FIELDS, PREFIXES, MultiresolutionPrior,
    build_multires_banks)
from test_fa1_r763_temporal_cross import TemporalTests


class MultiresolutionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = TemporalTests('test_exact_parent_parity_and_missing_history')
        self.fixture.setUp()
        self.model = self.fixture.model
        self.model.__class__ = MultiresolutionPrior
        self.args = self.fixture.args
        recent = int(self.args[1][0])
        older = int(self.args[0][0])
        third = 7
        heads = self.fixture.fixture.heads
        h30, h31 = heads[third].tolist()
        h20, h21 = heads[older].tolist()
        addresses = (h30 * HEADS + h21, h31 * HEADS + h20,
                     recent * HEADS + h30, recent * HEADS + h21)
        for prefix, address in zip(PREFIXES, addresses):
            self.model.register_buffer(prefix + '_keys',
                                       torch.tensor([address * VOCAB + 4]))
            self.model.register_buffer(prefix + '_counts', torch.tensor([5]))
            self.model.register_buffer(prefix + '_addresses',
                                       torch.tensor([address]))
            self.model.register_buffer(prefix + '_totals', torch.tensor([5]))
        self.model.r764_strength = nn.Parameter(torch.zeros(()))
        self.model.r764_log_tau = nn.Parameter(
            torch.tensor([math.log(64.)] * 4))
        self.model.r764_selector = nn.Sequential(
            nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 5))
        nn.init.zeros_(self.model.r764_selector[2].weight)
        with torch.no_grad():
            self.model.r764_selector[2].bias.copy_(
                torch.tensor([0., 0., 0., 0., 2.]))

    def test_exact_initial_parity_and_invalid_third_word(self):
        with torch.no_grad():
            self.model._r764_enabled = False
            parent = self.model(*self.args,
                                third_older=torch.tensor([7]))[0]
            self.model._r764_enabled = True
            migrated = self.model(*self.args,
                                  third_older=torch.tensor([7]))[0]
            self.assertTrue(torch.equal(parent, migrated))
            self.model.r764_strength.fill_(0.7)
            missing = self.model(*self.args,
                                 third_older=torch.tensor([7]),
                                 valid_third=torch.tensor([False]))[0]
            self.assertTrue(torch.equal(parent, missing))

    def test_trainable_selector_and_all_counterfactuals(self):
        with torch.no_grad():
            self.model.r764_strength.fill_(0.7)
        probability = self.model(*self.args,
                                 third_older=torch.tensor([7]))[0]
        self.assertTrue(torch.isfinite(probability).all())
        self.assertTrue(((probability >= 0) & (probability <= 1)).all())
        probability.backward()
        for item in (self.model.r764_strength,
                     self.model.r764_log_tau,
                     self.model.r764_selector[2].weight):
            self.assertTrue(torch.isfinite(item.grad).all())
        with torch.no_grad():
            for control in ('lexical_matched', 'wrong_older',
                            'wrong_geometry_values', 'disabled'):
                p = self.model(*self.args,
                               third_older=torch.tensor([7]),
                               control=control)[0]
                self.assertTrue(torch.isfinite(p).all())

    def test_four_bank_train_counts_do_not_bridge_books(self):
        class CorpusFixture:
            def __init__(self, *args, **kwargs):
                self.seq = np.array([7, 2, 3, 4, 7, 2, 3, 4], dtype=np.int64)
                self.doc_start = np.array([0, 0, 0, 0, 4, 4, 4, 4])
        with patch('fa1_r764_multiresolution.Corpus', CorpusFixture):
            banks = build_multires_banks(
                '/unused', '/unused', self.fixture.fixture.heads)
        self.assertEqual(len(BANK_FIELDS), 16)
        for prefix in PREFIXES:
            self.assertEqual(int(banks[prefix + '_counts'].sum()), 2)
            self.assertEqual(int(banks[prefix + '_totals'].sum()), 2)
        self.assertNotEqual(
            int(banks['coarse_plus_addresses'][0]),
            int(banks['coarse_minus_addresses'][0]))
        self.assertEqual(
            ROUND, 'FA1_R7_64_MULTISCALE_TEMPORAL_DIAMOND_V1')


if __name__ == '__main__':
    unittest.main()
