#!/usr/bin/env python3
"""Causal and finite-distribution regressions for R7.57."""
import unittest

import numpy as np
import torch

from fa1_r757_geometric_transition_router import (
    TransitionRouter, signed_pair_numpy, VOCAB, BANK_FIELDS)


class TestTrainOnlyTransitions(unittest.TestCase):
    def banks(self):
        return {
            'word_keys': torch.tensor([2 * VOCAB + 3, 2 * VOCAB + 4]),
            'word_counts': torch.tensor([2, 1]),
            'word_totals': torch.cat([torch.zeros(2, dtype=torch.long),
                                      torch.tensor([3]),
                                      torch.zeros(VOCAB - 3, dtype=torch.long)]),
            'signed_keys': torch.tensor([7 * VOCAB + 3, 7 * VOCAB + 4]),
            'signed_counts': torch.tensor([2, 1]),
            'signed_totals': torch.cat([torch.zeros(7, dtype=torch.long),
                                        torch.tensor([3]),
                                        torch.zeros(32640 * 2 - 8, dtype=torch.long)]),
        }

    def test_optimizer_buffers_have_exact_names(self):
        router = TransitionRouter(self.banks())
        self.assertEqual({n for n, _ in router.named_buffers()},
                         set(BANK_FIELDS))

    def test_loo_removes_current_train_label(self):
        router = TransitionRouter(self.banks())
        address = torch.tensor([2, 2, 2, 1])
        target = torch.tensor([3, 4, 0, 3])
        regular, support = router.lookup(router.word_keys, router.word_counts,
                                         router.word_totals, address, target)
        loo, loo_support = router.lookup(
            router.word_keys, router.word_counts, router.word_totals,
            address, target, leave_one_out=True)
        self.assertAlmostEqual(regular[0].item(), 2/3, places=6)
        self.assertAlmostEqual(loo[0].item(), .5, places=6)
        self.assertAlmostEqual(loo[1].item(), 0.)
        self.assertEqual(support[0], 3)
        self.assertEqual(loo_support[0], 2)

    def test_signed_pair_is_orientation_sensitive(self):
        ids = signed_pair_numpy(np.array([[1, 4], [4, 1], [2, 4]]))
        self.assertNotEqual(ids[0], ids[1])
        self.assertNotEqual(ids[0], ids[2])

    def test_default_router_exactly_preserves_parent(self):
        router = TransitionRouter(self.banks())
        lookup = torch.zeros((VOCAB, 2), dtype=torch.long)
        lookup[2] = torch.tensor([0, 4])
        q = torch.tensor([2])
        t = torch.tensor([3])
        summary = torch.ones((1, 32))
        _, _, lex, geo, _, _ = router(q, t, lookup, summary)
        self.assertTrue(torch.equal(lex, torch.zeros_like(lex)))
        self.assertTrue(torch.equal(geo, torch.zeros_like(geo)))


if __name__ == '__main__':
    unittest.main()
