#!/usr/bin/env python3
"""Source-only audit for R7.54 CPU pilot and causal book interventions."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from fa1_r754_cpu_pilot import (
    book_gate, foreign_successor_history, frozen_fingerprint, select_score)
from fa1_r753_corpus import with_successor_history


class CausalBookHistoryTests(unittest.TestCase):
    def test_independent_history_preserves_inherited_inputs_and_boundaries(self):
        class CorpusFixture:
            def __init__(self, *args, **kwargs):
                self.seq = np.asarray([9, 2, 3, 4, 5, 6])
                self.doc_start = np.asarray([0, 0, 0, 0, 4, 4])
            def get(self, positions, context):
                lags = np.arange(context, 0, -1)
                ix = positions[:, None] - lags[None, :]
                valid = ix >= self.doc_start[positions, None]
                values = self.seq[np.maximum(ix, 0)].copy()
                values[~valid] = 1
                return values, ix
        data = {"positions": np.asarray([3, 5]),
                "target": torch.tensor([4, 6]),
                "word_ids": torch.tensor([[8, 9], [2, 5]])}
        with patch("fa1_r753_corpus.Corpus", CorpusFixture):
            x = with_successor_history(data, "/unused", "dev", "/unused", 4)
        self.assertTrue(torch.equal(x["word_ids"], data["word_ids"]))
        self.assertTrue(torch.equal(x["target"], data["target"]))
        self.assertEqual(x["r753_successor_history_ids"].tolist(),
                         [[1, 9, 2, 3], [1, 1, 1, 5]])
        self.assertEqual(x["r753_successor_history_mask"].tolist(),
                         [[False, True, True, True], [False, False, False, True]])

    def test_foreign_history_never_leaks_donor_target_or_parent(self):
        data = {
            "target": torch.tensor([2, 3, 4, 5]),
            "book_ids": torch.tensor([10, 10, 20, 20]),
            "word_ids": torch.tensor([[2], [3], [4], [5]]),
            "r753_successor_history_ids": torch.tensor(
                [[10, 2], [10, 3], [20, 4], [20, 5]]),
            "r753_successor_history_mask": torch.ones((4, 2), dtype=torch.bool),
        }
        negative = foreign_successor_history(data, 754)
        self.assertTrue(torch.equal(negative["target"], data["target"]))
        self.assertTrue(torch.equal(negative["word_ids"], data["word_ids"]))
        self.assertEqual(negative["r753_successor_history_ids"][:, 0].tolist(),
                         [20, 20, 10, 10])

    def test_fewer_than_two_books_fails_closed(self):
        data = {"book_ids": torch.tensor([10, 10]),
                "r753_successor_history_ids": torch.ones((2, 2)),
                "r753_successor_history_mask": torch.ones((2, 2))}
        with self.assertRaises(ValueError):
            foreign_successor_history(data, 1)


class CheckpointGates(unittest.TestCase):
    def test_only_inherited_weights_and_optimizer_slots_are_fingerprinted(self):
        p = torch.nn.Parameter(torch.tensor([1., 2.]))
        q = torch.nn.Parameter(torch.tensor([3.]))
        adam = torch.optim.AdamW([p, q])
        fake = SimpleNamespace(_adamw=adam, _legacy_state={})
        initial = frozen_fingerprint({"old": p}, fake)
        with torch.no_grad():
            q.add_(1.)
        self.assertEqual(initial, frozen_fingerprint({"old": p}, fake))
        with torch.no_grad():
            p.add_(1.)
        self.assertNotEqual(initial, frozen_fingerprint({"old": p}, fake))

    def test_book_bootstrap_requires_positive_independent_book_ci(self):
        ids = torch.tensor([1, 1, 2, 2, 3, 3])
        gain = torch.tensor([.1, .1, .2, .2, .3, .3])
        ok, info = book_gate({"gain": gain}, ("gain",), ids, 754)
        self.assertTrue(ok)
        self.assertGreater(info["gain"]["book_bootstrap_ci95"][0], 0.)
        not_ok, _ = book_gate({"gain": -gain}, ("gain",), ids, 754)
        self.assertFalse(not_ok)

    def test_selection_prefers_attribution_gates(self):
        good = {"strict_geometry_dev_gate_passed": True,
                "word_gate_passed": True, "candidate_word_nll": 5.2}
        bad = {"strict_geometry_dev_gate_passed": False,
               "word_gate_passed": True, "candidate_word_nll": 5.0}
        self.assertGreater(select_score(good), select_score(bad))


if __name__ == "__main__":
    unittest.main()
