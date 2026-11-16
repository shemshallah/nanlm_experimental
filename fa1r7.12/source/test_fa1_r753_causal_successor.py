#!/usr/bin/env python3
"""R7.53 source-only causal and probability contract regressions."""
from __future__ import annotations

import unittest
import numpy as np
import torch

from fa1_r748_mirror_pair_values import pair_indices
from fa1_r753_causal_successor import (
    causal_successor_evidence, compose_inherited_distribution, ROUND)
from fa1_r753_corpus import balanced_causal_positions


class SuccessorContracts(unittest.TestCase):
    def setUp(self):
        self.lookup = torch.tensor([[i, 100 + i] for i in range(24)])
        for i in (2, 6, 10, 14):
            self.lookup[i] = torch.tensor([17, 200])
        self.lookup[18] = torch.tensor([200, 17])
        self.history = torch.tensor([[6, 8, 2, 3, 10, 5, 2, 3,
                                      14, 7, 18, 4, 2]])
        self.mask = torch.ones_like(self.history, dtype=torch.bool)

    def evidence(self, y, mode="full", history=None, mask=None):
        return causal_successor_evidence(
            self.history if history is None else history,
            self.mask if mask is None else mask,
            self.lookup, torch.tensor([y]), mode)

    def test_pair_count_and_mirror(self):
        a, valid = pair_indices(torch.tensor([0, 254, 17]),
                                torch.tensor([1, 255, 200]))
        self.assertEqual(a.tolist(), [0, 32639, 4381])
        self.assertTrue(valid.all().item())
        b, _ = pair_indices(torch.tensor([200]), torch.tensor([17]))
        self.assertEqual(int(b[0]), int(a[2]))

    def test_correct_word_and_signed_support(self):
        qw, qg, wc, gc, distinct, eligible = self.evidence(3)
        self.assertAlmostEqual(float(qw[0]), 1.)
        self.assertAlmostEqual(float(qg[0]), 0.)
        self.assertEqual((int(wc[0]), int(gc[0]), int(distinct[0])),
                         (2, 3, 3))
        self.assertTrue(bool(eligible[0]))

    def test_nonlexical_successor_distribution(self):
        p = [float(self.evidence(i)[1][0]) for i in range(24)]
        self.assertAlmostEqual(sum(p), 1., places=6)
        self.assertGreater(p[8], 0.)
        self.assertGreater(p[5], 0.)
        self.assertGreater(p[7], 0.)
        self.assertEqual(p[4], 0.)  # Only reverse-oriented 18->4.

    def test_wrong_address_only_scrambles_keys(self):
        correct = self.evidence(8)
        wrong = self.evidence(8, "wrong_addresses")
        self.assertEqual(int(correct[2][0]), int(wrong[2][0]))
        self.assertLess(int(wrong[3][0]), int(correct[3][0]))

    def test_wrong_successor_only_scrambles_values(self):
        correct = self.evidence(8)
        wrong = self.evidence(8, "wrong_successors")
        self.assertEqual(int(correct[3][0]), int(wrong[3][0]))
        self.assertNotEqual(float(correct[1][0]), float(wrong[1][0]))

    def test_masked_cross_book_tokens_ignored(self):
        h = torch.cat((torch.tensor([[11, 2, 7]]), self.history), dim=1)
        m = torch.cat((torch.zeros((1, 3), dtype=torch.bool), self.mask), dim=1)
        other = torch.cat((torch.tensor([[5, 9, 11]]), self.history), dim=1)
        expected = self.evidence(8, history=h, mask=m)
        got = self.evidence(8, history=other, mask=m)
        for a, b in zip(expected[:5], got[:5]):
            self.assertTrue(torch.allclose(a.float(), b.float()))

    def test_short_history_fallback(self):
        history = torch.tensor([[1, 3, 4, 2]])
        mask = torch.tensor([[False, True, True, True]])
        _, _, wc, gc, distinct, eligible = self.evidence(
            3, history=history, mask=mask)
        self.assertEqual(int(wc[0]), 0)
        self.assertFalse(bool(eligible[0]))
        self.assertEqual(int(gc[0]), 0)
        self.assertEqual(int(distinct[0]), 0)

    def test_empty_history_is_rejected(self):
        with self.assertRaises(ValueError):
            self.evidence(3, history=torch.tensor([[2]]),
                          mask=torch.tensor([[True]]))

    def test_distribution_composition_and_normalization(self):
        torch.manual_seed(753)
        for _ in range(64):
            base = torch.softmax(torch.randn(24, dtype=torch.float64), 0)
            old_q = torch.softmax(torch.randn(24, dtype=torch.float64), 0)
            word_q = torch.softmax(torch.randn(24, dtype=torch.float64), 0)
            geo_q = torch.softmax(torch.randn(24, dtype=torch.float64), 0)
            alpha0 = torch.tensor(.12, dtype=torch.float64)
            w = torch.tensor(.08, dtype=torch.float64)
            g = torch.tensor(.04, dtype=torch.float64)
            new_a, new_q = compose_inherited_distribution(
                alpha0, old_q, word_q, geo_q, w, g)
            observed = (1 - new_a) * base + new_a * new_q
            old = (1 - alpha0) * base + alpha0 * old_q
            direct = (1 - w - g) * old + w * word_q + g * geo_q
            self.assertTrue(torch.allclose(observed, direct, atol=1.e-14))
            self.assertAlmostEqual(float(observed.sum()), 1., places=12)

    def test_zero_gain_preserves_original_objects(self):
        alpha = torch.tensor([.125])
        prior = torch.tensor([.25])
        a, q = compose_inherited_distribution(
            alpha, prior, torch.ones(1), torch.zeros(1),
            torch.zeros(1), torch.zeros(1))
        self.assertIs(a, alpha)
        self.assertIs(q, prior)

    def test_invalid_mixture_rejected(self):
        with self.assertRaises(ValueError):
            compose_inherited_distribution(
                torch.tensor([.1]), torch.tensor([.5]),
                torch.tensor([.5]), torch.tensor([.5]),
                torch.tensor([.2]), torch.tensor([.2]))

    def test_version_records_direct_parent(self):
        self.assertIn("R7_53", ROUND)


class BalancedSampler(unittest.TestCase):
    def test_all_books_including_short_and_strict_bounds(self):
        lengths = [9, 16, 280, 50000, 5]
        doc_start = []
        starts = []
        offset = 0
        for length in lengths:
            starts.append(offset)
            offset += length
        doc_start = []
        offset = 0
        for length in lengths:
            doc_start.extend([offset] * length)
            offset += length
        FakeCorpus = type("FakeCorpus", (), {
            "seq": np.arange(sum(lengths)),
            "doc_start": np.asarray(doc_start, dtype="<u4")})
        positions = balanced_causal_positions(FakeCorpus(), 32, 753)
        observed = set(int(FakeCorpus.doc_start[p]) for p in positions)
        self.assertEqual(observed, set(starts))
        for p in positions:
            first = int(FakeCorpus.doc_start[p])
            self.assertGreater(p, first)
            self.assertLess(p, len(FakeCorpus.seq))

    def test_short_sampling_reproducible(self):
        class FakeCorpus:
            seq = np.arange(32)
            doc_start = np.asarray([0]*5+[5]*7+[12]*20)
        a = balanced_causal_positions(FakeCorpus(), 40, 753)
        b = balanced_causal_positions(FakeCorpus(), 40, 753)
        self.assertTrue(np.array_equal(a, b))


if __name__ == "__main__":
    unittest.main()
