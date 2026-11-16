"""R49 causal history displacement and reciprocal-only differential tests."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from poincare_frontier import Config, MirrorLM
from probe_r49_causal_history import (book_bootstrap, paired_inputs,
                                      paired_records, score_books)


class CausalHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @staticmethod
    def data():
        rng = np.random.default_rng(49)
        books = [rng.integers(0, 256, 80, dtype=np.uint16) for _ in range(3)]
        for book in books:
            book[-1] = 256
        return np.concatenate(books), [80] * 3

    def test_disjoint_donors_and_recent_true_suffix(self):
        mem, lengths = self.data()
        groups, total = paired_records(lengths, 24, 8, 5, 49)
        self.assertEqual(total, len(mem))
        self.assertEqual(groups, paired_records(lengths, 24, 8, 5, 49)[0])
        for i, group in enumerate(groups):
            for book, start, donor_book, donor_start in group:
                self.assertEqual(i, book)
                self.assertNotEqual(book, donor_book)
                self.assertTrue(i * 80 <= start < (i + 1) * 80 - 26)
                self.assertTrue(donor_book * 80 <= donor_start < (donor_book + 1) * 80 - 16)
                true, wrong, target = paired_inputs(mem, start, donor_start, 24, 8)
                self.assertTrue(np.array_equal(true[-8:], wrong[-8:]))
                self.assertEqual(target, int(mem[start + 24]))
                self.assertFalse(np.array_equal(true[:16], wrong[:16]))

    def test_target_is_never_in_visible_input(self):
        mem, lengths = self.data()
        groups, _ = paired_records(lengths, 24, 8, 1, 13)
        _, pos, _, donor = groups[0][0]
        genuine, counter, target = paired_inputs(mem, pos, donor, 24, 8)
        changed = mem.copy()
        changed[pos + 24] = (target + 1) % 256
        g2, c2, t2 = paired_inputs(changed, pos, donor, 24, 8)
        self.assertTrue(np.array_equal(genuine, g2))
        self.assertTrue(np.array_equal(counter, c2))
        self.assertNotEqual(target, t2)

    def test_book_bootstrap_and_family_adjustment(self):
        values = [.01, -.03, .04, .005, .03, -.02, .015, .02, .001, .05, -.01, .02]
        ordinary = book_bootstrap(values, 2000, 49, 1)
        adjusted = book_bootstrap(values, 2000, 49, 4)
        self.assertEqual(ordinary['books_positive'], 9)
        self.assertLessEqual(adjusted['ci95_bonferroni'][0],
                             ordinary['ci95_bonferroni'][0])
        self.assertGreaterEqual(adjusted['ci95_bonferroni'][1],
                                ordinary['ci95_bonferroni'][1])
        with self.assertRaises(ValueError):
            book_bootstrap([.1])

    def test_fixed_near_must_be_proper_context(self):
        _, lengths = self.data()
        for near in (0, 24, -1):
            with self.assertRaises(ValueError):
                paired_records(lengths, 24, near, 2, 49)

    def test_tiny_real_model_returns_finite_matched_scores(self):
        mem, lengths = self.data()
        groups, _ = paired_records(lengths, 12, 4, 2, 49)
        torch.manual_seed(49)
        cfg = Config(vocab_size=257, dim=32, layers=2, heads=4,
                     pairs=4, topk_pairs=2, expert_hidden=16, context=12,
                     moe_every=2, symbiotic_rank=8, hierarchy_rank=8,
                     micro_meta=True, simplex_transfer=True,
                     reciprocal_scales=True)
        model = MirrorLM(cfg).eval()
        rows = score_books(model, mem, groups, 12, 4, batch_size=2)
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertTrue(np.isfinite(list(row.values())).all())
            self.assertAlmostEqual(row['correct_history_advantage'],
                                   row['ce_wrong'] - row['ce_true'])
            self.assertTrue(0 <= row['top5_true'] <= 1)


if __name__ == '__main__':
    unittest.main()
