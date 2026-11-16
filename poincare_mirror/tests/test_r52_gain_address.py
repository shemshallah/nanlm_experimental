"""R52 regressions: reversible gain, paired controls, and exact state recovery."""
import math
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from probe_r52_gain_address import (GAIN_LEVELS, build_report, conditions,
                                    paired_rows, reversible_cold_gain)
from geometry_address_controls import fixed_permutations
from poincare_frontier import SparseSimplexTransfer
from compare_geometry_books import METRICS
from unittest.mock import patch


class FakeReciprocal(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(-0.000367079745))
        self.enabled = True


class FakeGeometry(nn.Module):
    def __init__(self):
        super().__init__()
        self.simplex = SparseSimplexTransfer(8)
        self.triangle_table = nn.Embedding(8192, 8)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.reciprocal = FakeReciprocal()
        self.geometry = FakeGeometry()


def artificial_metrics(ce):
    return {name: (ce if name.startswith('ce_') else .5)
            for name in METRICS}


class GainAddressTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()

    def test_predeclared_controls_are_paired_and_nonduplicate(self):
        controls = conditions()
        self.assertEqual(len(controls), 3 + 2*len(GAIN_LEVELS))
        self.assertEqual(len({label for label, _, _ in controls}), len(controls))
        self.assertEqual(controls[0], ('inherited_exact', None, 'exact'))
        self.assertEqual(controls[1][2], 'all_scrambled')
        self.assertEqual(controls[2][2], 'reciprocal_disabled')
        for i, expected in enumerate(GAIN_LEVELS):
            self.assertEqual(controls[3+2*i][1:], (expected, 'exact'))
            self.assertEqual(controls[4+2*i][1:], (expected, 'all_scrambled'))

    def test_gain_uses_effective_tanh_parameterization_and_restores_bits(self):
        old = self.model.reciprocal.gate.detach().clone()
        for expected in (*GAIN_LEVELS, None):
            with reversible_cold_gain(self.model, expected) as actual:
                self.assertAlmostEqual(actual, math.tanh(8*float(
                    self.model.reciprocal.gate)), places=7)
                if expected is not None:
                    self.assertAlmostEqual(actual, expected, places=7)
            self.assertTrue(torch.equal(old, self.model.reciprocal.gate))

    def test_invalid_and_exceptional_overrides_cannot_change_checkpoint(self):
        before = self.model.reciprocal.gate.detach().clone()
        for bad in (float('nan'), float('inf'), 1., -1., 2.):
            with self.assertRaises(ValueError):
                with reversible_cold_gain(self.model, bad):
                    pass
            self.assertTrue(torch.equal(before, self.model.reciprocal.gate))
        with self.assertRaisesRegex(RuntimeError, 'forced'):
            with reversible_cold_gain(self.model, .1):
                raise RuntimeError('forced')
        self.assertTrue(torch.equal(before, self.model.reciprocal.gate))

    def test_full_intervention_replays_and_restores_embeddings_and_buffers(self):
        before = {name: x.clone() for name, x in self.model.state_dict().items()}
        permutations = fixed_permutations(2026)
        def mocked_measure(model, _mem, _records, _context, _batch):
            gain = float(torch.tanh(8*model.reciprocal.gate).detach())
            # Exercise hooks and broken incidence using an ordinary embedding.
            edge = model.geometry.simplex.pair_shared(torch.tensor([1]))
            inc = model.geometry.simplex.d1[0, 0].item()
            ce = (3. if not model.reciprocal.enabled else 2. + gain +
                  float(edge.sum())*.000001 + inc*.00001)
            return [artificial_metrics(ce), artificial_metrics(ce+.1)]
        with patch('probe_r52_gain_address.measure_books', mocked_measure):
            rows, _, used = paired_rows(self.model, None, [[], []], 8, 1,
                                        permutations)
        self.assertEqual(len(rows), len(conditions()))
        self.assertAlmostEqual(used['gain_3_exact'], .03, places=6)
        self.assertFalse(self.model.geometry.simplex.pair_shared._forward_pre_hooks)
        self.assertTrue(self.model.reciprocal.enabled)
        for key, value in before.items():
            self.assertTrue(torch.equal(self.model.state_dict()[key], value), key)

    def test_address_contrast_sign_and_bootstrap_are_interpretable(self):
        rows = {}
        elapsed = {}
        gains = {}
        for label, gain, mode in conditions():
            ce = 2.0 + (.02 if mode == 'all_scrambled' else 0.)
            rows[label] = [artificial_metrics(ce+.001*i) for i in range(12)]
            elapsed[label], gains[label] = 1.0, gain if gain is not None else -.0029
        outcome, specificity = build_report(rows, elapsed, gains, 6144, 6144,
                                            200, 2026)
        self.assertEqual(len(outcome), len(conditions()))
        self.assertAlmostEqual(specificity['inherited_exact']['ce_1'][
            'exact_advantage'], .02, places=6)
        self.assertGreater(specificity['inherited_exact']['ce_1'][
            'ci95_bonferroni'][0], 0)
        self.assertEqual(specificity['inherited_exact']['ce_1'][
            'books_exact_better'], 12)


if __name__ == '__main__':
    unittest.main()
