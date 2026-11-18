#!/usr/bin/env python3
"""Promotion is fail-closed; never asserts external TEST superiority."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from fa1_default import DEFAULT_ARCHITECTURE_ROUND, load_default
from fa1_r764_promotion import check_dev_gates
from fa1_r764_multiresolution import ROUND


def positive_metrics():
    bb = lambda lo: {'book_bootstrap_ci95': [lo, lo + .003]}
    return {
        'strict_geometry_gate_passed': True,
        'hierarchy_gain': .005,
        'hierarchy_bootstrap': bb(.001),
        'geometry_over_lexical_matched': .005,
        'correct_vs_wrong_older': .006,
        'correct_vs_wrong_geometry_values': .011,
        'book_bootstrap': {
            'geometry_over_lexical_matched': bb(.0004),
            'correct_vs_wrong_older': bb(.001),
            'correct_vs_wrong_geometry_values': bb(.007)}}


class PromotionTests(unittest.TestCase):
    def test_positive_book_clustered_gates(self):
        check_dev_gates(positive_metrics())
        self.assertEqual(DEFAULT_ARCHITECTURE_ROUND, ROUND)

    def test_refuse_negative_history_or_lexical_interval(self):
        for field in ('geometry_over_lexical_matched',
                      'correct_vs_wrong_older',
                      'correct_vs_wrong_geometry_values'):
            m = positive_metrics()
            m['book_bootstrap'][field]['book_bootstrap_ci95'][0] = -.0001
            with self.assertRaises(ValueError):
                check_dev_gates(m)

    def test_refuse_missing_hierarchy_gain(self):
        m = positive_metrics()
        m['hierarchy_bootstrap']['book_bootstrap_ci95'][0] = 0.
        with self.assertRaises(ValueError):
            check_dev_gates(m)

    def test_default_rejects_any_unpromoted_checkpoint(self):
        with patch('fa1_default.load_unified',
                   return_value=(None, SimpleNamespace(param_groups=[1] * 37),
                                 None, None,
                                 {'architecture_round': ROUND,
                                  'promoted': False})):
            with self.assertRaises(ValueError):
                load_default('/unused', '/unused/candidate.pt')


if __name__ == '__main__':
    unittest.main()
