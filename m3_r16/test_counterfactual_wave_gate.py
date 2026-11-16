"""M3 R16 invariants: exact fallback, book isolation and honest attribution."""

import unittest

import numpy as np

from m3_r16.counterfactual_wave_gate import (
    GEOMETRY_ARMS, evaluate, predict_pair, residual_features, whole_book_deltas,
)
from m3_r15.learned_wave_readout import normalized_logits


class CounterfactualTests(unittest.TestCase):
    def test_residual_is_exact_difference(self):
        x = np.arange(20, dtype=np.float32).reshape(2, 2, 5)
        data = {"features": {"uncoupled": x, "spatial_wave": x + 3}}
        np.testing.assert_array_equal(residual_features(data, "spatial_wave"), np.full_like(x, 3))

    def test_gate_zero_is_function_preserving(self):
        rng = np.random.default_rng(17)
        prior = rng.uniform(size=(2, 256)); prior /= prior.sum(axis=1, keepdims=True)
        x = rng.normal(size=(2, 256, 5)).astype(np.float32)
        data = {"base_logp": np.log(prior), "features": {"uncoupled": x}}
        weights = rng.normal(size=5)
        model = {"oscillator": {"weights": weights}, "specialist": {"weights": np.ones(5)}}
        for arm in GEOMETRY_ARMS:
            np.testing.assert_array_equal(
                predict_pair(data, arm, model, 0), normalized_logits(data["base_logp"], x, weights)
            )

    def test_bootstrap_per_book_conservation(self):
        books = [b"a" * 40, b"b" * 50, b"c" * 60]
        n = sum(len(b) - 10 for b in books)
        out = whole_book_deltas(np.full(n, 3.0), np.full(n, 2.5), books, warmup=10)
        self.assertEqual(out["books_improved"], 3)
        self.assertAlmostEqual(out["whole_book_bootstrap_95pct_bits"][0], 0.5)

    def test_small_book_disjoint_end_to_end(self):
        train = [(b"alpha beta gamma delta " * 6), (b"owl sings beside the river " * 6),
                 (b"beta delta alpha gamma " * 6), (b"stars shine above the city " * 6)]
        dev = [(b"gamma delta alpha beta " * 6), (b"owl flies away in the rain " * 6)]
        result = evaluate(train, dev, warmup=64)
        self.assertGreater(result["dev_positions"], 100)
        for arm in GEOMETRY_ARMS:
            key = "oscillator_plus_" + arm
            self.assertTrue(np.isfinite(result["dev"][key]["bpb"]))
            if result["frozen_scales"][arm] == 0:
                self.assertEqual(result["dev"][key]["delta_bpb_vs_oscillator"], 0)


if __name__ == "__main__":
    unittest.main()
