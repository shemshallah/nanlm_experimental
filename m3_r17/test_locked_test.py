"""Pre-TEST unit checks; do not open NaNROUTER holdout books locally."""

import hashlib
import unittest
from unittest.mock import patch

import numpy as np

from m3_r15.learned_wave_readout import normalized_logits
from m3_r17 import locked_test as m


class LockedR16Test(unittest.TestCase):
    def test_weights_and_gate_frozen(self):
        self.assertEqual(m.ARM, "spatial_wave")
        self.assertEqual(m.GATE, 1.0)
        self.assertEqual(m.R16_OSCILLATOR_WEIGHTS.shape, (5,))
        self.assertEqual(m.R16_SPATIAL_RESIDUAL_WEIGHTS.shape, (5,))
        self.assertTrue(np.all(np.isfinite(m.R16_SPATIAL_RESIDUAL_WEIGHTS)))

    def test_predictions_use_graph_only_difference(self):
        rng = np.random.default_rng(101)
        lp = np.log(np.full((3, 256), 1 / 256))
        uncoupled = rng.normal(0, 0.5, size=(3, 256, 5))
        data = {"base_logp": lp, "features": {
            "uncoupled": uncoupled,
            "spatial_wave": uncoupled.copy(),
        }}
        carrier, spatial = m.frozen_predictions(data)
        np.testing.assert_array_equal(carrier, spatial)
        data["features"]["spatial_wave"][0, 2, 2] += 4
        _, changed = m.frozen_predictions(data)
        self.assertGreater(float(np.abs(changed - spatial).max()), 1e-5)
        np.testing.assert_allclose(np.exp(changed).sum(axis=1), 1.0, atol=1e-12)

    def test_locked_predict_matches_manual_logits(self):
        rng = np.random.default_rng(9)
        data = {"base_logp": np.log(np.full((2, 256), 1/256)), "features": {
            "uncoupled": rng.standard_normal((2,256,5)),
            "spatial_wave": rng.standard_normal((2,256,5)),
        }}
        a, b = m.frozen_predictions(data)
        expected_a = normalized_logits(data["base_logp"], data["features"]["uncoupled"], m.R16_OSCILLATOR_WEIGHTS)
        expected_b = normalized_logits(expected_a, data["features"]["spatial_wave"] - data["features"]["uncoupled"], m.R16_SPATIAL_RESIDUAL_WEIGHTS)
        np.testing.assert_array_equal(a, expected_a)
        np.testing.assert_array_equal(b, expected_b)

    def test_prefix_rejects_mismatch_and_duplicate(self):
        books = [f"distinct book {i}".encode() for i in range(20)]
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            m.verify_r16_prefix(books)
        with self.assertRaisesRegex(ValueError, "book count"):
            m.verify_r16_prefix(books[:-1])
        with patch.object(m, "R16_TRAIN_DEV_SHA256", tuple(hashlib.sha256(b).hexdigest() for b in books[:14])):
            self.assertEqual(len(m.verify_r16_prefix(books)), 20)
            books[19] = books[0]
            with self.assertRaisesRegex(ValueError, "overlapping"):
                m.verify_r16_prefix(books)


if __name__ == "__main__":
    unittest.main()
