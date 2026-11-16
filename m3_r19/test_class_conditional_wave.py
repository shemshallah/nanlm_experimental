"""R19 synthetic controls: no NaNROUTER TEST document is opened."""
import json
import unittest
from unittest.mock import patch

import numpy as np

from m3_r19 import class_conditional_wave as m


class R19Tests(unittest.TestCase):
    def test_split_never_reuses_r17_or_r18_test_for_fit_or_dev(self):
        self.assertEqual(m.PRIOR_SEALED, 52)
        self.assertEqual(m.TOTAL_BOOKS, 84)
        self.assertEqual((m.CALIBRATION_BOOKS, m.GATE_BOOKS, m.DEV_BOOKS), (16, 4, 12))
        self.assertEqual(m.SCALES[0], 0.0)
        manifest = json.loads(m.MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["r18_test_sha256"]), 32)

    def test_zero_scale_is_exact_and_log_probs_normalize(self):
        rng = np.random.default_rng(19)
        logits = rng.normal(0, 1, size=(5, 256))
        logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        residual = rng.normal(0, .2, size=(5, 256))
        self.assertIs(m.scaled_predictions(logp, residual, 0.0, 0.0), logp)
        for space, nonspace in ((1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (.25, .5)):
            updated = m.scaled_predictions(logp, residual, space, nonspace)
            np.testing.assert_allclose(np.exp(updated).sum(axis=1), 1, atol=1e-12)

    def test_full_scale_matches_frozen_spatial_wave(self):
        rng = np.random.default_rng(303)
        data = {
            "base_logp": np.full((3, 256), -np.log(256)),
            "features": {
                "uncoupled": rng.normal(size=(3, 256, 5)).astype(np.float32),
                "spatial_wave": rng.normal(size=(3, 256, 5)).astype(np.float32),
            }
        }
        carrier, full = m.frozen_predictions(data)
        np.testing.assert_allclose(
            m.scaled_predictions(carrier, m.residual_scores(data), 1.0, 1.0),
            full, atol=2e-14
        )

    def test_train_only_gate_chooses_nonspace_safe_candidate(self):
        fit = []
        gate = []
        for s in m.SCALES:
            for n in m.SCALES:
                good = (s, n) == (1.0, 0.0)
                bad = (s, n) == (1.0, 1.0)
                fit.append({
                    "space_scale": s, "nonspace_scale": n,
                    "bpb": 3.98 if good else (3.95 if bad else 4.0),
                    "nonspace_bpb": 4.49 if good else (4.51 if bad else 4.5),
                    "gain_vs_oscillator": .02 if good else (.05 if bad else 0),
                    "nonspace_gain_vs_oscillator": .01 if good else (-.01 if bad else 0),
                })
                gate.append({
                    "space_scale": s, "nonspace_scale": n,
                    "bpb": 3.97 if good else 4.0,
                    "nonspace_bpb": 4.49 if good else 4.5,
                    "gain_vs_oscillator": .03 if good else 0,
                    "nonspace_gain_vs_oscillator": .01 if good else 0,
                })
        chosen = m.choose_gate(fit, gate)
        self.assertEqual((chosen["space_scale"], chosen["nonspace_scale"]), (1.0, 0.0))
        self.assertIn([0.0, 0.0], chosen["fit_shortlist"])

    def test_streamed_dev_zero_residual_falls_back_to_carrier(self):
        b = b"aa aaaaa"
        fake = {
            "base_logp": np.full((4, 256), -np.log(256)),
            "targets": np.array([32, 97, 97, 32], dtype=np.int64),
            "nonspace": np.array([False, True, True, False]),
            "unicode_chars": 4, "elapsed_s": 1.0,
            "features": {
                "uncoupled": np.zeros((4, 256, 5), dtype=np.float32),
                "spatial_wave": np.zeros((4, 256, 5), dtype=np.float32),
            },
        }
        # Small synthetic warmup only for this unit test; production uses 1024.
        with patch.object(m, "collect", return_value=fake), patch.object(m, "WARMUP", 4):
            scored = m.evaluate_dev([b] * 12, np.full(256, 1 / 256),
                                    (1.0, 0.0), [f"fixture-{i}" for i in range(12)])
        self.assertEqual(scored["dev_positions"], 48)
        self.assertEqual(scored["dev"]["oscillator"]["bpb"], 8.0)
        self.assertAlmostEqual(scored["dev"]["train_selected"]["gain_bpb_vs_oscillator"], 0.0, places=12)
        self.assertFalse(scored["dev"]["train_selected"]["dev_gate_pass"])


if __name__ == "__main__":
    unittest.main()
