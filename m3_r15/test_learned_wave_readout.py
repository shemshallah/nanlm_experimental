"""Predictive readout numerical and leakage tests."""

import unittest

import numpy as np

from m3_r13.mirror_wave import M3MirrorWave
from m3_r15.learned_wave_readout import ARM_CONFIGS, objective, fit_readout, evaluate, wave_features


class R15Tests(unittest.TestCase):
    def test_wave_channels_are_causal_and_directional(self):
        w = M3MirrorWave(ARM_CONFIGS["spatial_wave"])
        before = wave_features(w, 65)
        w.inject_transition(65, 66)
        after = wave_features(w, 65)
        self.assertEqual(after.shape, (256, 5))
        self.assertFalse(np.allclose(before, after))
        self.assertEqual(w.ticks, 0)

    def test_convex_objective_gradient_and_hessian(self):
        rng = np.random.default_rng(20261116)
        p = rng.uniform(size=(12, 256)); p /= p.sum(axis=1, keepdims=True)
        logp = np.log(p)
        f = rng.normal(size=(12, 256, 5))
        targets = rng.integers(0, 256, size=12)
        weights = rng.normal(size=5) * .1
        loss, grad, hess = objective(logp, f, targets, weights, .2)
        eps = 1e-5
        for k in range(5):
            perturb = np.eye(5)[k] * eps
            positive = objective(logp, f, targets, weights + perturb, .2)[0]
            negative = objective(logp, f, targets, weights - perturb, .2)[0]
            self.assertAlmostEqual(grad[k], (positive - negative) / (2 * eps), places=5)
        np.testing.assert_allclose(hess, hess.T, atol=1e-10)
        self.assertGreater(np.linalg.eigvalsh(hess).min(), 0)
        self.assertTrue(np.isfinite(loss))

    def test_fitting_decreases_training_objective(self):
        rng = np.random.default_rng(13)
        p = rng.uniform(size=(48, 256)); p /= p.sum(axis=1, keepdims=True)
        f = rng.normal(size=(48, 256, 5)).astype(np.float32)
        data = {
            "base_logp": np.log(p),
            "targets": rng.integers(0, 256, size=48),
            "features": {"spatial_wave": f},
        }
        fitted = fit_readout(data, "spatial_wave")
        self.assertLessEqual(fitted["objective"][-1], fitted["objective"][0] + 1e-9)

    def test_evaluate_frozen_on_disjoint_books(self):
        train = [b"alpha beta gamma delta " * 7, b"a little owl flew by " * 7]
        dev = [b"alpha gamma beta delta " * 7, b"the little bird flew past " * 7]
        r = evaluate(train, dev, warmup=64)
        self.assertIn("memory_only", r["dev"])
        self.assertTrue(np.isfinite(r["dev"]["spatial_wave"]["bpb"]))
        self.assertEqual(len(r["dev"]["reciprocal_wave"]["fitted_weights"]), 5)
        self.assertFalse(r["r12_checkpoint_loaded"])


if __name__ == "__main__":
    unittest.main()
