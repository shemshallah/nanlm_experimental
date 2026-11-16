"""Causality, matched-controls, and metric regression tests for M3 R14."""

import unittest

import numpy as np

from m3_r13.mirror_wave import M3MirrorWave
from m3_r14.predictive_probe import (
    ARM_CONFIGS,
    ExactWindowMemory,
    apply_gain,
    evaluate_books,
    train_unigram_prior,
    wave_score,
)


class PredictiveProbeTests(unittest.TestCase):
    def test_two_memory_windows_expire_exactly(self):
        m = ExactWindowMemory(np.ones(256) / 256)
        for i in range(2200):
            m.observe(i % 256, (i + 1) % 256)
        self.assertEqual(int(m.near_count.sum()), 128)
        self.assertEqual(int(m.old_count.sum()), 2048)
        self.assertAlmostEqual(m.predict(0).sum(), 1)
        self.assertTrue(np.all(m.near_count >= 0))
        self.assertTrue(np.all(m.old_count >= 0))

    def test_source_not_updated_before_target(self):
        m = ExactWindowMemory(np.ones(256) / 256)
        before = m.predict(ord("a")).copy()
        m.observe(ord("a"), ord("b"))
        after = m.predict(ord("a"))
        self.assertNotEqual(before[ord("b")], after[ord("b")])

    def test_directional_wave_score_and_target_independence(self):
        w = M3MirrorWave(ARM_CONFIGS["local_wave"])
        forward_before = wave_score(w, 65)
        w.inject_transition(65, 66)
        self.assertEqual(wave_score(w, 65).shape, (256,))
        self.assertNotEqual(wave_score(w, 65)[66], forward_before[66])
        self.assertEqual(w.parity_error(), 0)

    def test_identity_gain_and_normalized_distribution(self):
        prior = train_unigram_prior([b"all the green water in the city"])
        base = np.log(prior)[None, :]
        score = np.arange(256)[None, :] / 256
        np.testing.assert_allclose(apply_gain(base, score, 0), base, atol=1e-12)
        self.assertAlmostEqual(float(np.exp(apply_gain(base, score, 0.5)).sum()), 1)

    def test_end_to_end_book_disjoint_pilot_is_finite(self):
        train = [(b"alpha beta gamma delta  " * 9), (b"little bird sat beside the tree  " * 6)]
        dev = [(b"gamma alpha beta gamma  " * 9), (b"little bird moved toward the sea  " * 6)]
        result = evaluate_books(train, dev, warmup=80)
        self.assertEqual(set(result["dev"]), {"memory_only", *ARM_CONFIGS})
        self.assertTrue(np.isfinite(result["dev"]["memory_only"]["bpb"]))
        self.assertGreater(result["dev_positions"], 0)
        self.assertTrue(set(result["frozen_gain"].values()).issubset({-0.8,-0.4,-0.2,-0.1,0,0.1,0.2,0.4,0.8}))


if __name__ == "__main__":
    unittest.main()
