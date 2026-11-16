"""M3 canonical online inference must match the frozen R20 batch equations."""
import unittest

import numpy as np

from m3_default.predictor import M3DefaultPredictor, DEFAULT_MODE
from m3_r14.predictive_probe import train_unigram_prior
from m3_r15.learned_wave_readout import collect
from m3_r17.locked_test import frozen_predictions
from m3_r19.class_conditional_wave import scaled_predictions, residual_scores
from m3_r20.signed_mirror_memory import causal_successor_deltas
from m3_r20.successor_dev import combined_logp


class M3StreamingParity(unittest.TestCase):
    def test_default_is_r20_and_reset_is_reproducible(self):
        p = M3DefaultPredictor(np.ones(256))
        self.assertEqual(DEFAULT_MODE, "r20")
        with self.assertRaises(ValueError):
            p.predict_proba()
        p.observe(65)
        first = p.predict_proba()
        self.assertAlmostEqual(float(first.sum()), 1.0, places=12)
        p.observe(66)
        p.reset()
        p.observe(65)
        np.testing.assert_array_equal(first, p.predict_proba())

    def test_read_does_not_train_or_mutate_state(self):
        p = M3DefaultPredictor(np.ones(256))
        for b in b"ab abracadabra ab":
            p.observe(b)
        before = p.signed.invariants()
        waves = {key: v.pair_u.copy() for key, v in p.waves.items()}
        a = p.predict_all()
        b = p.predict_all()
        for mode in a:
            np.testing.assert_array_equal(a[mode], b[mode])
            self.assertAlmostEqual(float(np.exp(a[mode]).sum()), 1.0, places=12)
        self.assertEqual(before, p.signed.invariants())
        for key, value in waves.items():
            np.testing.assert_array_equal(value, p.waves[key].pair_u)

    def test_streaming_matches_original_r15_wave_and_r20_batch(self):
        book = (b"abc abd abc abf abc abg abc abf abf abc "
                b"the quick brown fox jumps over the lazy dog ")
        prior = train_unigram_prior([book])
        warmup = 12
        batch = collect([book], prior, warmup)
        carrier, _ = frozen_predictions(batch)
        r19 = scaled_predictions(carrier, residual_scores(batch), .25, 1.0)
        pair, tri, inv = causal_successor_deltas(book, prior, batch["base_logp"], warmup)
        expected = combined_logp(r19, pair, tri, .5, .5)
        streaming = M3DefaultPredictor(prior)
        streaming.observe(book[0])
        actual = []
        for pos in range(1, len(book)):
            if pos >= warmup:
                actual.append(streaming.predict_logp())
            streaming.observe(book[pos])
        self.assertEqual(len(actual), len(expected))
        np.testing.assert_allclose(np.asarray(actual), expected,
                                   rtol=0.0, atol=2e-10)
        self.assertEqual(inv["edge_mass"], len(book)-1)

    def test_exact_r19_fallback_is_available_without_state_change(self):
        p = M3DefaultPredictor(np.ones(256), mode="r19")
        for byte in b"the same causal prefix":
            p.observe(byte)
        before = p.signed.invariants()
        self.assertIsInstance(p.predict_logp(), np.ndarray)
        np.testing.assert_array_equal(p.predict_logp("r19"), p.predict_all()["r19"])
        self.assertEqual(before, p.signed.invariants())


if __name__ == "__main__":
    unittest.main()
