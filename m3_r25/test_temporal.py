"""R25 strict zero-gain parity, causality, independent memory and training tests."""
import unittest

import numpy as np

from m3_default.predictor import M3DefaultPredictor
from m3_r25.temporal import M3R25Predictor, TemporalSuccessors, readout
from m3_r25.train_dev import ScalarAdam, gate_pass


class TemporalMemoryTests(unittest.TestCase):
    def test_no_prediction_before_observation_and_target_is_causal(self):
        prior = np.arange(1, 257, dtype=np.float64)
        model = M3R25Predictor(prior, np.array([0.3, 0.6, 0.2]))
        with self.assertRaises(ValueError):
            model.predict_logp()
        for byte in b"abc abc abc abc abc abc ":
            model.observe(byte)
        before = model.temporal.observations
        snapshot = [dict(row) for row in model.temporal.all_counts[2].values()]
        p1 = model.predict_logp().copy()
        p2 = model.predict_logp().copy()
        np.testing.assert_array_equal(p1, p2)
        self.assertEqual(before, model.temporal.observations)
        self.assertEqual(snapshot, [dict(row) for row in model.temporal.all_counts[2].values()])
        self.assertAlmostEqual(float(np.exp(p1).sum()), 1., places=12)
        model.observe(88)
        self.assertEqual(before + 1, model.temporal.observations)

    def test_zero_feedback_is_bitwise_canonical_r24(self):
        prior = np.arange(1, 257, dtype=np.float64)
        base = M3DefaultPredictor(prior, mode="r24")
        candidate = M3R25Predictor(prior)
        for byte in b"the whale sees a wave. the whale sees a wave! " * 2:
            base.observe(byte)
            candidate.observe(byte)
            np.testing.assert_array_equal(base.predict_logp(), candidate.predict_logp())
        candidate.weights[:] = (0.0, 0.3, 0.0)
        candidate.reset()
        candidate.observe(65)
        self.assertAlmostEqual(float(np.exp(candidate.predict_logp()).sum()), 1., places=12)

    def test_correct_successor_vs_wrong_and_expiration(self):
        memory = TemporalSuccessors(near=4, horizon=7)
        base = np.log(np.ones(256) / 256)
        for byte in b"abXabXabXabX":
            memory.observe(byte)
        memory.context.clear()
        memory.context.extend(b"ab")
        correct = memory.features(base)
        wrong = memory.features(base, wrong_values=True)
        self.assertGreater(correct[0, ord("X")], wrong[0, ord("X")])
        self.assertLessEqual(sum(sum(row.values()) for row in memory.all_counts[2].values()), 7)
        self.assertLessEqual(sum(sum(row.values()) for row in memory.near_counts[2].values()), 4)
        for b in b"zzzzzzzzzz":
            memory.observe(b)
        self.assertNotIn(b"ab", memory.all_counts[2])

    def test_independent_books_and_readout_shapes(self):
        a = TemporalSuccessors()
        b = TemporalSuccessors()
        for byte in b"mississippi mississippi":
            a.observe(byte)
        self.assertEqual(b.observations, 0)
        base = np.log(np.ones(256) / 256)
        self.assertFalse(np.any(b.features(base)))
        self.assertIs(readout(base, b.features(base), np.zeros(3)), base)
        with self.assertRaises(ValueError):
            readout(base, b.features(base), np.zeros(4))
        with self.assertRaises(ValueError):
            b.observe(256)

    def test_adam_progress_and_gate_reject(self):
        adam = ScalarAdam()
        adam.update(np.array([0.2, -0.1, 0.3]))
        self.assertEqual(adam.step, 1)
        self.assertTrue(np.isfinite(adam.theta).all())
        fake = {"arms": {"r24": {"bpb": 2.0},
                 "r25": {"gain_bpb_vs_r24": -0.01, "nonspace_gain_bpb_vs_r24": 0.2,
                          "books_improved": 4, "bpb": 2.01},
                 "wrong_successor": {"bpb": 2.6},
                 "unrelated_history": {"bpb": 2.5}}}
        self.assertFalse(gate_pass(fake, 3))


if __name__ == "__main__":
    unittest.main()
