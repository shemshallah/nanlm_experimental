"""R29 migration, causality, normalization, expiry and control tests."""
import unittest

import numpy as np

from m3_default.predictor import M3DefaultPredictor, R26_THETA
from m3_r29.lexical import (M3R29Predictor, WordSuccessorMemory,
                            LEXICAL_CHANNELS, NEW_WEIGHTS, TOTAL_WEIGHTS,
                            lexical_readout, pack_lexical)
from m3_r29.train_dev import FrozenParentAdam, gate_pass


class LexicalTests(unittest.TestCase):
    def test_exact_zero_init_parity_at_each_byte_and_after_reset(self):
        prior = np.arange(1, 257, dtype=np.float64)
        canonical = M3DefaultPredictor(prior)
        candidate = M3R29Predictor(prior)
        self.assertEqual(TOTAL_WEIGHTS, 14 + NEW_WEIGHTS)
        self.assertEqual(len(R26_THETA), 14)
        sample = b"The quick brown fox. The quick green fox. " * 3
        for byte in sample:
            canonical.observe(int(byte))
            candidate.observe(int(byte))
            np.testing.assert_array_equal(candidate.predict_logp(), canonical.predict_logp())
            np.testing.assert_array_equal(candidate.predict_logp("r26"), canonical.predict_logp())
            np.testing.assert_array_equal(candidate.predict_logp("r24"), canonical.predict_logp("r24"))
            np.testing.assert_array_equal(candidate.predict_logp("r20"), canonical.predict_logp("r20"))
        canonical.reset()
        candidate.reset()
        canonical.observe(97)
        candidate.observe(97)
        np.testing.assert_array_equal(candidate.predict_logp(), canonical.predict_logp())

    def test_repeatable_read_only_features_and_altered_successor_control(self):
        memory = WordSuccessorMemory(near_horizon=12, long_horizon=32)
        base = np.log(np.ones(256) / 256)
        for byte in b"the brown cat the brown camel the brown c":
            memory.observe(byte)
        count = memory.observed
        first = memory.features(base)
        np.testing.assert_array_equal(first, memory.features(base))
        self.assertEqual(memory.observed, count)
        self.assertEqual(first.shape, (LEXICAL_CHANNELS, 256))
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue(np.any(first))
        self.assertFalse(np.array_equal(first, memory.features(base, wrong_values=True)))
        packed = pack_lexical(first)
        self.assertEqual(packed.shape, (NEW_WEIGHTS, 256))
        self.assertTrue(np.all(packed[LEXICAL_CHANNELS:, 32] == 0))
        p = lexical_readout(base, packed, np.ones(NEW_WEIGHTS) * .2)
        self.assertAlmostEqual(np.exp(p).sum(), 1., places=12)
        np.testing.assert_array_equal(lexical_readout(base, packed, np.zeros(NEW_WEIGHTS)), base)
        for table in memory.tables:
            self.assertLessEqual(len(table.entries), table.horizon)
            self.assertEqual(sum(map(lambda r: sum(r.values()), table.rows.values())),
                             len(table.entries))

    def test_unrelated_history_uses_actual_query_context(self):
        actual = WordSuccessorMemory()
        foreign = WordSuccessorMemory()
        for byte in b"the red fox the red fox the red f":
            actual.observe(byte)
        for byte in b"the red frog the red frog the red fro":
            foreign.observe(byte)
        base = np.log(np.ones(256) / 256)
        cross = foreign.features(base, query=actual)
        self.assertEqual(cross.shape, (4, 256))
        self.assertTrue(np.any(cross))
        self.assertFalse(np.array_equal(actual.features(base), cross))

    def test_invalid_bytes_inputs_and_parent_refusal(self):
        memory = WordSuccessorMemory()
        with self.assertRaises(ValueError):
            WordSuccessorMemory(near_horizon=0)
        with self.assertRaises(ValueError):
            memory.observe(-1)
        with self.assertRaises(ValueError):
            memory.observe(256)
        with self.assertRaises(ValueError):
            memory.features(np.zeros(4))
        with self.assertRaises(ValueError):
            M3R29Predictor(np.ones(256), np.zeros(7))
        with self.assertRaises(ValueError):
            FrozenParentAdam(__import__("pathlib").Path("no-R26-checkpoint.npz"))

    def test_gate_rejects_accidental_metric_regressions(self):
        baseline = dict(bpb=3.25, nonspace_bpb=4.0, top1=.43, top5=.72)
        good = dict(bpb=3.24, nonspace_bpb=3.99, top1=.44, top5=.73,
                    books_improved=4)
        measured = {"arms": {"r26": baseline, "r29": good,
                              "wrong_successor": dict(bpb=3.5),
                              "unrelated_history": dict(bpb=3.6)}}
        self.assertTrue(gate_pass(measured, minimum_improved=3))
        measured["arms"]["r29"]["top5"] = .71
        self.assertFalse(gate_pass(measured, minimum_improved=3))


if __name__ == "__main__":
    unittest.main()
