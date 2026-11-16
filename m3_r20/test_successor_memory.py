"""M3 R20 causality/topology/control tests; use synthetic data only."""
import unittest
from unittest.mock import patch

import numpy as np

from m3_r20 import signed_mirror_memory as memory
from m3_r20 import successor_dev as driver


class PairGeometryTests(unittest.TestCase):
    def test_32640_unordered_addresses_cover_every_pair_once(self):
        all_ids = {memory.pair_index(a, b) for a in range(256) for b in range(a+1, 256)}
        self.assertEqual(len(all_ids), 32640)
        self.assertEqual((min(all_ids), max(all_ids)), (0, 32639))
        self.assertEqual(memory.pair_index(14, 209), memory.pair_index(209, 14))
        with self.assertRaises(ValueError):
            memory.pair_index(7, 7)

    def test_signed_rails_and_diagonal_are_distinct(self):
        m = memory.SignedMirrorSuccessorMemory(np.ones(256))
        idx = memory.pair_index(10, 20)
        m.observe(None, 10, 20)
        self.assertEqual(m.s_plus[idx], 1)
        self.assertEqual(m.s_minus[idx], 0)
        m.observe(10, 20, 10)
        self.assertEqual(m.s_plus[idx], 1)
        self.assertEqual(m.s_minus[idx], 1)
        m.observe(20, 10, 10)
        self.assertEqual(m.diagonal[10], 1)
        inv = m.invariants()
        self.assertEqual(inv["edge_mass"], 3)
        self.assertEqual(inv["triangle_mass"], 2)
        self.assertTrue(inv["pair_nonnegative"])

    def test_triangle_successor_never_enters_before_observation(self):
        m = memory.SignedMirrorSuccessorMemory(np.ones(256))
        before_pair, before_tri = m.predict(65, 66)
        np.testing.assert_array_equal(before_pair, before_tri)
        m.observe(65, 66, 67)
        after_pair, after_tri = m.predict(65, 66)
        self.assertGreater(after_pair[67], before_pair[67])
        self.assertGreater(after_tri[67], after_pair[67])
        wrong_context = m.predict(66, 66)[1]
        np.testing.assert_array_equal(after_pair, wrong_context)

    def test_bounded_window_evicts_same_addresses_and_triples(self):
        m = memory.SignedMirrorSuccessorMemory(np.ones(256), horizon=2)
        m.observe(1, 2, 3)
        m.observe(2, 3, 4)
        m.observe(3, 4, 5)
        inv = m.invariants()
        self.assertEqual((inv["retained_edges"], inv["edge_mass"]), (2, 2))
        self.assertEqual((inv["retained_triangles"], inv["triangle_mass"]), (2, 2))
        self.assertNotIn((1, 2), m.ordered_triangles)
        self.assertEqual(m.directed_row(2)[3], 0)

    def test_causal_score_alignment_and_finite_features(self):
        raw = bytes([32, 65, 66, 67, 65, 66, 68, 65, 66, 67])
        lp = np.full((len(raw)-2, 256), -np.log(256), dtype=np.float64)
        pair, tri, invariants = memory.causal_successor_deltas(
            raw, np.ones(256)/256, lp, warmup=2
        )
        self.assertEqual(pair.shape, (8, 256))
        self.assertEqual(tri.shape, (8, 256))
        self.assertTrue(np.all(np.isfinite(pair)) and np.all(np.isfinite(tri)))
        self.assertEqual(invariants["edge_mass"], 9)
        self.assertEqual(invariants["triangle_mass"], 8)


class R20DriverTests(unittest.TestCase):
    def test_reserved_r19_test_is_not_train_or_dev(self):
        self.assertEqual(driver.R19_RESERVED_TEST_END, 100)
        self.assertEqual(driver.TOTAL_BOOKS, 132)
        self.assertEqual(driver.R19_TRAIN_DEV_END, 84)
        self.assertEqual((driver.FIT_COUNT, driver.GATE_COUNT, driver.DEV_COUNT),
                         (16, 4, 12))

    def test_function_preserving_fallback_and_controls(self):
        logp = np.full((2, 256), -np.log(256))
        pair = np.zeros((2, 256))
        tri = np.zeros((2, 256))
        self.assertIs(driver.combined_logp(logp, pair, tri, 0.0, 0.0), logp)
        rows = driver.dev_predictions(logp, pair, tri, (.5, 1.0))
        for value in rows.values():
            np.testing.assert_allclose(np.exp(value).sum(axis=1), 1, atol=1e-12)

    def test_grid_selection_rejects_nonspace_harm(self):
        fit, gate = [], []
        for a in driver.GRIDS:
            for b in driver.GRIDS:
                good = (a,b)==(.5,.5)
                bad = (a,b)==(1.,1.)
                fit.append({"pair_scale":a,"triangle_scale":b,
                            "bpb":3.95 if good else (3.8 if bad else 4.0),
                            "gain_bpb":.05 if good else (.2 if bad else 0),
                            "nonspace_gain_bpb":.01 if good else (-.03 if bad else 0)})
                gate.append({"pair_scale":a,"triangle_scale":b,
                             "bpb":3.95 if good else 4.0,
                             "nonspace_bpb":4.49 if good else 4.5,
                             "gain_bpb":.05 if good else 0,
                             "nonspace_gain_bpb":.01 if good else 0})
        chosen = driver.choose_scales(fit, gate)
        self.assertEqual((chosen["pair_scale"],chosen["triangle_scale"]), (.5,.5))

    def test_synthetic_streamed_dev_runs_without_learning(self):
        fake_data = {"targets": np.array([32,97,32,97]),
                     "nonspace": np.array([False,True,False,True]),
                     "unicode_chars":4,"elapsed_s":1.,
                     "base_logp":np.full((4,256),-np.log(256))}
        lp = np.full((4,256),-np.log(256))
        fake = (fake_data, lp, np.zeros((4,256)), np.zeros((4,256)),
                {"pair_nonnegative":True,"edge_mass":4,
                 "retained_edges":4,"triangle_mass":3,"retained_triangles":3})
        with patch.object(driver, "collect_candidate", return_value=fake), \
             patch.object(driver, "WARMUP", 4):
            result = driver.evaluate_dev([b"aa aaaaa"]*12,
                                         [f"synthetic-{i}" for i in range(12)],
                                         np.full(256,1/256),(.5,.5))
        self.assertEqual(result["dev_positions"],48)
        self.assertAlmostEqual(result["dev"]["train_selected"]["gain_bpb_vs_r19"],0,places=12)
        self.assertFalse(result["dev"]["train_selected"]["dev_gate_pass"])


if __name__ == "__main__":
    unittest.main()
