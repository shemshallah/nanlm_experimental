"""R31 checkpoint, split and non-regression acceptance invariants."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from m3_r31.train_dev import (ARMS, DEV, FIT, GATE, SEALED_TEST,
                              FrozenR30Adam, gate_pass)


class TrainingGateTests(unittest.TestCase):
    def test_parent_checkpoint_required(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                FrozenR30Adam(Path(directory) / "not-a-checkpoint.npz")

    def test_split_disjoint_and_test_sealed(self):
        intervals = [FIT, GATE, DEV, SEALED_TEST]
        for left, right in zip(intervals, intervals[1:]):
            self.assertEqual(left[1], right[0])
        self.assertEqual(ARMS[0:3], ("r29", "r30", "r31"))

    def test_regression_gate_checks_all_metrics_and_controls(self):
        data = {"arms": {
            "r29": {"bpb": 3.2, "nonspace_bpb": 4.1,
                    "top1": .50, "top5": .75},
            "r30": {"bpb": 3.19},
            "r31": {"bpb": 3.18, "nonspace_bpb": 4.08,
                    "top1": .51, "top5": .76, "books_improved": 9},
            "wrong_successor": {"bpb": 3.21},
            "unrelated_matched": {"bpb": 3.195},
        }}
        self.assertTrue(gate_pass(data, 9))
        for field, value in (("bpb", 3.2), ("nonspace_bpb", 4.11),
                             ("top1", .49), ("top5", .74),
                             ("books_improved", 8)):
            adjusted = {"arms": {arm: metrics.copy()
                                 for arm, metrics in data["arms"].items()}}
            adjusted["arms"]["r31"][field] = value
            self.assertFalse(gate_pass(adjusted, 9), field)
        adjusted = {"arms": {arm: metrics.copy()
                             for arm, metrics in data["arms"].items()}}
        adjusted["arms"]["unrelated_matched"]["bpb"] = 3.17
        self.assertFalse(gate_pass(adjusted, 9))

    def test_invalid_nonfinite_gradient(self):
        # Guard is checked before any checkpoint-based optimizer can advance.
        instance = FrozenR30Adam.__new__(FrozenR30Adam)
        instance.step = 0
        with self.assertRaisesRegex(ValueError, "non-finite"):
            instance.update(np.array([0., np.nan, 0.]))
        self.assertEqual(instance.step, 0)


if __name__ == "__main__":
    unittest.main()
