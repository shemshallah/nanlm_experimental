#!/usr/bin/env python3
"""Guard the Z4 CPU architecture row definitions."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_cpu_architecture_row import ROWS

FEEDBACK_FLAGS = (
    "--feedback-token-residual",
    "--feedback-route-concept",
    "--feedback-object-carry",
    "--feedback-memory-mtp",
    "--feedback-mtp-carry",
)
NO_FEEDBACK_FLAGS = tuple(
    "--no-" + flag.removeprefix("--") for flag in FEEDBACK_FLAGS
)


class FeedbackRowTests(unittest.TestCase):
    def test_feedback_bus_row_enables_every_gate(self):
        row = ROWS["concept_feedback_bus"]
        for flag in FEEDBACK_FLAGS:
            self.assertIn(flag, row)
        self.assertIn("--feedback-gate-warmup-steps", row)
        index = row.index("--feedback-gate-warmup-steps")
        self.assertEqual(row[index + 1], "0")

    def test_feedback_warmup_row_is_explicit(self):
        row = ROWS["concept_feedback_warmup"]
        for flag in FEEDBACK_FLAGS:
            self.assertIn(flag, row)
        index = row.index("--feedback-gate-warmup-steps")
        self.assertEqual(row[index + 1], "200")

    def test_no_feedback_row_disables_every_gate(self):
        row = ROWS["concept_no_feedback"]
        for flag in NO_FEEDBACK_FLAGS:
            self.assertIn(flag, row)
        for flag in FEEDBACK_FLAGS:
            self.assertNotIn(flag, row)
        self.assertNotIn("--feedback-gate-warmup-steps", row)


if __name__ == "__main__":
    unittest.main(verbosity=2)
