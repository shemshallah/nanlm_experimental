"""The *existing real terminal verifier* must own training of the neural policy."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from terminal_rsi.unified import PRESETS, Preset
from terminal_rsi.unified_agent import UnifiedTerminalPolicy
from terminal_rsi.planner import TerminalPlan, run_plan
from terminal_rsi.terminal_agent import CommandTask


class VerifiedUnifiedRSITests(unittest.TestCase):
    def setUp(self):
        self.cfg = Preset("cpu", 32, 1, 4, 64, 1, 2, 1e-3, 1)
        self.patcher = patch.dict(PRESETS, {"cpu": self.cfg})
        self.patcher.start()
        torch.manual_seed(7)

    def tearDown(self):
        self.patcher.stop()

    def test_verified_terminal_execution_changes_actor_and_neural_weights(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            task = {
                "family": "verified_echo",
                "choices": [
                    [sys.executable, "-c", "print('RIGHT')"],
                    [sys.executable, "-c", "print('WRONG')"],
                ],
                "expected_stdout": "RIGHT",
            }
            plan = TerminalPlan.parse({"plan_id": "rsitest", "goal": "print the verified word", "steps": [task]})
            actor = UnifiedTerminalPolicy("cpu")
            before_gate = actor.action_gate.detach().clone()
            before_weight = actor.neural.output[0].weight.detach().clone()
            checkpoint = root / "terminal_rsi_unified_agent.pt"
            result = run_plan(plan, actor, root, checkpoint, train=True)
            self.assertEqual(result["status"], "succeeded")
            self.assertTrue(result["records"][0]["verified_success"])
            self.assertEqual(actor.total_updates, 1)
            self.assertEqual(actor.neural_updates, 1)
            self.assertNotEqual(actor.receipt_digest, "0" * 64)
            self.assertGreater((actor.neural.output[0].weight - before_weight).abs().max().item(), 0)
            self.assertGreater((actor.action_gate - before_gate).abs().item(), 0)
            reloaded = UnifiedTerminalPolicy.load(checkpoint, mode="cpu")
            self.assertEqual(reloaded.rows, actor.rows)
            self.assertEqual(reloaded.receipt_digest, actor.receipt_digest)
            self.assertEqual(reloaded.neural_updates, 1)
            self.assertTrue(torch.equal(reloaded.action_gate, actor.action_gate))
            self.assertTrue(torch.equal(reloaded.neural.output[0].weight,
                                        actor.neural.output[0].weight))

    def test_expected_stdout_does_not_leak_into_neural_ranker(self):
        actor = UnifiedTerminalPolicy("cpu")
        cmd = (sys.executable, "-c", "print('SAFE')")
        a = CommandTask("t", "family", (cmd,), "SAFE", 0)
        b = CommandTask("t", "family", (cmd,), "FUTURE_SECRET", 0)
        with torch.no_grad():
            self.assertTrue(torch.equal(actor._candidate_score(a, 0),
                                        actor._candidate_score(b, 0)))

    def test_unverified_actions_cannot_claim_success(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            plan = TerminalPlan.parse({
                "plan_id": "failtest", "goal": "never accept wrong verifier",
                "steps": [{"family": "refuse_false_success",
                           "choices": [[sys.executable, "-c", "print('WRONG')"]],
                           "expected_stdout": "UNREACHABLE"}]})
            actor = UnifiedTerminalPolicy("cpu")
            result = run_plan(plan, actor, root, root / "bad.pt", train=True)
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["records"][0]["verified_success"])
            self.assertEqual(actor.neural_updates, 1)
            self.assertEqual(actor.total_updates, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
