"""Unit contracts for the RSI self-builder: real proposals, verification, rollback."""
from __future__ import annotations

import unittest
import torch
from torch import nn

from terminal_rsi.rsi_self_builder import RSISelfBuilder


class TinyOriginal(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor_logits = nn.Parameter(torch.tensor([100.0, -100.0]))
        self.actor_feedback = nn.Linear(4, 2, bias=False)
        nn.init.zeros_(self.actor_feedback.weight)
        self.reward_critic = nn.Linear(4, 1)
        self.register_buffer("feedback_state", torch.zeros(4))
        self.register_buffer("verified_events", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def observe_terminal(self, verified):
        self.verified_events.add_(1)
        self.feedback_state[0] = float(verified)


class TinyUpgrades(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(1, 4)
        nn.init.zeros_(self.router.bias)
        self.refine_scale = nn.Parameter(torch.tensor(-3.0))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.original = TinyOriginal()
        self.upgrades = TinyUpgrades()
        self.residual_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, previous, target, aux):
        logits = self.upgrades.router.bias[1].expand(len(target), 8)
        bits = torch.ones_like(logits)
        return logits, bits


class BuilderContracts(unittest.TestCase):
    def _setup(self, interval=1):
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        contexts = torch.zeros(4, 32, dtype=torch.long)
        targets = torch.zeros(4, dtype=torch.long)
        features = torch.zeros(4, 6, dtype=torch.int32)
        return model, RSISelfBuilder(model, optimizer, contexts, targets, features,
                                     interval=interval)

    def test_real_verified_growth_and_exact_rejected_rollback(self):
        torch.manual_seed(123)
        model, core = self._setup()
        calls = []
        def one_training_step(m, optimizer, h, t, a):
            calls.append(1)
            return {"result": "worked"}
        report = core.train_step(one_training_step, core.history, core.targets, core.auxiliary)
        self.assertEqual(report["result"], "worked")
        self.assertEqual(calls, [1])
        self.assertEqual(core.proposals, 1)
        self.assertEqual(core.accepted, 1)
        self.assertTrue(core.events[0]["verified"])
        self.assertGreater(float(model.upgrades.router.bias[1]), 0.0)
        original = model.upgrades.router.bias.detach().clone()
        core.train_step(one_training_step, core.history, core.targets, core.auxiliary)
        self.assertEqual(core.proposals, 2)
        self.assertEqual(core.rejected, 1)
        self.assertTrue(torch.equal(original, model.upgrades.router.bias.detach()))
        self.assertEqual(int(model.original.verified_events), 2)
        self.assertEqual(core.accepted + core.rejected, core.proposals)

    def test_checkpoint_roundtrip_and_policy_mismatch_rejected(self):
        model, core = self._setup(interval=5)
        for _ in range(5):
            core.train_step(lambda *args: {}, core.history, core.targets, core.auxiliary)
        snapshot = core.state_dict()
        self.assertEqual(snapshot["total_updates"], 5)
        _, new = self._setup(interval=5)
        new.load_state_dict(snapshot)
        self.assertEqual(new.state_dict(), snapshot)
        _, wrong = self._setup(interval=3)
        with self.assertRaisesRegex(ValueError, "policy"):
            wrong.load_state_dict(snapshot)

    def test_invalid_internal_probe_and_ledger_refused(self):
        model, core = self._setup()
        with self.assertRaisesRegex(ValueError, "Non-empty"):
            RSISelfBuilder(model, core.optimizer,
                           torch.empty(0, 32), torch.empty(0), torch.empty(0, 6))
        bad = core.state_dict()
        bad["proposals"] = 1
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            core.load_state_dict(bad)


if __name__ == "__main__":
    unittest.main()
