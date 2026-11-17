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
        return model, RSISelfBuilder(
            model, optimizer, contexts, targets, features,
            interval=interval, audit_history=contexts.clone(),
            audit_targets=targets.clone(), audit_auxiliary=features.clone())

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

    def test_rejects_probe_overfit_even_when_first_partition_improves(self):
        class ConflictingModel(TinyModel):
            def forward(self, previous, target, aux):
                direction = torch.where(aux[:, 0].float() > 0, -1.0, 1.0)
                logits = (direction * self.upgrades.router.bias[1])[:, None].expand(-1, 8)
                return logits, torch.ones_like(logits)

        model = ConflictingModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        contexts = torch.zeros(4, 32, dtype=torch.long)
        targets = torch.zeros(4, dtype=torch.long)
        verify = torch.zeros(4, 6, dtype=torch.int32)
        audit = torch.ones(4, 6, dtype=torch.int32)
        core = RSISelfBuilder(model, optimizer, contexts, targets, verify,
                              audit_history=contexts.clone(), audit_targets=targets.clone(),
                              audit_auxiliary=audit, interval=1)
        original = model.upgrades.router.bias.detach().clone()
        event = core.propose_and_verify()
        self.assertGreater(event["verified_bpb_gain"], 0)
        self.assertLess(event["audit_bpb_gain"], 0)
        self.assertFalse(event["verified"])
        self.assertTrue(torch.equal(original, model.upgrades.router.bias))
        self.assertLessEqual(event["reward"], 0)
        self.assertNotEqual(event["reward"], -0.25)

    def test_v1_checkpoint_ledger_migrates_without_optimizer_reset(self):
        model, core = self._setup(interval=5)
        snapshot = core.state_dict()
        snapshot["schema"] = "terminal-rsi-self-builder-v1"
        for key in ("min_gain_bpb", "verification_count", "audit_count"):
            snapshot.pop(key)
        _, successor = self._setup(interval=5)
        successor.load_state_dict(snapshot)
        self.assertEqual(successor.state_dict()["schema"],
                         "terminal-rsi-self-builder-v3")
        self.assertEqual(successor.proposals, core.proposals)

    def test_audit_required_to_prevent_single_probe_reward_hacking(self):
        model, core = self._setup()
        with self.assertRaisesRegex(ValueError, "Separate internal TRAIN"):
            RSISelfBuilder(model, core.optimizer, core.history,
                           core.targets, core.auxiliary)

    def test_rsi_learns_actual_target_and_mutation_direction_not_fixed_parity(self):
        model, core = self._setup()
        first = core.propose_and_verify()
        second = core.propose_and_verify()
        self.assertEqual((first["policy_arm"], second["policy_arm"]), (0, 1))
        self.assertEqual((first["delta"], second["delta"]), (+0.08, -0.08))
        self.assertGreater(core.arm_rewards[0], core.arm_rewards[1])
        self.assertEqual(core.arm_counts[:2], [1, 1])
        snapshot = core.state_dict()
        _, recovered = self._setup()
        recovered.load_state_dict(snapshot)
        self.assertEqual(recovered.arm_counts, core.arm_counts)
        self.assertEqual(recovered.arm_rewards, core.arm_rewards)

    def test_v2_fixed_direction_history_migrates_to_new_policy(self):
        model, core = self._setup()
        core.propose_and_verify()
        earlier = core.state_dict()
        earlier["schema"] = "terminal-rsi-self-builder-v2"
        for key in ("arm_counts", "arm_rewards", "bandit_epoch_proposals"):
            earlier.pop(key)
        _, successor = self._setup()
        successor.load_state_dict(earlier)
        self.assertEqual(successor.proposals, 1)
        self.assertEqual(successor.bandit_epoch_proposals, 1)
        self.assertEqual(sum(successor.arm_counts), 0)
        self.assertEqual(successor.state_dict()["schema"], "terminal-rsi-self-builder-v3")

    def test_malformed_bandit_credit_refused_on_resume(self):
        _, core = self._setup()
        bad = core.state_dict()
        bad["arm_rewards"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Malformed RSI bandit"):
            core.load_state_dict(bad)

    def test_invalid_internal_probe_and_ledger_refused(self):
        model, core = self._setup()
        with self.assertRaisesRegex(ValueError, "Non-empty"):
            RSISelfBuilder(model, core.optimizer,
                           torch.empty(0, 32), torch.empty(0), torch.empty(0, 6),
                           audit_history=core.audit_history, audit_targets=core.audit_targets,
                           audit_auxiliary=core.audit_auxiliary)
        bad = core.state_dict()
        bad["proposals"] = 1
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            core.load_state_dict(bad)


if __name__ == "__main__":
    unittest.main()
