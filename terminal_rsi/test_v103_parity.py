"""Verbatim v103 implementation parity before any RSI upgrade or GPU run."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from terminal_rsi.legacy_v103_reference import PARITY_SCHEMA, V103ParityGPUFullSystem
from terminal_rsi.unified import PRESETS, Preset
from terminal_rsi.verbatim_v103 import VerbatimV103RSI, restore_v103_parent, exact_parent_parity, build_v103_optimizer


class VerbatimGPUParityCPUTests(unittest.TestCase):
    def setUp(self):
        self.small = Preset("cpu", 32, 1, 4, 64, 1, 2, 1e-3, 1)
        self.patch = patch.dict(PRESETS, {"cpu": self.small})
        self.patch.start()
        torch.manual_seed(1003)
        self.history = torch.randint(0, 256, (2, 32))
        self.target = torch.randint(0, 256, (2,))

    def tearDown(self):
        self.patch.stop()

    def test_v103_source_verbatim_and_zero_impact_upgrade(self):
        original = V103ParityGPUFullSystem()
        model = VerbatimV103RSI("cpu")
        model.original.load_state_dict(original.state_dict(), strict=True)
        report = exact_parent_parity(model, self.history, self.target)
        self.assertEqual(report["max_abs_logit_error"], 0)
        self.assertTrue(report["exact_parity"])
        self.assertEqual(model.residual_gate.item(), 0)
        self.assertEqual(model.original.state_dict().keys(), original.state_dict().keys())
        for key, tensor in original.state_dict().items():
            self.assertTrue(torch.equal(model.original.state_dict()[key], tensor), key)

    def test_trainable_upgrade_obeys_inherited_bitwise_path(self):
        model = VerbatimV103RSI("cpu")
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        old, _ = model.original(self.history, self.target)
        out, bits = model(self.history, self.target)
        self.assertTrue(torch.equal(old, out))
        loss = F.binary_cross_entropy_with_logits(out, bits)
        loss.backward()
        self.assertIsNotNone(model.residual_gate.grad)
        self.assertGreater(model.residual_gate.grad.abs().item(), 0)
        optimizer.step()
        self.assertNotEqual(model.residual_gate.item(), 0)
        optimizer.zero_grad(set_to_none=True)
        logits, bits = model(self.history, self.target)
        F.binary_cross_entropy_with_logits(logits, bits).backward()
        self.assertIsNotNone(model.upgrades.output[0].weight.grad)
        self.assertGreater(model.upgrades.output[0].weight.grad.abs().sum().item(), 0)

    def test_verbatim_multistep_execution_replays_all_original_states(self):
        """Literal step-by-step parity to archived GPU v103 training semantics.

        The added recursive branch is gate-disabled *only for this exact replay*.
        This proves the original training path itself did not change.
        """
        parent = V103ParityGPUFullSystem()
        child = VerbatimV103RSI("cpu")
        child.original.load_state_dict(parent.state_dict(), strict=True)
        child.residual_gate.requires_grad_(False)
        for parameter in child.upgrades.parameters():
            parameter.requires_grad_(False)
        original_base = [p for name, p in parent.named_parameters()
                         if not name.startswith("v103_")]
        original_v103 = [p for name, p in parent.named_parameters()
                         if name.startswith("v103_")]
        old_opt = torch.optim.AdamW([
            {"params": original_base, "lr": 2e-3},
            {"params": original_v103, "lr": 5e-4},
        ])
        new_opt = build_v103_optimizer(child)
        old_named = dict(parent.named_parameters())
        new_named = dict(child.original.named_parameters())
        for step in range(5):
            history = (self.history + step * 13) % 256
            target = (self.target + step * 29) % 256
            aux = parent._default_aux(history).to(torch.int32)
            original_logits, target_bits = parent(history, target, aux)
            unified_logits, unified_bits = child(history, target, aux)
            self.assertTrue(torch.equal(original_logits, unified_logits),
                            "original forward changed at step "+str(step))
            self.assertTrue(torch.equal(target_bits, unified_bits))
            old_loss = F.binary_cross_entropy_with_logits(original_logits, target_bits)
            new_loss = F.binary_cross_entropy_with_logits(unified_logits, unified_bits)
            old_opt.zero_grad(set_to_none=True)
            new_opt.zero_grad(set_to_none=True)
            old_loss.backward()
            new_loss.backward()
            for name in old_named:
                a = old_named[name].grad
                b = new_named[name].grad
                self.assertEqual(a is None, b is None, name+" grad presence")
                if a is not None:
                    self.assertTrue(torch.equal(a, b), name+" grad "+str(step))
            torch.nn.utils.clip_grad_norm_(parent.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(child.parameters(), 1.0)
            old_opt.step()
            new_opt.step()
            parent.observe_online_bits(history, target, aux)
            child.original.observe_online_bits(history, target, aux)
            bpb = float(8 * old_loss.detach() / torch.log(torch.tensor(2.0)))
            acc = float(((original_logits.detach() > 0)
                         == (target_bits > 0.5)).float().mean())
            parent.observe_byte_loss(bpb, acc)
            child.observe_verified_byte(bpb, acc)
            for name, value in parent.state_dict().items():
                self.assertTrue(torch.equal(value, child.original.state_dict()[name]),
                                "original buffer/weight "+name+" step "+str(step))
            for name in old_named:
                old_state = old_opt.state.get(old_named[name], {})
                new_state = new_opt.state.get(new_named[name], {})
                self.assertEqual(set(old_state), set(new_state),
                                 name+" state keys step "+str(step))
                for key, value in old_state.items():
                    replica = new_state[key]
                    if torch.is_tensor(value):
                        self.assertTrue(torch.equal(value, replica),
                                        name+" Adam "+key+" step "+str(step))
                    else:
                        self.assertEqual(value, replica)
        parent.observe_terminal(True)
        child.observe_verified_terminal(True)
        self.assertTrue(torch.equal(parent.verified_events,
                                    child.original.verified_events))
        self.assertTrue(torch.equal(parent.feedback_state,
                                    child.original.feedback_state))

    def test_historical_two_optimizer_groups_remain_distinct(self):
        prior = V103ParityGPUFullSystem()
        base = [p for name, p in prior.named_parameters()
                if not name.startswith("v103_")]
        newer = [p for name, p in prior.named_parameters()
                 if name.startswith("v103_")]
        opt = torch.optim.AdamW([
            {"params": base, "lr": 2e-3},
            {"params": newer, "lr": 5e-4},
        ])
        for _ in range(2):
            old, bits = prior(self.history, self.target)
            opt.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(old, bits).backward()
            opt.step()
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "real_group_layout.pt"
            torch.save({"schema": PARITY_SCHEMA, "model": prior.state_dict(),
                        "optimizer": opt.state_dict(), "steps": 59182,
                        "configuration": {"causal_copy_enabled": True}}, file)
            model = VerbatimV103RSI("cpu")
            combined = build_v103_optimizer(model)
            summary = restore_v103_parent(file, model, combined)
            self.assertEqual(summary["legacy_adam_states"], len(opt.state))
            self.assertEqual(len(combined.param_groups), 3)
            self.assertEqual([g["lr"] for g in combined.param_groups[:2]],
                             [2e-3, 5e-4])
            self.assertEqual(model.original.causal_copy_enabled, True)
            for pname, parent_param in prior.named_parameters():
                child_param = dict(model.original.named_parameters())[pname]
                # A zero-initialized v103_emit gate legitimately leaves
                # some specialist parameters without Adam moments yet.
                # Their ABSENCE must also round-trip verbatim.
                a = opt.state.get(parent_param, {})
                b = combined.state.get(child_param, {})
                self.assertEqual(set(a), set(b), pname + " state keys")
                for k in ("step", "exp_avg", "exp_avg_sq"):
                    if k in a:
                        self.assertTrue(torch.equal(a[k], b[k]), pname + "." + k)
            self.assertNotIn(model.residual_gate, combined.state)
            self.assertTrue(exact_parent_parity(model, self.history,
                                               self.target)["exact_parity"])

    def test_original_adam_state_and_online_buffers_recover_exactly(self):
        prior = V103ParityGPUFullSystem()
        opt = torch.optim.AdamW(prior.parameters(), lr=1e-3)
        logits, bits = prior(self.history, self.target)
        F.binary_cross_entropy_with_logits(logits, bits).backward()
        opt.step()
        prior.observe_online_bits(self.history, self.target, prior._default_aux(self.history))
        prior.observe_byte_loss(3.1, 0.5)
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "parent_v103.pt"
            torch.save({"schema": PARITY_SCHEMA, "model": prior.state_dict(),
                        "optimizer": opt.state_dict(), "steps": 127,
                        "configuration": {"causal_copy_enabled": False}}, file)
            model = VerbatimV103RSI("cpu")
            unified_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            summary = restore_v103_parent(file, model, unified_opt)
            self.assertEqual(summary["legacy_step"], 127)
            self.assertEqual(summary["legacy_adam_states"], len(opt.state))
            self.assertEqual(len(unified_opt.state), len(opt.state))
            self.assertTrue(torch.equal(model.original.v103_slot_counts,
                                        prior.v103_slot_counts))
            self.assertTrue(torch.equal(model.original.feedback_state,
                                        prior.feedback_state))
            exact_parent_parity(model, self.history, self.target)
            # Inherited moments must land on the intended original parameter,
            # not on the newly registered top-level residual gate.
            a = prior.context_table.weight
            b = model.original.context_table.weight
            self.assertTrue(torch.equal(opt.state[a]["exp_avg"],
                                        unified_opt.state[b]["exp_avg"]))
            self.assertNotIn(model.residual_gate, unified_opt.state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
