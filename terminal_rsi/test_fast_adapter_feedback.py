"""Tests for the revived fast-adapter feedback loop and legacy migration."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from terminal_rsi.fast_weights import RankAdapter, reopen_fast_adapter
from terminal_rsi.unified import PRESETS, Preset
from terminal_rsi.verbatim_v103 import (
    WORD_VOCAB_TENSORS, VerbatimV103RSI, build_v103_optimizer,
    expand_word_vocab_state, load_optimizer_allow_extra_params,
    resume_combined, save_combined,
)


TINY = Preset("cpu", 32, 1, 4, 64, 1, 2, 1e-3, 1)


class FastAdapterInitTests(unittest.TestCase):
    def test_zero_output_with_live_gradient(self):
        torch.manual_seed(3)
        adapter = RankAdapter(8, vocab=256, rank=4)
        state = torch.randn(2, 8)
        output = adapter(state)
        self.assertEqual(float(output.abs().max()), 0.0)
        output.sum().backward()
        self.assertGreater(float(adapter.B.grad.abs().max()), 0.0)
        self.assertGreater(float(adapter.bias.grad.abs().max()), 0.0)

    def test_reopen_only_when_both_factors_are_dead(self):
        model = nn.Module()
        model.upgrades = nn.Module()
        model.upgrades.fast_adapter = RankAdapter(8, vocab=16, rank=4)
        with torch.no_grad():
            model.upgrades.fast_adapter.A.zero_()
        self.assertTrue(reopen_fast_adapter(model))
        self.assertGreater(
            float(model.upgrades.fast_adapter.A.abs().max()), 0.0)
        self.assertEqual(
            float(model.upgrades.fast_adapter.B.abs().max()), 0.0)
        self.assertFalse(reopen_fast_adapter(model))


class CopyHeadTests(unittest.TestCase):
    def setUp(self):
        self.patch = patch.dict(PRESETS, {"cpu": TINY})
        self.patch.start()
        torch.manual_seed(13)

    def tearDown(self):
        self.patch.stop()

    def test_copy_logits_follow_the_causal_match_candidates(self):
        model = VerbatimV103RSI("cpu")
        with torch.no_grad():
            model.upgrades.copy_gate.fill_(0.1)
        history = torch.randint(0, 256, (2, 32))
        aux = torch.zeros((2, 6), dtype=torch.int32)
        aux[:, 2] = ord("a")
        aux[:, 3] = 5
        aux[:, 4] = ord("b")
        aux[:, 5] = 12
        out = model.upgrades(history, aux=aux)
        copy = out["copy_logits"]
        self.assertEqual(copy.shape, (2, 256))
        self.assertGreater(float(copy[0, ord("a")]), 0.0)
        self.assertGreater(float(copy[0, ord("b")]), 0.0)
        self.assertAlmostEqual(float(copy[0].sum()),
                               float(copy[0, ord("a")] + copy[0, ord("b")]),
                               places=6)
        empty = model.upgrades(history, aux=None)["copy_logits"]
        self.assertEqual(float(empty.abs().max()), 0.0)

    def test_copy_gate_receives_gradient(self):
        from terminal_rsi.train_verbatim import train_one_batch
        model = VerbatimV103RSI("cpu")
        optimizer = build_v103_optimizer(model)
        history = torch.randint(0, 256, (2, 32))
        target = torch.tensor([ord("a"), ord("a")])
        aux = torch.zeros((2, 6), dtype=torch.int32)
        aux[:, 2] = target
        aux[:, 3] = 5
        train_one_batch(model, optimizer, history, target, aux,
                        upgrade_loss_weight=1.0, microbatch=0)
        self.assertIsNotNone(model.upgrades.copy_gate.grad)
        self.assertGreater(
            float(model.upgrades.copy_gate.grad.abs().max()), 0.0)

    def test_legacy_checkpoint_missing_copy_gate_is_allowed(self):
        source = VerbatimV103RSI("cpu")
        state = source.state_dict()
        state.pop("upgrades.copy_gate")
        target = VerbatimV103RSI("cpu")
        missing, unexpected = target.load_state_dict(state, strict=True)
        self.assertIn("upgrades.copy_gate", missing)
        self.assertFalse(unexpected)


class VocabularyExpansionTests(unittest.TestCase):
    def setUp(self):
        self.patch = patch.dict(PRESETS, {"cpu": TINY})
        self.patch.start()
        torch.manual_seed(11)

    def tearDown(self):
        self.patch.stop()

    def test_overlapping_rows_are_preserved_and_tail_is_initialized(self):
        source = VerbatimV103RSI("cpu", word_vocab_size=64)
        state = source.state_dict()
        expanded = expand_word_vocab_state(state, 64, 128)
        target = VerbatimV103RSI("cpu", word_vocab_size=128)
        target.load_state_dict(expanded, strict=False)
        target_state = target.state_dict()
        for name in WORD_VOCAB_TENSORS:
            self.assertEqual(target_state[name].shape[0], 128)
            self.assertTrue(torch.equal(
                state[name], target_state[name][:64]), name)
        self.assertGreater(float(
            target_state["upgrades.hierarchy.unit_heads.word.weight"][64:]
            .abs().max()), 0.0)
        self.assertEqual(float(
            target_state["word_bridge.out.weight"][64:].abs().max()), 0.0)
        self.assertEqual(float(
            target_state["word_bridge.out.bias"][64:].abs().max()), 0.0)

    def test_shrinking_or_expanding_without_rows_is_rejected(self):
        state = VerbatimV103RSI("cpu", word_vocab_size=64).state_dict()
        with self.assertRaisesRegex(ValueError, "larger"):
            expand_word_vocab_state(state, 64, 64)
        with self.assertRaisesRegex(ValueError, "lacks"):
            expand_word_vocab_state(
                {k: v for k, v in state.items()
                 if k != "word_bridge.out.bias"}, 64, 128)


class LegacyOptimizerMigrationTests(unittest.TestCase):
    def test_extra_adapter_parameters_join_the_side_group(self):
        first = nn.Parameter(torch.tensor([1.0, 2.0]))
        second = nn.Parameter(torch.zeros(3))
        old = torch.optim.AdamW([first])
        old.zero_grad()
        (first.sum()).backward()
        old.step()
        state = old.state_dict()

        expanded = torch.optim.AdamW([first, second])
        self.assertTrue(load_optimizer_allow_extra_params(
            expanded, state, {second}))
        self.assertEqual(len(expanded.param_groups[0]["params"]), 2)
        self.assertIn(first, expanded.state)
        expanded.zero_grad()
        (first.sum() + second.sum()).backward()
        expanded.step()
        self.assertIn(second, expanded.state)

    def test_two_groups_accept_two_allowed_parameters(self):
        first = nn.Parameter(torch.tensor([1.0, 2.0]))
        second = nn.Parameter(torch.tensor([3.0]))
        extra_a = nn.Parameter(torch.zeros(2))
        extra_b = nn.Parameter(torch.zeros(1))
        old = torch.optim.AdamW([
            {"params": [first]}, {"params": [second]}])
        old.zero_grad()
        (first.sum() + second.sum()).backward()
        old.step()
        expanded = torch.optim.AdamW([
            {"params": [first, extra_a]}, {"params": [second, extra_b]}])
        self.assertTrue(load_optimizer_allow_extra_params(
            expanded, old.state_dict(), {extra_a, extra_b}))
        self.assertEqual(len(expanded.param_groups[0]["params"]), 2)
        self.assertEqual(len(expanded.param_groups[1]["params"]), 2)
        self.assertIn(first, expanded.state)
        self.assertIn(second, expanded.state)

    def test_unexplained_extra_parameters_are_rejected(self):
        first = nn.Parameter(torch.tensor([1.0]))
        stranger = nn.Parameter(torch.zeros(1))
        old = torch.optim.AdamW([first])
        old.zero_grad()
        first.sum().backward()
        old.step()
        with self.assertRaisesRegex(ValueError, "Unexplained"):
            load_optimizer_allow_extra_params(
                torch.optim.AdamW([first, stranger]),
                old.state_dict(), set())


class LegacyResumeAdapterTests(unittest.TestCase):
    def setUp(self):
        self.patch = patch.dict(PRESETS, {"cpu": TINY})
        self.patch.start()
        torch.manual_seed(7)
        self.context = torch.randint(0, 256, (2, 32), dtype=torch.long)
        self.target = torch.randint(0, 256, (2,), dtype=torch.long)

    def tearDown(self):
        self.patch.stop()

    def test_legacy_checkpoint_resumes_with_a_trainable_adapter(self):
        from terminal_rsi.train_verbatim import train_one_batch

        model = VerbatimV103RSI("cpu")
        old_optimizer = build_v103_optimizer(
            model, freeze_fast_adapter=True)
        aux = model.original._default_aux(self.context).to(torch.int32)
        train_one_batch(model, old_optimizer, self.context, self.target, aux)
        # Simulate the historical checkpoints: the adapter was frozen before
        # this config existed, so both factors are exactly zero.
        with torch.no_grad():
            model.upgrades.fast_adapter.A.zero_()
            model.upgrades.fast_adapter.B.zero_()
            model.upgrades.fast_adapter.bias.zero_()
        provenance = {
            "source_sha256": "orig-sha",
            "feature_manifest": {"sha": "causal"},
            "dev_sha256": "dev-sha",
            "sampler": {"cursor": 0, "stride": 1, "train_positions": 64},
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.pt"
            save_combined(path, model, old_optimizer, 11, provenance)
            recovered = VerbatimV103RSI("cpu")
            recovered_optimizer = build_v103_optimizer(recovered)
            report = resume_combined(
                path, recovered, recovered_optimizer,
                source_sha256="orig-sha",
                feature_manifest={"sha": "causal"},
                dev_sha256="dev-sha")
            self.assertEqual(report["step"], 11)
            self.assertTrue(reopen_fast_adapter(recovered))
            adapter_params = list(recovered.upgrades.fast_adapter.parameters())
            side = recovered_optimizer.param_groups[3]["params"]
            for parameter in adapter_params:
                self.assertTrue(any(parameter is item for item in side))
            before = recovered.upgrades.fast_adapter.B.detach().clone()
            aux = recovered.original._default_aux(self.context).to(torch.int32)
            train_one_batch(recovered, recovered_optimizer, self.context,
                            self.target, aux)
            self.assertFalse(torch.equal(
                before, recovered.upgrades.fast_adapter.B.detach()))


if __name__ == "__main__":
    unittest.main()
