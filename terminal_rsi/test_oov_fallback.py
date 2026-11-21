"""Unit tests for the character/byte OOV speller and gate audit."""
from __future__ import annotations

import unittest

import torch
from torch import nn

from terminal_rsi.feedback_audit import audit_gates, is_feedback_gate, reopen_gates
from terminal_rsi.oov_fallback import spell_oov


class FakeUpgrades(nn.Module):
    def forward(self, x):
        batch = x.shape[0]
        logits = torch.full((batch, 256), -100.0)
        logits[:, ord("a")] = 10.0
        logits[:, ord("b")] = 9.0
        logits[:, ord(" ")] = 0.0
        return {"unit_byte_raw": logits}


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.upgrades = FakeUpgrades()


class ToyGates(nn.Module):
    def __init__(self):
        super().__init__()
        self.upgrades = nn.Module()
        self.upgrades.hierarchy = nn.Module()
        self.upgrades.hierarchy.bottom_up_gates = nn.Parameter(torch.zeros(3))
        self.upgrades.hierarchy.top_down_gates = nn.Parameter(torch.zeros(3))
        self.upgrades.hierarchy.lateral_gates = nn.Parameter(torch.zeros(3))
        self.upgrades.hierarchy.boundary_gate = nn.Parameter(torch.zeros(()))
        self.upgrades.hierarchy.pair_memory = nn.Module()
        self.upgrades.hierarchy.pair_memory.gate = nn.Parameter(torch.zeros(()))
        self.upgrades.hierarchy.top_moe = nn.Module()
        self.upgrades.hierarchy.top_moe.gate = nn.Parameter(torch.zeros(()))
        self.upgrades.hierarchy.bottom_extensions = nn.Module()
        self.upgrades.hierarchy.bottom_extensions.gate = nn.Parameter(torch.zeros(()))
        self.upgrades.symbiosis_gate = nn.Parameter(torch.zeros(()))
        self.upgrades.symbiosis_output_gate = nn.Parameter(torch.zeros(()))
        self.upgrades.fast_weight = nn.Module()
        self.upgrades.fast_weight.scale = nn.Parameter(torch.zeros(()))
        self.upgrades.backbone = nn.ModuleList([nn.Module()])
        self.upgrades.backbone[0].moe = nn.Module()
        self.upgrades.backbone[0].moe.gate = nn.Parameter(torch.zeros(()))


class OOVFallbackTests(unittest.TestCase):
    def test_spell_returns_allowed_surfaces(self):
        model = FakeModel()
        result = spell_oov(model, b"hello ", beam=4, max_len=3, top_k=5,
                           device=torch.device("cpu"))
        self.assertTrue(result)
        for surface, score in result:
            self.assertTrue(surface)
            self.assertTrue(all(ch in "ab" for ch in surface))
            self.assertIsInstance(score, float)

    def test_spell_is_deterministic_and_causal(self):
        model = FakeModel()
        a = spell_oov(model, b"hello ", beam=3, max_len=2, top_k=3,
                      device=torch.device("cpu"))
        b = spell_oov(model, b"hello ", beam=3, max_len=2, top_k=3,
                      device=torch.device("cpu"))
        self.assertEqual(a, b)


class GateAuditTests(unittest.TestCase):
    def test_reopen_and_audit(self):
        model = ToyGates()
        self.assertTrue(any(is_feedback_gate(n) for n, _ in model.named_parameters()))
        report = audit_gates(model)
        self.assertTrue(report["dead"])
        changed = reopen_gates(model, 0.5)
        self.assertGreaterEqual(len(changed), 8)
        report = audit_gates(model)
        self.assertFalse(report["dead"])
        for name, value in report["gates"].items():
            self.assertAlmostEqual(value["mean"], 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
