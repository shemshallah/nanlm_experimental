"""CPU tests for the causal hierarchy bus and byte-driven generation."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from terminal_rsi.hierarchical_bus import (
    CausalHierarchy, bit_logits_to_byte_log_probs, boundary_loss,
    boundary_targets, causal_mean, hierarchy_parameter_count,
)
from terminal_rsi.unified import (
    PRESETS, Preset, TerminalRSI, static_upgrade_parameter_count,
)
from terminal_rsi.verbatim_v103 import VerbatimV103RSI


TINY = Preset("cpu", 32, 1, 4, 64, 1, 2, 3e-4, 1)


class HierarchyBusTests(unittest.TestCase):
    def test_parameter_count_matches_real_module(self):
        for width in (32, 128, 704):
            self.assertEqual(
                hierarchy_parameter_count(width),
                sum(p.numel() for p in CausalHierarchy(width).parameters()))

    def test_zero_gates_are_an_exact_identity(self):
        torch.manual_seed(3)
        bus = CausalHierarchy(64)
        state = torch.randn(2, 16, 64)
        out = bus(state)
        self.assertTrue(torch.equal(out.state, state))
        self.assertTrue(torch.equal(out.byte_prior, torch.zeros_like(out.byte_prior)))
        for gate in out.gates.values():
            self.assertTrue(torch.equal(gate, torch.zeros_like(gate)))

    def test_causal_mean_has_no_future_leakage(self):
        left = torch.arange(1 * 10 * 1, dtype=torch.float32).reshape(1, 10, 1)
        right = left.clone()
        right[:, 6:] += 1000.0
        a = causal_mean(left, 4)
        b = causal_mean(right, 4)
        self.assertTrue(torch.equal(a[:, :6], b[:, :6]))

    def test_boundary_targets_only_use_current_and_past(self):
        first = torch.tensor([[65, 98, 32, 46, 33, 100]])
        second = first.clone()
        second[0, 4] = 90
        a = boundary_targets(first)
        b = boundary_targets(second)
        self.assertTrue(torch.equal(a[:, :4], b[:, :4]))
        self.assertEqual(a.shape, (1, 6, 3))
        self.assertEqual(a.dtype, torch.float32)

    def test_bit_factorization_round_trip(self):
        torch.manual_seed(11)
        logits = torch.randn(3, 256) * 2.0
        target = torch.randint(0, 256, (3,))
        bit_logits = VerbatimV103RSI._byte_to_conditional_bit_logits(
            logits, target)
        reconstructed = bit_logits_to_byte_log_probs(bit_logits)
        reference = F.log_softmax(logits, dim=-1)
        self.assertLess(float((reconstructed - reference).abs().max()), 1e-5)

    def test_boundary_loss_trains_hierarchy_parameters(self):
        with patch.dict(PRESETS, {"cpu": TINY, "t4": TINY, "blackwell": TINY}):
            model = VerbatimV103RSI("cpu")
            history = torch.randint(0, 256, (2, 32))
            target = torch.randint(0, 256, (2,))
            aux = torch.randint(0, 256, (2, 6), dtype=torch.int32)
            _, _, _, upgrade_out = model(
                history, target, aux, return_aux=True, return_hierarchy=True)
            loss = boundary_loss(
                upgrade_out["boundary_logits"], boundary_targets(history))
            loss.backward()
            self.assertIsNotNone(model.upgrades.hierarchy.boundary.weight.grad)
            self.assertGreater(
                float(model.upgrades.hierarchy.boundary.weight.grad.abs().sum()), 0)

    def test_hierarchy_changes_parameter_count(self):
        self.assertEqual(
            static_upgrade_parameter_count(TINY),
            sum(p.numel() for p in TerminalRSI(TINY).parameters()))
        self.assertGreater(hierarchy_parameter_count(704), 0)

    def test_byte_generation_runs(self):
        from terminal_rsi.hierarchical_generate import generate_bytes
        with patch.dict(PRESETS, {"cpu": TINY, "t4": TINY, "blackwell": TINY}):
            model = VerbatimV103RSI("cpu")
            result = generate_bytes(
                model, b"The ", steps=3, temperature=1.0, seed=7)
            self.assertEqual(len(result["bytes"]), 3)
            self.assertEqual(len(result["records"]), 3)
            self.assertIn("text", result)
            for record in result["records"]:
                self.assertGreaterEqual(record["probability"], 0.0)


if __name__ == "__main__":
    unittest.main()
