"""CPU acceptance for Blackwell planning without allocating 3B weights."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from terminal_rsi.unified import (
    PRESETS, Preset, TerminalRSI, blackwell_memory_plan,
    static_upgrade_parameter_count,
)
from terminal_rsi.verbatim_v103 import VerbatimV103RSI, exact_parent_parity


class BlackwellPreflightTests(unittest.TestCase):
    def test_static_parameter_count_matches_real_cpu_model(self):
        tiny = Preset("cpu", 32, 1, 4, 64, 1, 2, 3e-4, 1)
        model = TerminalRSI(tiny)
        self.assertEqual(static_upgrade_parameter_count(tiny),
                         sum(p.numel() for p in model.parameters()))

    def test_96_gib_memory_budget_has_provisional_headroom(self):
        plan = blackwell_memory_plan(physical_vram_gib=96.0)
        self.assertGreater(plan["parameter_count_additive_specialist"], 3_000_000_000)
        self.assertGreater(plan["provisional_headroom_gib"], 0)
        self.assertTrue(plan["gradient_checkpointing"])
        self.assertTrue(plan["native_pytorch_causal_sdpa"])
        self.assertFalse(plan["gpu_peak_verified"])
        self.assertTrue(plan["archived_v103_fp32_unchanged"])

    def test_all_3_modes_preserve_exact_original_logits_with_zero_gate(self):
        small = {
            mode: Preset(mode, 32, 1, 4, 64, 1, 2, 3e-4, 1)
            for mode in ("cpu", "t4", "blackwell")
        }
        with patch.dict(PRESETS, small):
            for mode in small:
                with self.subTest(mode=mode):
                    torch.manual_seed(103)
                    model = VerbatimV103RSI(mode)
                    history = torch.randint(0, 256, (1, 32))
                    target = torch.randint(0, 256, (1,))
                    aux = torch.randint(0, 256, (1, 6), dtype=torch.int32)
                    proof = exact_parent_parity(model, history, target, aux)
                    self.assertEqual(proof["max_abs_logit_error"], 0.0)
                    self.assertEqual(model.residual_gate.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
