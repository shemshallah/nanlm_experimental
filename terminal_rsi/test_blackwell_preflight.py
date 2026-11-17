"""CPU acceptance for Blackwell planning without allocating 3B weights."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from terminal_rsi.unified import (
    ORIGINAL_V103_PARAMETER_COUNT, PRESETS, Preset, TerminalRSI,
    blackwell_memory_plan, static_upgrade_parameter_count,
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
        # Blackwell is a hardware profile for the same v3-sized graph.
        self.assertEqual(plan["parameter_count_total"], 6_495_162)
        self.assertEqual(plan["parameter_count_original_v103"],
                         ORIGINAL_V103_PARAMETER_COUNT)
        self.assertGreater(plan["provisional_headroom_gib"], 80)
        self.assertFalse(plan["gradient_checkpointing"])
        self.assertTrue(plan["native_pytorch_causal_sdpa"])
        self.assertFalse(plan["gpu_peak_verified"])
        self.assertTrue(plan["archived_v103_fp32_unchanged"])

    def test_all_modes_are_the_same_v3_sized_graph(self):
        models = {mode: TerminalRSI(PRESETS[mode]) for mode in PRESETS}
        counts = {mode: sum(p.numel() for p in model.parameters())
                  for mode, model in models.items()}
        self.assertEqual(len(set(counts.values())), 1)
        self.assertEqual(counts["cpu"], 3_712_646)
        self.assertEqual(counts["cpu"] + ORIGINAL_V103_PARAMETER_COUNT + 1,
                         6_495_162)
        reference = models["cpu"].state_dict()
        for mode in ("t4", "blackwell"):
            candidate = models[mode].state_dict()
            self.assertEqual(set(candidate), set(reference))
            for name, tensor in reference.items():
                self.assertEqual(candidate[name].shape, tensor.shape,
                                 msg=f"{mode}:{name}")

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "BF16 CUDA required")
    def test_cpu_blackwell_gate_zero_parity(self):
        torch.manual_seed(103)
        cpu = VerbatimV103RSI("cpu").eval()
        cuda = VerbatimV103RSI("blackwell").to("cuda").eval()
        cuda.load_state_dict(cpu.state_dict())
        history = torch.randint(0, 256, (2, 32))
        target = torch.randint(0, 256, (2,))
        aux = torch.randint(0, 256, (2, 6), dtype=torch.int32)
        with torch.no_grad():
            old_cpu, _ = cpu.original(history, target, aux)
            old_cuda, _ = cuda.original(history.cuda(), target.cuda(),
                                        aux.cuda())
            full_cpu, _ = cpu(history, target, aux)
            full_cuda, _ = cuda(history.cuda(), target.cuda(), aux.cuda())
        self.assertLess(float((old_cpu - old_cuda.cpu()).abs().max()), 2e-3)
        self.assertLess(float((full_cpu - full_cuda.cpu()).abs().max()), 2e-3)

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
