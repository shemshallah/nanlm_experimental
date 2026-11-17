"""CPU acceptance for Blackwell execution-profile planning (same 6.5M graph)."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

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
        # Blackwell is a hardware profile for the same ~100M graph.
        total = plan["parameter_count_total"]
        self.assertGreater(total, 95_000_000)
        self.assertLess(total, 110_000_000)
        self.assertGreater(plan["provisional_headroom_gib"], 0)
        self.assertTrue(plan["gradient_checkpointing"])
        self.assertTrue(plan["native_pytorch_causal_sdpa"])
        self.assertFalse(plan["gpu_peak_verified"])
        self.assertTrue(plan["archived_v103_fp32_unchanged"])

    def test_all_modes_are_the_same_100m_graph(self):
        models = {mode: TerminalRSI(PRESETS[mode]) for mode in PRESETS}
        counts = {mode: sum(p.numel() for p in model.parameters())
                  for mode, model in models.items()}
        self.assertEqual(len(set(counts.values())), 1)
        self.assertEqual(counts["cpu"],
                         static_upgrade_parameter_count(PRESETS["cpu"]))
        total = counts["cpu"] + ORIGINAL_V103_PARAMETER_COUNT + 1
        self.assertEqual(total, 100_537_498)
        self.assertEqual(set(counts.values()), {counts["cpu"]})
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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_windows_accept_cpu_auxiliary(self):
        from terminal_rsi.verbatim_metrics import byte_metrics_wrapped
        model = VerbatimV103RSI("blackwell").to("cuda").eval()
        windows = torch.randint(0, 256, (2, 33), device="cuda")
        auxiliary = torch.randint(0, 256, (2, 6), dtype=torch.int32)  # CPU by design
        result = byte_metrics_wrapped(
            model, windows, n=2, topk_n=1, auxiliary=auxiliary)
        self.assertIn("bpb", result)
        self.assertEqual(result["evaluated_bytes"], 2)

    def test_tree_bit_logits_match_masked_reference(self):
        from terminal_rsi.verbatim_v103 import VerbatimV103RSI
        torch.manual_seed(7)
        logits = torch.randn(4, 256)
        target = torch.randint(0, 256, (4,))
        log_probs = F.log_softmax(logits.float(), dim=-1)
        choices = torch.arange(256)
        reference = []
        for depth in range(8):
            prefix = target >> (8 - depth)
            same = (choices[None, :] >> (8 - depth)) == prefix[:, None]
            bit_is_one = ((choices >> (7 - depth)) & 1).bool()[None, :]
            mass_one = torch.logsumexp(
                log_probs.masked_fill(~(same & bit_is_one), -torch.inf), -1)
            mass_zero = torch.logsumexp(
                log_probs.masked_fill(~(same & ~bit_is_one), -torch.inf), -1)
            reference.append(mass_one - mass_zero)
        reference = torch.stack(reference, dim=1)
        got = VerbatimV103RSI._byte_to_conditional_bit_logits(logits, target)
        self.assertLess(float((reference - got).abs().max()), 1e-5)

    def test_v4_freezes_residual_gate_for_rsi(self):
        from terminal_rsi.verbatim_v103 import build_v103_optimizer
        model = VerbatimV103RSI("cpu")
        optimizer = build_v103_optimizer(model, freeze_residual_gate=True)
        self.assertFalse(model.residual_gate.requires_grad)
        parameter_ids = {
            id(parameter) for group in optimizer.param_groups
            for parameter in group["params"]}
        self.assertNotIn(id(model.residual_gate), parameter_ids)

    def test_gpu_window_cache_matches_causal_contexts(self):
        from terminal_rsi.train_verbatim import (
            build_causal_contexts, build_window_cache,
        )
        raw = torch.randint(0, 256, (4096,), dtype=torch.uint8)
        history = 256
        windows, targets = build_window_cache(raw, history, torch.device("cpu"))
        positions = torch.randint(history, len(raw), (32,))
        expected = build_causal_contexts(raw, positions, history)
        actual = torch.cat((windows[positions], targets[positions, None]),
                           dim=1).long()
        self.assertTrue(torch.equal(expected, actual))

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
