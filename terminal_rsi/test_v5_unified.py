"""CPU integration tests for the v5 unified graph."""
from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from terminal_rsi.fast_weights import FastWeightHead, fast_weight_parameter_count
from terminal_rsi.hierarchical_bus import CausalHierarchy
from terminal_rsi.low_rank_moe import SparseRankOneMoE, moe_parameter_count
from terminal_rsi.pair_memory import (
    N_PAIR, N_PALINDROME, PAIR_CONCEPT_COUNT, PairConceptMemory,
    pair_parameter_count,
)
from terminal_rsi.unit_heads import (
    TOKEN_VOCAB_SIZE, WORD_VOCAB_SIZE, SENTENCE_VOCAB_SIZE,
    UnitHeads, build_unit_labels, unit_head_loss, unit_parameter_count,
)
from terminal_rsi.unified import (
    PRESETS, Preset, TerminalRSI, static_upgrade_parameter_count,
)
from terminal_rsi.verbatim_v103 import VerbatimV103RSI, build_v103_optimizer
from terminal_rsi.train_verbatim import train_one_batch

TINY = Preset("cpu", 32, 2, 4, 64, 1, 2, 3e-4, 1)
BPE = Path(__file__).parent / "reference" / "nancm_v103_bpe.txt"


class V5ModuleTests(unittest.TestCase):
    def test_pair_geometry_closes(self):
        self.assertEqual(PAIR_CONCEPT_COUNT, 2 * 32 ** 3)
        self.assertEqual(2 * N_PAIR + N_PALINDROME, PAIR_CONCEPT_COUNT)
        memory = PairConceptMemory(16)
        self.assertEqual(pair_parameter_count(16),
                         sum(p.numel() for p in memory.parameters()))
        bottom = torch.randn(2, 5, 16)
        top = torch.randn(2, 5, 16)
        out = memory(bottom, top)
        self.assertTrue(torch.equal(out.sentence, top))
        self.assertTrue(int(out.pair_index.min()) >= 0)
        self.assertTrue(int(out.pair_index.max()) < PAIR_CONCEPT_COUNT)
        self.assertEqual(float(out.pair_gate), 0.0)

    def test_moe_zero_gate_is_identity(self):
        moe = SparseRankOneMoE(32, experts=256)
        x = torch.randn(2, 7, 32)
        y = moe(x)
        self.assertTrue(torch.equal(x, y))
        self.assertEqual(moe_parameter_count(32),
                         sum(p.numel() for p in moe.parameters()))

    def test_fast_weight_zero_init(self):
        head = FastWeightHead(32)
        state = torch.randn(2, 7, 32)
        self.assertTrue(torch.equal(head(state), torch.zeros(2, 7, 256)))
        self.assertEqual(fast_weight_parameter_count(32),
                         sum(p.numel() for p in head.parameters()))

    def test_unit_labels_are_causal_and_vocabularies_exist(self):
        raw = b"Hello world. This is a second sentence! Alpha beta."
        labels = build_unit_labels(raw, BPE)
        self.assertTrue(labels.word_vocab)
        self.assertTrue(labels.token_vocab)
        self.assertTrue(labels.sentence_vocab)
        # Position 4 is the last byte of "Hello"; the next word is "world".
        world = labels.word_vocab[b"world."]
        self.assertEqual(int(labels.next_word[4]), world)
        # Changing a future byte cannot change an earlier next-unit label.
        changed = bytearray(raw)
        changed[-1] = ord("?")
        labels2 = build_unit_labels(bytes(changed), BPE)
        self.assertEqual(int(labels.next_word[10]), int(labels2.next_word[10]))

    def test_unit_heads_and_loss(self):
        heads = UnitHeads(16)
        subword = torch.randn(2, 5, 16)
        word = torch.randn(2, 5, 16)
        sentence = torch.randn(2, 5, 16)
        logits = heads(subword, word, sentence)
        self.assertEqual(logits["token"].shape[1], TOKEN_VOCAB_SIZE)
        self.assertEqual(logits["word"].shape[1], WORD_VOCAB_SIZE)
        self.assertEqual(logits["sentence"].shape[1], SENTENCE_VOCAB_SIZE)
        targets = {
            "next_token": torch.tensor([1, 2]),
            "next_word": torch.tensor([1, 2]),
            "next_sentence": torch.tensor([1, 2]),
        }
        loss, stats = unit_head_loss(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("word_top1", stats)
        self.assertEqual(unit_parameter_count(16),
                         sum(p.numel() for p in heads.parameters()))

    def test_fused_glu_migration_round_trip(self):
        model = TerminalRSI(TINY)
        state = model.state_dict()
        old = {}
        for key, value in state.items():
            if key.endswith(".gate_value.weight"):
                prefix = key[:-len("gate_value.weight")]
                half = value.shape[0] // 2
                old[prefix + "gate.weight"] = value[:half]
                old[prefix + "value.weight"] = value[half:]
            else:
                old[key] = value
        migrated = TerminalRSI.migrate_state_dict(old)
        for key, value in state.items():
            self.assertTrue(torch.equal(migrated[key], value), msg=key)

    def test_distillation_and_unit_training_step(self):
        with patch.dict(PRESETS, {"cpu": TINY, "t4": TINY, "blackwell": TINY}):
            model = VerbatimV103RSI("cpu")
            optimizer = build_v103_optimizer(
                model, freeze_residual_gate=True)
            history = torch.randint(0, 256, (2, 32))
            target = torch.randint(0, 256, (2,))
            aux = torch.randint(0, 256, (2, 6), dtype=torch.int32)
            unit_targets = {
                "next_token": torch.tensor([1, 2]),
                "next_word": torch.tensor([1, 2]),
                "next_sentence": torch.tensor([1, 2]),
            }
            report = train_one_batch(
                model, optimizer, history, target, aux,
                microbatch=1, upgrade_loss_weight=1.0,
                boundary_loss_weight=0.0, unit_targets=unit_targets,
                unit_loss_weight=0.1, distill_weight=0.1,
                distill_samples=2, moe_loss_weight=0.01)
            self.assertTrue(torch.isfinite(torch.tensor(report["bpb"])))
            self.assertIn("distill_kl", report)
            self.assertTrue(0.0 <= report["word_top1"] <= 1.0)
            self.assertIn("moe_load_balance", report)

    def test_kv_cache_matches_one_pass_full_sequence(self):
        torch.manual_seed(5)
        model = TerminalRSI(TINY).eval()
        prompt = torch.randint(0, 256, (1, 16))
        with torch.no_grad():
            prefill = model.forward_prefill(prompt)
            full = model(prompt, passes=1)["logits"][0, :, -1]
            next_byte = torch.randint(0, 256, (1, 1))
            full_next = model(
                torch.cat((prompt, next_byte), dim=1), passes=1
            )["logits"][0, :, -1]
            cached = model.forward_token(
                next_byte, position=prompt.shape[1],
                caches=prefill["caches"],
                basis_buffer=prefill["basis"],
                state_buffer=prefill["states"])
        self.assertLess(float((prefill["current"] - full).abs().max()), 2e-4)
        self.assertLess(float((cached["current"] - full_next).abs().max()), 2e-4)

    def test_unit_decoder_is_zero_gated_and_changes_final_logits(self):
        with patch.dict(PRESETS, {"cpu": TINY, "t4": TINY, "blackwell": TINY}):
            model = VerbatimV103RSI("cpu")
            history = torch.randint(0, 256, (2, 32))
            target = torch.randint(0, 256, (2,))
            aux = torch.randint(0, 256, (2, 6), dtype=torch.int32)
            with torch.no_grad():
                base, _ = model.original(history, target, aux)
                exact, _ = model(history, target, aux)
                gate = model.upgrades.hierarchy.unit_byte_gate
                gate.fill_(1.0)
                opened, _ = model(history, target, aux)
                gate.zero_()
                closed, _ = model(history, target, aux)
            self.assertLess(float((exact - base).abs().max()), 1e-5)
            self.assertLess(float((closed - base).abs().max()), 1e-5)
            self.assertGreater(float((opened - base).abs().max()), 1e-4)

    def test_full_parent_distillation_is_per_sample(self):
        from terminal_rsi.train_verbatim import _full_parent_log_probs
        with patch.dict(PRESETS, {"cpu": TINY, "t4": TINY, "blackwell": TINY}):
            model = VerbatimV103RSI("cpu")
            history = torch.randint(0, 256, (2, 32))
            aux = torch.randint(0, 256, (2, 6), dtype=torch.int32)
            with torch.no_grad():
                logp = _full_parent_log_probs(model, history, aux, 2)
            self.assertEqual(logp.shape, (2, 256))
            self.assertTrue(torch.allclose(
                logp.logsumexp(-1), torch.zeros(2), atol=1e-5))

    def test_ablation_switches_are_wired(self):
        bus = CausalHierarchy(
            32, ablation={"units": False, "pair": False,
                          "boundary_pooling": False})
        state = torch.randn(2, 8, 32)
        out = bus(state)
        self.assertEqual(out.unit_logits, {})
        self.assertTrue(torch.equal(out.state, state))
        self.assertEqual(float(out.pair_entropy), 0.0)

    def test_parameter_count_includes_all_v5_modules(self):
        self.assertEqual(static_upgrade_parameter_count(TINY),
                         sum(p.numel() for p in TerminalRSI(TINY).parameters()))


if __name__ == "__main__":
    unittest.main()
