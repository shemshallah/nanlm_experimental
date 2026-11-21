"""CPU integration tests for the v5 unified graph."""
from __future__ import annotations

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from terminal_rsi.fast_weights import FastWeightHead, fast_weight_parameter_count
from terminal_rsi.hierarchical_bus import CausalHierarchy
from terminal_rsi.low_rank_moe import SparseRankOneMoE, moe_parameter_count
from terminal_rsi.pair_memory import (
    N_PAIR, N_PALINDROME, PAIR_CONCEPT_COUNT, PAIR_SLOTS, PairConceptMemory,
    pair_addresses, pair_parameter_count,
)
from terminal_rsi.unit_heads import (
    TOKEN_VOCAB_SIZE, WORD_VOCAB_SIZE, SENTENCE_VOCAB_SIZE,
    UnitHeads, build_unit_labels, unit_head_loss, unit_parameter_count,
    word_prefix_byte_log_probs,
)
from terminal_rsi.word_bridge import (
    WordTokenBridge, build_word4_table, word4_bucket,
)
from terminal_rsi.unified import (
    PRESETS, Preset, TerminalRSI, static_upgrade_parameter_count,
)
from terminal_rsi.verbatim_v103 import VerbatimV103RSI, build_v103_optimizer
from terminal_rsi.train_verbatim import top5_byte_loss, train_one_batch

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

    def test_pair_addresses_cover_entire_bank(self):
        left = torch.arange(PAIR_SLOTS).repeat_interleave(PAIR_SLOTS)
        right = torch.arange(PAIR_SLOTS).repeat(PAIR_SLOTS)
        addresses = pair_addresses(left, right)
        self.assertEqual(int(addresses.min()), 0)
        self.assertEqual(int(addresses.max()), PAIR_CONCEPT_COUNT - 1)
        self.assertEqual(torch.unique(addresses).numel(), PAIR_CONCEPT_COUNT)
        # Boundary points of the four disjoint ranges.
        self.assertEqual(int(pair_addresses(
            torch.tensor([0]), torch.tensor([1]))[0]), 0)
        self.assertEqual(int(pair_addresses(
            torch.tensor([0]), torch.tensor([0]))[0]), N_PAIR)
        self.assertEqual(int(pair_addresses(
            torch.tensor([1]), torch.tensor([0]))[0]), PAIR_CONCEPT_COUNT // 2)
        self.assertEqual(int(pair_addresses(
            torch.tensor([PAIR_SLOTS - 1]),
            torch.tensor([PAIR_SLOTS - 1]))[0]), PAIR_CONCEPT_COUNT - 1)

    def test_word_prefix_prior_is_normalized_and_finite(self):
        word_logits = torch.randn(2, 8, requires_grad=True)
        history = torch.tensor([[32, 97, 98, 99, 32], [32, 120, 121, 0, 32]])
        word_bytes = torch.tensor([[97, 98, 99], [120, 121, 0]])
        lengths = torch.tensor([3, 2])
        prior = word_prefix_byte_log_probs(
            word_logits, history, word_bytes, lengths)
        self.assertEqual(tuple(prior.shape), (2, 256))
        self.assertTrue(bool(torch.isfinite(prior).all()))
        normalizer = torch.logsumexp(prior, dim=-1)
        self.assertTrue(torch.allclose(normalizer, torch.zeros(2), atol=1e-5))
        (prior * torch.randn_like(prior)).sum().backward()
        self.assertTrue(bool(torch.isfinite(word_logits.grad).all()))
        # A gate exactly at zero must not create 0 * log(0) NaNs.
        gate = torch.zeros((), requires_grad=True)
        scale = 0.05 * torch.tanh(gate)
        (scale * prior.detach()).sum().backward()
        self.assertTrue(bool(torch.isfinite(gate.grad).all()))

    def test_word_token_bridge_zero_init(self):
        bridge = WordTokenBridge(16)
        out = bridge(torch.tensor([[1, 2, 3], [0, 4, 5]]))
        self.assertEqual(tuple(out.shape), (2, 16))
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))
        out.sum().backward()
        self.assertTrue(bool(torch.isfinite(bridge.out.weight.grad).all()))

    def test_hashed_word4_table(self):
        ids = [1, 2, 3, 4, 1, 2, 3, 5, 1, 2, 3, 4]
        next_ids, logp = build_word4_table(ids, buckets=16, topk=4)
        self.assertEqual(tuple(next_ids.shape), (16, 4))
        self.assertTrue(bool(torch.isfinite(logp).all()))
        bucket = word4_bucket(torch.tensor([1]), torch.tensor([2]),
                              torch.tensor([3]), buckets=16)
        self.assertTrue(0 <= int(bucket) < 16)

    def test_previous_word_labels(self):
        labels = build_unit_labels(b"alpha beta gamma delta", BPE)
        start = b"alpha beta gamma delta".index(b"delta")
        gamma = labels.word_vocab[b"gamma"]
        self.assertEqual(int(labels.prev_word_1[start]), gamma)
        self.assertEqual(int(labels.prev_word_2[start]),
                         labels.word_vocab[b"beta"])

    def test_word_boundary_window_and_ngram_loss(self):
        raw = (b"alpha beta gamma delta epsilon zeta eta theta "
               b"alpha beta gamma delta epsilon zeta eta theta") * 2
        with patch.dict(PRESETS, {"cpu": TINY, "t4": TINY,
                                  "blackwell": TINY}):
            model = VerbatimV103RSI("cpu")
            labels = build_unit_labels(raw, BPE)
            model.set_word_vocab(labels.word_vocab)
            model.set_word_ngram(raw)
            optimizer = build_v103_optimizer(
                model, freeze_residual_gate=True)
            history = torch.frombuffer(
                bytearray(raw[:32]), dtype=torch.uint8).long()[None].repeat(2, 1)
            target = torch.frombuffer(
                bytearray(raw[32:34]), dtype=torch.uint8).long()
            aux = torch.randint(0, 256, (2, 6), dtype=torch.int32)
            unit = labels.gather(torch.tensor([31, 31]))
            window = labels.gather_window(torch.tensor([0, 16]), 32)
            report = train_one_batch(
                model, optimizer, history, target, aux, microbatch=0,
                upgrade_loss_weight=1.0, unit_targets=unit,
                unit_loss_weight=0.2, unit_byte_loss_weight=0.2,
                unit_byte_distill_weight=0.0, distill_weight=0.0,
                boundary_loss_weight=0.0, moe_loss_weight=0.01,
                top5_byte_loss_weight=0.1,
                unit_window_targets=window,
                word_boundary_loss_weight=0.2,
                word_top5_loss_weight=0.1,
                word_ngram_weight=0.5)
            self.assertFalse(report.get("nonfinite_skip", False))
            self.assertIn("next_word_units", report)
            self.assertGreater(report["next_word_units"], 0)
            self.assertGreater(report["current_word_units"], 0)

    def test_top5_byte_loss_is_finite_and_ranked(self):
        bit_logits = torch.randn(4, 8, requires_grad=True)
        target = torch.randint(0, 256, (4,))
        loss = top5_byte_loss(bit_logits, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(bit_logits.grad).all()))

    def test_per_token_upgrade_gate_starts_zero(self):
        model = TerminalRSI(TINY)
        self.assertIsNotNone(model.upgrade_gate)
        self.assertEqual(float(model.upgrade_gate.weight.abs().max()), 0.0)
        self.assertEqual(float(model.upgrade_gate.bias.abs().max()), 0.0)
        state = torch.randint(0, 256, (2, 9))
        out = model(state)
        self.assertIn("gate_features", out)
        self.assertEqual(tuple(out["gate_features"].shape), (2, TINY.width))

    def test_moe_zero_gate_is_identity(self):
        moe = SparseRankOneMoE(32, experts=256)
        x = torch.randn(2, 7, 32)
        y = moe(x)
        self.assertTrue(torch.equal(x, y))
        self.assertEqual(moe_parameter_count(32),
                         sum(p.numel() for p in moe.parameters()))

    def test_moe_load_balance_carries_gradient(self):
        moe = SparseRankOneMoE(32, experts=8)
        x = torch.randn(2, 5, 32)
        moe(x)
        self.assertTrue(moe.load_balance_loss.requires_grad)
        moe.load_balance_loss.backward()
        self.assertIsNotNone(moe.router.weight.grad)
        self.assertTrue(bool(torch.isfinite(moe.router.weight.grad).all()))

    def test_moe_deepcopy_after_forward(self):
        moe = SparseRankOneMoE(32, experts=8)
        moe(torch.randn(2, 5, 32))
        clone = copy.deepcopy(moe)
        self.assertIsInstance(clone, SparseRankOneMoE)
        self.assertTrue(bool(torch.isfinite(clone.load_balance_loss).all()))

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

    def test_unit_heads_support_arbitrary_word_vocab(self):
        heads = UnitHeads(16, word_vocab_size=8192)
        subword = torch.randn(2, 5, 16)
        word = torch.randn(2, 5, 16)
        sentence = torch.randn(2, 5, 16)
        logits = heads(subword, word, sentence)
        self.assertEqual(logits["word"].shape[1], 8192)
        self.assertEqual(logits["current_word"].shape[1], 8192)
        self.assertEqual(unit_parameter_count(16, word_vocab_size=8192),
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

    def test_fast_adapter_is_zero_and_can_change_final_logits(self):
        with patch.dict(PRESETS, {"cpu": TINY, "t4": TINY, "blackwell": TINY}):
            model = VerbatimV103RSI("cpu")
            history = torch.randint(0, 256, (2, 32))
            target = torch.randint(0, 256, (2,))
            aux = torch.randint(0, 256, (2, 6), dtype=torch.int32)
            with torch.no_grad():
                base, _ = model.original(history, target, aux)
                exact, _ = model(history, target, aux)
                adapter = model.upgrades.fast_adapter
                adapter.A.normal_(0.0, 0.02)
                adapter.B.normal_(0.0, 0.02)
                opened, _ = model(history, target, aux)
                adapter.A.zero_(); adapter.B.zero_(); adapter.bias.zero_()
                closed, _ = model(history, target, aux)
            self.assertLess(float((exact - base).abs().max()), 1e-5)
            self.assertLess(float((closed - base).abs().max()), 1e-5)
            self.assertGreater(float((opened - base).abs().max()), 1e-4)

    def test_fast_adapter_population_runs_on_cached_states(self):
        from terminal_rsi.fast_rsi import FastCache, run_fast_adapter_rsi
        with patch.dict(PRESETS, {"cpu": TINY, "t4": TINY, "blackwell": TINY}):
            model = VerbatimV103RSI("cpu")
            targets = torch.randint(0, 256, (32,))
            cache = FastCache(
                inputs=torch.randn(32, model.upgrades.fast_adapter.input_dim),
                base_bits=torch.zeros(32, 8),
                bits=((targets[:, None] >> torch.arange(7, -1, -1)) & 1).float(),
                targets=targets,
            )
            stats = run_fast_adapter_rsi(
                model, cache, cache, candidates=8, rounds=2, steps=2,
                lr=0.05, rank=8, min_gain=0.0001, z_score=1.0, shards=4,
                seed=3)
            self.assertGreaterEqual(stats["proposals"], 16)
            self.assertGreaterEqual(stats["accepted"], 0)
            self.assertTrue(torch.isfinite(torch.tensor(stats["noop_probe_bpb"])))

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
