#!/usr/bin/env python3
"""Tests for the v3 memory path (hashed n-grams + memory cross-attention).

Run from the repository root:

    python -m unittest z3/test_memory_attention.py -v
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

try:  # package import when run via ``python -m unittest z3/test_memory_attention.py``
    from .memory_attention import (
        HashedNgramMemory,
        MemoryBias,
        MemoryCrossAttention,
        SparseBias,
        V3MemoryPath,
    )
except ImportError:  # direct import when run from inside z3/
    from memory_attention import (
        HashedNgramMemory,
        MemoryBias,
        MemoryCrossAttention,
        SparseBias,
        V3MemoryPath,
    )


def toy_corpus() -> np.ndarray:
    """4-gram (0, 1, 2, 3) -> 9 seven times, then -> 8 three times."""
    return np.array([0, 1, 2, 3, 9] * 7 + [0, 1, 2, 3, 8] * 3, dtype=np.int64)


class HashedNgramMemoryTest(unittest.TestCase):
    def test_defaults_are_one_million_top1(self):
        mem = HashedNgramMemory()
        self.assertEqual(mem.size, 1 << 20)
        self.assertEqual(mem.order, 4)
        self.assertEqual(mem.min_count, 5)
        # 64-bit tags + sparse per-slot values; no (1M, vocab) table.
        self.assertLess(mem.nbytes(), 64 * 1024 * 1024)
        self.assertEqual(mem.keys.ndim, 1)
        self.assertEqual(mem.next_tok.ndim, 1)
        self.assertEqual(mem.counts.ndim, 1)
        self.assertEqual(mem.keys.shape[0], 1 << 20)

    def test_vectorized_hash_matches_scalar_hash(self):
        mem = HashedNgramMemory(size=1 << 12)
        corpus = toy_corpus()
        windows = np.lib.stride_tricks.sliding_window_view(corpus, mem.order)
        vectorized = mem._hash_rows(windows)
        scalar = [mem._hash_context_int(row) for row in windows]
        self.assertEqual([int(x) for x in vectorized], scalar)

    def test_build_and_query_toy_corpus(self):
        mem = HashedNgramMemory(size=1 << 16, min_count=5)
        mem.build(toy_corpus())

        hit = mem.lookup([0, 1, 2, 3])
        self.assertTrue(hit.hit)
        self.assertEqual(hit.next_token, 9)       # top-1 successor
        self.assertEqual(hit.count, 7)            # occurrences of 4-gram -> 9
        self.assertGreater(hit.count, mem.min_count)
        self.assertEqual(mem.lookup([4, 5, 6, 7]).hit, False)

        # Longer contexts use only the final `order` tokens.
        self.assertEqual(mem.lookup([42, 43, 0, 1, 2, 3]).next_token, 9)

        # Short contexts never hit.
        self.assertFalse(mem.lookup([0, 1, 2]).hit)

        bias = mem.bias([0, 1, 2, 3], vocab_size=16, alpha=0.5)
        expected = np.zeros(16, dtype=np.float32)
        expected[9] = np.float32(0.5 * math.log(7))
        np.testing.assert_allclose(bias, expected, rtol=1e-6, atol=1e-7)

    def test_min_count_filters_bias(self):
        mem = HashedNgramMemory(size=1 << 16, min_count=8).build(toy_corpus())
        self.assertTrue(mem.lookup([0, 1, 2, 3]).hit)
        self.assertEqual(mem.lookup([0, 1, 2, 3]).count, 7)
        self.assertFalse(mem.bias([0, 1, 2, 3], vocab_size=16).any())

    def test_collision_tags_block_false_hits(self):
        # Every context maps to the single bucket; only the first tag survives.
        mem = HashedNgramMemory(size=1, order=2, min_count=1)
        mem.build([1, 2, 3, 4, 5, 6])
        self.assertEqual(len(mem), 1)
        self.assertGreaterEqual(mem.collisions, 1)

        first = mem.lookup([1, 2])
        self.assertTrue(first.hit)
        self.assertEqual(first.next_token, 3)

        # The colliding context must not return the incumbent's successor.
        collided = mem.lookup([4, 5])
        self.assertFalse(collided.hit)
        self.assertEqual(collided.bucket, first.bucket)
        self.assertNotEqual(collided.key, first.key)

    def test_accepts_torch_tokens_and_rejects_bad_input(self):
        corpus = torch.tensor(toy_corpus())
        mem = HashedNgramMemory(size=1 << 16).build(corpus)
        self.assertEqual(mem.lookup(torch.tensor([0, 1, 2, 3])).next_token, 9)

        self.assertEqual(len(HashedNgramMemory().build([1, 2, 3], max_tokens=-1)), 0)
        with self.assertRaises(ValueError):
            HashedNgramMemory().build(np.zeros((2, 4), dtype=np.int64))
        with self.assertRaises(ValueError):
            HashedNgramMemory().build([0, -1, 2, 3, 4])

    def test_lookup_many_matches_scalar_lookup(self):
        mem = HashedNgramMemory(size=1 << 16, min_count=5).build(toy_corpus())
        contexts = np.array([
            [0, 1, 2, 3],
            [1, 2, 3, 9],
            [4, 5, 6, 7],
            [9, 0, 1, 2],
        ])
        hit, next_tok, count = mem.lookup_many(contexts)
        for i, ctx in enumerate(contexts):
            scalar = mem.lookup(ctx)
            self.assertEqual(bool(hit[i]), scalar.hit)
            self.assertEqual(int(next_tok[i]), scalar.next_token)
            self.assertEqual(int(count[i]), scalar.count)


class MemoryBiasTest(unittest.TestCase):
    def setUp(self):
        self.corpus = toy_corpus()
        self.mem = HashedNgramMemory(size=1 << 16, min_count=5).build(self.corpus)
        self.vocab = 16

    def test_sparse_bias_matches_per_position_lookup(self):
        tokens = torch.tensor(self.corpus[:16]).unsqueeze(0)  # (1, 16)
        module = MemoryBias(self.mem, self.vocab, alpha_init=0.5, min_count=5)
        sparse = module(tokens)

        self.assertIsInstance(sparse, SparseBias)
        self.assertEqual(sparse.shape, (1, 16, self.vocab))
        self.assertLessEqual(len(sparse), tokens.numel())
        self.assertEqual(sparse.index.shape, (len(sparse), 3))
        self.assertEqual(sparse.value.shape, (len(sparse),))

        expected = np.zeros((1, 16, self.vocab), dtype=np.float32)
        ids = tokens[0].numpy()
        for t in range(len(ids)):
            if t + 1 < self.mem.order:
                continue
            hit = self.mem.lookup(ids[t - self.mem.order + 1:t + 1])
            if hit.hit and hit.count >= self.mem.min_count:
                expected[0, t, hit.next_token] = np.float32(0.5 * math.log(hit.count))

        np.testing.assert_allclose(sparse.dense().detach().numpy(), expected,
                                   rtol=1e-6, atol=1e-7)

        logits = torch.randn(1, 16, self.vocab)
        biased = logits.clone()
        sparse.add_(biased)
        torch.testing.assert_close(biased, logits + sparse.dense().detach())

    def test_min_count_and_unseen_contexts_produce_no_bias(self):
        tokens = torch.tensor(self.corpus[:8]).unsqueeze(0)
        strict = MemoryBias(self.mem, self.vocab, min_count=8)
        self.assertEqual(len(strict(tokens)), 0)
        self.assertEqual(strict(tokens).dense().abs().sum().item(), 0.0)

        unseen = torch.tensor([[10, 11, 12, 13, 14, 15]])
        self.assertEqual(len(strict(unseen)), 0)

    def test_alpha_gets_finite_gradient(self):
        tokens = torch.tensor(self.corpus[:12]).unsqueeze(0)
        module = MemoryBias(self.mem, self.vocab, alpha_init=0.25, min_count=5)
        sparse = module(tokens)
        self.assertGreater(len(sparse), 0)
        sparse.value.sum().backward()
        self.assertIsNotNone(module.alpha.grad)
        self.assertTrue(torch.isfinite(module.alpha.grad).all())
        self.assertGreater(abs(float(module.alpha.grad)), 0.0)

    def test_batch_indices_stay_in_range(self):
        tokens = torch.tensor(self.corpus[:12]).repeat(3, 1)  # (3, 12)
        sparse = MemoryBias(self.mem, self.vocab, alpha_init=0.5)(tokens)
        self.assertGreater(len(sparse), 0)
        self.assertTrue(torch.all(sparse.index[:, 0] < 3))
        self.assertTrue(torch.all(sparse.index[:, 1] < 12))
        self.assertTrue(torch.all(sparse.index[:, 2] >= 0))
        self.assertTrue(torch.all(sparse.index[:, 2] < self.vocab))


class MemoryCrossAttentionTest(unittest.TestCase):
    def test_shapes_and_entropy(self):
        torch.manual_seed(0)
        module = MemoryCrossAttention(d_model=32, n_heads=4, n_slots=6)
        x = torch.randn(2, 7, 32)
        out, aux = module(x, return_attn=True)

        self.assertEqual(out.shape, (2, 7, 32))
        self.assertEqual(module.memory.shape, (6, 32))
        entropy = aux["memory_entropy"]
        self.assertEqual(entropy.ndim, 0)
        self.assertTrue(torch.isfinite(entropy))
        self.assertGreaterEqual(float(entropy.detach()), 0.0)
        self.assertLessEqual(float(entropy.detach()), math.log(6) + 1e-5)
        self.assertEqual(aux["memory_entropy_per_head"].shape, (4,))

        attn = aux["memory_attn"]
        self.assertEqual(attn.shape, (2, 4, 7, 6))
        torch.testing.assert_close(
            attn.sum(-1), torch.ones(2, 4, 7), rtol=1e-5, atol=1e-5
        )

    def test_layernorm_variant(self):
        module = MemoryCrossAttention(32, 4, n_slots=5, norm="layernorm")
        out, aux = module(torch.randn(1, 4, 32))
        self.assertEqual(out.shape, (1, 4, 32))
        self.assertTrue(torch.isfinite(aux["memory_entropy"]))

    def test_causal_future_tokens_do_not_change_past(self):
        torch.manual_seed(1)
        module = MemoryCrossAttention(32, 4, n_slots=8, dropout=0.0).eval()
        x = torch.randn(2, 12, 32)
        cut = 5
        x_future = x.clone()
        x_future[:, cut:] = torch.randn_like(x_future[:, cut:])

        out_a, aux_a = module(x, return_attn=True)
        out_b, aux_b = module(x_future, return_attn=True)

        torch.testing.assert_close(out_a[:, :cut], out_b[:, :cut],
                                   rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(aux_a["memory_attn"][:, :, :cut],
                                   aux_b["memory_attn"][:, :, :cut],
                                   rtol=1e-6, atol=1e-7)
        # Sanity: the future really did change.
        self.assertFalse(torch.allclose(out_a[:, cut:], out_b[:, cut:]))

    def test_all_gradients_are_finite(self):
        torch.manual_seed(2)
        module = MemoryCrossAttention(32, 4, n_slots=7)
        x = torch.randn(2, 6, 32, requires_grad=True)
        out, aux = module(x)
        (out.pow(2).mean() + 0.01 * aux["memory_entropy"]).backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        for name, param in module.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
        self.assertGreater(float(module.memory.grad.abs().sum()), 0.0)
        self.assertGreater(float(module.residual_scale.grad.abs()), 0.0)

    def test_no_t_quadratic_allocation(self):
        """Only the (B, H, T, n_slots) cross-attention may be T-wide."""
        torch.manual_seed(3)
        module = MemoryCrossAttention(32, 4, n_slots=16).eval()
        x = torch.randn(2, 256, 32)

        class ShapeRecorder(TorchDispatchMode):
            def __init__(self):
                super().__init__()
                self.shapes = []

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                self._record(out)
                return out

            def _record(self, obj):
                if isinstance(obj, torch.Tensor):
                    self.shapes.append(tuple(obj.shape))
                elif isinstance(obj, (list, tuple)):
                    for item in obj:
                        self._record(item)

        with torch.no_grad():
            with ShapeRecorder() as recorder:
                out, _ = module(x)

        self.assertEqual(out.shape, (2, 256, 32))
        for shape in recorder.shapes:
            if len(shape) >= 2:
                self.assertFalse(
                    shape[-2] >= 256 and shape[-1] >= 256,
                    f"unexpected T x T tensor allocation: {shape}",
                )


class V3MemoryPathTest(unittest.TestCase):
    def test_combined_path(self):
        torch.manual_seed(4)
        corpus = toy_corpus()
        path = V3MemoryPath(
            d_model=32, n_heads=4, vocab_size=16, n_slots=8,
            memory_size=1 << 12, min_count=5,
        ).build_memory(corpus)

        tokens = torch.tensor(corpus[:12]).unsqueeze(0)
        x = torch.randn(1, 12, 32)
        out, aux = path(x, tokens=tokens)
        self.assertEqual(out.shape, (1, 12, 32))
        self.assertIn("memory_entropy", aux)
        self.assertIn("logit_bias", aux)
        self.assertEqual(aux["logit_bias"].shape, (1, 12, 16))
        self.assertGreater(len(aux["logit_bias"]), 0)

        logits = torch.zeros(1, 12, 16)
        aux["logit_bias"].add_(logits)
        self.assertGreater(float(logits.abs().sum().detach()), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
