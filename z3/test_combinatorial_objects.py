#!/usr/bin/env python3
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from combinatorial_objects import (
    CombinatorialCodebook,
    CombinatorialMemory,
    binomial,
    combinadic_rank,
    combinadic_unrank,
    object_counts,
)


class CombinatorialObjectsTest(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(binomial(256, 2), 32640)
        self.assertEqual(binomial(256, 3), 2763520)
        self.assertEqual(object_counts(256)["pairs_and_triples"], 2796160)

    def test_rank_unrank_roundtrip(self):
        rng = random.Random(7)
        for vertices in (8, 16, 32):
            for k in (2, 3):
                for _ in range(30):
                    indices = tuple(sorted(rng.sample(range(vertices), k)))
                    rank = combinadic_rank(indices, vertices)
                    self.assertEqual(
                        combinadic_unrank(rank, k, vertices), indices
                    )

    def test_codebook_binding_and_growth(self):
        book = CombinatorialCodebook(vertices=16, dim=32, seed=3)
        self.assertEqual(book.counts()["pairs"], 120)
        self.assertEqual(book.counts()["triples"], 560)
        pair = book.pair(1, 7)
        triple = book.triple(1, 7, 9)
        self.assertEqual(pair.shape, (32,))
        self.assertAlmostEqual(float(pair.norm()), 1.0, places=5)
        self.assertAlmostEqual(float(triple.norm()), 1.0, places=5)
        self.assertLessEqual(
            book.pair_id(1, 7), book.counts()["pairs"] - 1
        )
        self.assertEqual(book.pair_from_id(book.pair_id(1, 7)), (1, 7))
        self.assertEqual(
            book.triple_from_id(book.triple_id(1, 7, 9)), (1, 7, 9)
        )

        old = book.vertex_codes.detach().clone()
        book.grow(8)
        self.assertEqual(book.vertices, 24)
        self.assertEqual(book.counts()["pairs"], binomial(24, 2))
        self.assertTrue(torch.equal(book.vertex_codes[:16], old))
        self.assertEqual(book.pair_from_id(book.pair_id(1, 7)), (1, 7))

    def test_sparse_combinatorial_memory_and_growth(self):
        memory = CombinatorialMemory(
            d_model=32, vertices=16, dim=16,
            top_vertices=6, top_pairs=4, seed=9,
        )
        x = torch.randn(2, 7, 32, requires_grad=True)
        out = memory(x)
        self.assertEqual(tuple(out.shape), (2, 7, 32))
        self.assertTrue(torch.isfinite(out).all())
        out.sum().backward()
        self.assertIsNotNone(memory.vertex_keys.grad)
        self.assertIsNotNone(memory.vertex_values.grad)
        old = memory.vertex_keys.detach().clone()
        memory.grow(4)
        self.assertEqual(memory.vertices, 20)
        self.assertTrue(torch.equal(memory.vertex_keys[:16], old))
        self.assertEqual(memory.counts()["pairs"], binomial(20, 2))
        out2 = memory(x.detach())
        self.assertTrue(torch.isfinite(out2).all())

    def test_future_permutation_and_scores(self):
        book = CombinatorialCodebook(vertices=16, dim=32, seed=5)
        query = torch.randn(32)
        pair_scores = book.score_pairs(query)
        self.assertEqual(pair_scores.shape, (120,))
        future = book.future(query, 2)
        self.assertEqual(future.shape, query.shape)
        self.assertFalse(torch.allclose(future, query))
        # A small candidate extension should stay bounded and sorted.
        values, ids = book.score_triples_from_pairs(
            query, torch.tensor([0, 1, 2]), topk=4
        )
        self.assertEqual(values.shape, (4,))
        self.assertEqual(ids.shape, (4,))
        self.assertTrue(torch.isfinite(values).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
