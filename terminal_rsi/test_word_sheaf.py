"""Tests for the cellular-sheaf word expert fusion and Ising/DPP selector."""
from __future__ import annotations

import contextlib
import io
import math
import sys
import unittest
from pathlib import Path

import numpy as np

# Allow ``python terminal_rsi/test_word_sheaf.py`` as well as
# ``python -m unittest terminal_rsi.test_word_sheaf``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terminal_rsi.word_sheaf import (  # noqa: E402
    CellularSheafFusion,
    IsingDPPSelect,
    fit_sheaf_weights,
    main,
    rankings_to_log_probs,
)


def _top5(scores: np.ndarray, targets: np.ndarray) -> float:
    order = np.argsort(-scores, axis=1)[:, :5]
    return float(np.mean(np.any(order == targets[:, None], axis=1)))


def _synthetic_regimes(
    n: int = 100, vocab: int = 4096, seed: int = 0
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Three experts, one correct per regime, plus agreement positions.

    The first half of the positions are agreement positions: every expert
    emits the same confident distribution.  The second half are regime
    positions: exactly one expert is confident and the other two are uniform.
    """
    rng = np.random.default_rng(seed)
    targets = rng.integers(1, vocab, size=n)
    regime = np.arange(n) % 3
    logits = np.full((n, 3, vocab), -math.log(vocab), dtype=np.float64)
    peak = math.log(0.9)
    tail = math.log(0.1 / (vocab - 1))
    half = n // 2
    for i in range(n):
        if i < half:
            for e in range(3):
                logits[i, e, :] = tail
                logits[i, e, targets[i]] = peak
        else:
            e = int(regime[i])
            logits[i, e, :] = tail
            logits[i, e, targets[i]] = peak
    return {
        "head": logits[:, 0, :],
        "bridge": logits[:, 1, :],
        "prior": logits[:, 2, :],
    }, targets


class CellularSheafFusionTests(unittest.TestCase):
    def test_fusion_beats_each_expert_and_h1_separates_disagreement(self):
        experts = ("head", "bridge", "prior")
        logs, targets = _synthetic_regimes()
        fusion = CellularSheafFusion(
            experts=experts,
            stalk_dim=4096,
            edges=[("head", "bridge"), ("bridge", "prior"),
                   ("prior", "head")],
            low_rank=2,
            mass="confidence",
            mass_power=1.0,
            seed=3,
        )
        fused, h1 = fusion.fuse(logs)
        self.assertEqual(fused.shape, (100, 4096))
        self.assertEqual(np.asarray(h1).shape, (100,))

        fused_top5 = _top5(fused, targets)
        expert_top5 = [_top5(logs[name], targets) for name in experts]
        self.assertGreater(
            fused_top5, max(expert_top5) + 0.05,
            msg=f"fusion={fused_top5} experts={expert_top5}")
        fused_top1 = float(np.mean(np.argmax(fused, axis=1) == targets))
        expert_top1 = [float(np.mean(np.argmax(logs[name], axis=1) == targets))
                       for name in experts]
        self.assertGreater(
            fused_top1, max(expert_top1) + 0.05,
            msg=f"fusion top1={fused_top1} experts={expert_top1}")

        half = 50
        h1_agree = fusion.h1_disagreement(
            {name: logs[name][:half] for name in experts})
        h1_disagree = fusion.h1_disagreement(
            {name: logs[name][half:] for name in experts})
        self.assertLess(float(np.mean(h1_agree)), 1e-9)
        self.assertGreater(float(np.mean(h1_disagree)), 0.05)
        self.assertGreater(
            float(np.mean(h1_disagree)),
            10.0 * float(np.mean(h1_agree)) + 1e-9)

    def test_harmonic_extension_is_identity_for_agreeing_experts(self):
        vocab = 64
        rng = np.random.default_rng(1)
        logs = rng.normal(size=(3, vocab))
        consensus = np.repeat(logs[:1], 3, axis=0)
        fusion = CellularSheafFusion(
            experts=("a", "b", "c"), stalk_dim=vocab, edges="triangle",
            low_rank=1, mass="uniform", normalize=False)
        fused, h1 = fusion.fuse(consensus)
        np.testing.assert_allclose(fused, consensus[0], rtol=1e-6, atol=1e-6)
        self.assertAlmostEqual(float(h1), 0.0, places=10)

    def test_tree_graph_has_no_h1(self):
        vocab = 16
        rng = np.random.default_rng(2)
        logs = {"a": rng.normal(size=vocab), "b": rng.normal(size=vocab)}
        fusion = CellularSheafFusion(
            experts=("a", "b"), stalk_dim=vocab, edges=[("a", "b")],
            low_rank=1)
        _fused, h1 = fusion.fuse(logs)
        self.assertEqual(float(h1), 0.0)


class FitTests(unittest.TestCase):
    def test_fit_temperatures_and_edge_weights(self):
        vocab, n, experts = 48, 64, 3
        rng = np.random.default_rng(4)
        targets = rng.integers(0, vocab, size=n)
        logits = np.full((n, experts, vocab), -math.log(vocab))
        for i in range(n):
            logits[i, 0, targets[i]] += 2.0
            if i % 2 == 0:
                logits[i, 1, targets[i]] += 1.0
            logits[i, 2] += rng.normal(0.0, 0.5, size=vocab)

        result = fit_sheaf_weights(
            logits, targets, experts=("head", "bridge", "prior"),
            edges="triangle", epochs=15, max_positions=64,
            edge_search_positions=16, edge_search_passes=1)
        self.assertEqual(result["temperatures"].shape, (3,))
        self.assertTrue(np.all(result["temperatures"] > 0))
        self.assertEqual(result["edge_weights"].shape, (3,))
        self.assertTrue(np.all(result["edge_weights"] >= 0))
        self.assertLess(result["train_ce"], result["baseline_ce"])
        self.assertLessEqual(result["train_ce"], result["baseline_ce"])

        fusion = CellularSheafFusion(
            experts=result["experts"], stalk_dim=vocab, edges="triangle",
            low_rank=0, mass="confidence")
        fusion.set_fit(result)
        np.testing.assert_allclose(
            fusion.temperatures, result["temperatures"])
        fused, h1 = fusion.fuse(logits)
        self.assertEqual(fused.shape, (n, vocab))
        self.assertTrue(np.all(np.isfinite(fused)))
        self.assertTrue(np.all(np.isfinite(h1)))


class IsingDPPSelectTests(unittest.TestCase):
    def test_dpp_and_mmr_return_unique_candidates(self):
        vocab = 24
        embeddings = np.random.default_rng(5).normal(size=(vocab, 6))
        scores = np.linspace(3.0, 0.0, 8)
        candidates = list(range(1, 9))
        for mode in ("dpp", "mmr"):
            selector = IsingDPPSelect(
                vocab_size=vocab, embeddings=embeddings, mode=mode,
                diversity=0.4, seed=0)
            chosen = selector.select(candidates, scores, top_k=5)
            self.assertEqual(len(chosen), 5)
            self.assertEqual(len(set(chosen)), 5)
            self.assertTrue(set(chosen).issubset(set(candidates)))

    def test_mmr_penalises_a_near_duplicate(self):
        embeddings = np.array([[1.0, 0.0], [0.999, 0.01], [0.0, 1.0]])
        selector = IsingDPPSelect(
            vocab_size=3, embeddings=embeddings, mode="mmr",
            diversity=1.0)
        chosen = selector.select(
            [0, 1, 2], np.array([2.0, 0.5, 0.0]), top_k=2)
        self.assertEqual(chosen[0], 0)
        self.assertEqual(chosen[1], 2)

    def test_pmi_field_promotes_context_word(self):
        vocab = 8
        pmi = np.zeros((vocab, vocab))
        for i in range(vocab):
            pmi[i, i] = 8.0
        selector = IsingDPPSelect(
            vocab_size=vocab, embedding_dim=4, mode="dpp",
            diversity=0.0, pmi_weight=1.0, context_weight=0.0)
        chosen = selector.select(
            [1, 2, 3], np.zeros(3), context_ids=[2], top_k=1,
            pmi_matrix=pmi)
        self.assertEqual(chosen, [2])

    def test_rankings_to_log_probs_are_normalised(self):
        rows = rankings_to_log_probs(
            np.array([[1, 2, -1], [3, -1, -1]]), 8)
        self.assertEqual(rows.shape, (2, 8))
        np.testing.assert_allclose(np.exp(rows).sum(axis=1), 1.0, rtol=1e-6)
        self.assertGreater(rows[0, 1], rows[0, 3])


class CliTests(unittest.TestCase):
    def test_missing_cache_is_reported_without_inventing_results(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["--cache", "/definitely/not/here/word_dev_cache.npz"])
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("cache not found", text)
        self.assertIn("skipping", text.lower())


if __name__ == "__main__":
    unittest.main()
