#!/usr/bin/env python3
"""Evaluate the new alien predictors from word_alien_more on the 16 MB dev."""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/marimo/nanlm_full2")
PILOT = Path("/marimo/broad7h/runs/trsi_10m_smoke")
CORPUS = Path("/marimo/broad7h/terminal_rsi_corpus_6h")
OUT = Path("/marimo/word_alien_more_results.json")
sys.path.insert(0, str(ROOT))
from terminal_rsi.unit_heads import _word_tokens  # noqa: E402
from terminal_rsi import word_alien_more as more  # noqa: E402

V = 4096
FIT_TOKENS = 60000


def words(text: bytes) -> list[str]:
    return [w.decode("utf-8", "ignore") for _s, _e, w in _word_tokens(text)]


def build_vocab(tokens):
    c = Counter(tokens)
    return {w: i + 1 for i, (w, _n) in enumerate(c.most_common(V - 1))}


def evaluate(name, model, excerpts, fit_ids, train_ids, t0):
    try:
        t = time.perf_counter()
        if hasattr(model, "fit"):
            model.fit(fit_ids)
        elif hasattr(model, "observe"):
            for w in fit_ids:
                model.observe(int(w), learn=True)
        fit_s = time.perf_counter() - t
        hit = total = 0
        for ex in excerpts:
            if len(ex) < 2:
                continue
            acc = model.topk_accuracy(ex, top_k=5)
            n = len(ex) - 1
            hit += acc * n
            total += n
        result = {"top5": hit / max(total, 1), "units": total,
                  "fit_seconds": round(fit_s, 1),
                  "total_seconds": round(time.perf_counter() - t0, 1)}
        print(name, json.dumps(result), flush=True)
        return result
    except Exception as exc:
        print(name, "ERR", type(exc).__name__, str(exc)[:200], flush=True)
        return {"error": type(exc).__name__ + ": " + str(exc)[:200]}


def main() -> int:
    t0 = time.perf_counter()
    train_raw = (PILOT / "train_cpu_pilot.bin").read_bytes()
    vocab = build_vocab(words(train_raw))
    train_ids = np.array([vocab.get(w, 0) for w in words(train_raw)],
                         dtype=np.int32)
    manifest = json.loads((PILOT / "win_dev.manifest.json").read_text())
    excerpts = []
    for book in manifest["selected_books"]:
        raw = (CORPUS / book["relative_path"]).read_bytes()
        start = int(book.get("excerpt_start", 0))
        length = int(book.get("length", 512))
        excerpts.append(np.array([
            vocab.get(w, 0) for w in words(raw[start:start + length])],
            dtype=np.int32))
    fit = train_ids[-FIT_TOKENS:]
    print("vocab", len(vocab), "train", len(train_ids),
          "fit", len(fit), "dev_excerpts", len(excerpts), flush=True)

    report = {"vocab": len(vocab), "train_words": len(train_ids),
              "fit_tokens": len(fit), "models": {}}
    candidates = [
        ("spiking_trace", more.SpikingTraceMemory(
            V, n_neurons=192, active=6)),
        ("reservoir", more.EchoStateReservoir(V, n_fast=64, n_slow=32)),
        ("kuramoto", more.KuramotoPrior(V)),
        ("cellular_automaton", more.CellularAutomatonWordPrior(V)),
        ("ant_colony", more.AntColonyPrior(V)),
        ("renormalization_blocks", more.RenormalizationBlockModel(V)),
        ("free_energy_regimes", more.FreeEnergyRegimeModel(V, n_regimes=32)),
    ]
    for name, model in candidates:
        report["models"][name] = evaluate(
            name, model, excerpts, fit, train_ids, t0)
    report["degree_features"] = {
        "note": "PersistentHomologyFeatures is a feature extractor, not a "
                "standalone predictor; covered by tests."
    }
    report["total_seconds"] = round(time.perf_counter() - t0, 1)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
