#!/usr/bin/env python3
"""Alien/nonparametric word top-5 probes.

All probes are causal: the target token is scored only after the context is
built. No 100M model forward is used.
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/marimo/nanlm_full2")
PILOT = Path("/marimo/broad7h/runs/trsi_10m_smoke")
CORPUS = Path("/marimo/broad7h/terminal_rsi_corpus_6h")
OUT = Path("/marimo/word_alien_results.json")
sys.path.insert(0, str(ROOT))
from terminal_rsi.unit_heads import _word_tokens  # noqa: E402

WORD_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_'’\-]*")
MAX_ORDER = 8
UNI = 20


def raw_words(text: bytes) -> list[str]:
    return [w.decode("utf-8", "ignore") for _s, _e, w in _word_tokens(text)]


def build_vocab(tokens: list[str], size: int) -> dict[str, int]:
    counts = Counter(tokens)
    return {word: i + 1 for i, (word, _c) in
            enumerate(counts.most_common(size - 1))}


def metrics(hits: list[list[int]], total: int, name: str) -> dict:
    return {"units": total,
            "top1": hits[0] / max(total, 1),
            "top5": hits[1] / max(total, 1),
            "top20": hits[2] / max(total, 1)}


def add_hit(hits, target, ranking):
    if ranking and ranking[0] == target:
        hits[0] += 1
    if target in ranking[:5]:
        hits[1] += 1
    if target in ranking[:20]:
        hits[2] += 1


def main() -> int:
    t0 = time.perf_counter()
    train_tokens = raw_words((PILOT / "train_cpu_pilot.bin").read_bytes())
    vocab = build_vocab(train_tokens, 4096)
    train_ids = [vocab.get(w, 0) for w in train_tokens]
    manifest = json.loads((PILOT / "win_dev.manifest.json").read_text())
    dev_excerpts = []
    for book in manifest["selected_books"]:
        raw = (CORPUS / book["relative_path"]).read_bytes()
        start = int(book.get("excerpt_start", 0))
        length = int(book.get("length", 512))
        dev_excerpts.append(raw[start:start + length])
    dev_tokens_all = [w for ex in dev_excerpts for w in raw_words(ex)]
    dev_ids_all = [vocab.get(w, 0) for w in dev_tokens_all]
    V = len(vocab) + 1
    unigram = Counter(train_ids)
    uni_top = [w for w, _ in unigram.most_common(UNI)]
    print("words train", len(train_ids), "dev", len(dev_ids_all),
          "vocab", len(vocab), "oov",
          round(sum(1 for w in dev_ids_all if w == 0) / len(dev_ids_all), 4),
          flush=True)

    report = {"train_words": len(train_ids), "dev_words": len(dev_ids_all),
              "vocab": len(vocab)}
    union5 = [set() for _ in dev_ids_all]
    union20 = [set() for _ in dev_ids_all]

    # --- LZP: last 5 continuations per context order ---
    t = time.perf_counter()
    lzp = [None] + [dict() for _ in range(MAX_ORDER)]
    for i, wid in enumerate(train_ids):
        for k in range(1, MAX_ORDER + 1):
            if i >= k:
                key = tuple(train_ids[i - k:i])
                row = lzp[k].get(key)
                if row is None:
                    row = []
                    lzp[k][key] = row
                row.append(wid)
                if len(row) > 5:
                    row.pop(0)
    hits = [0, 0, 0]
    total = 0
    for i in range(1, len(dev_ids_all)):
        target = dev_ids_all[i]
        ranking = None
        for k in range(min(MAX_ORDER, i), 0, -1):
            row = lzp[k].get(tuple(dev_ids_all[i - k:i]))
            if row:
                ranking = list(reversed(row))
                break
        if ranking is None:
            ranking = uni_top
        ranking = ranking + [w for w in uni_top if w not in ranking]
        add_hit(hits, target, ranking)
        union5[i].update(ranking[:5])
        union20[i].update(ranking[:20])
        total += 1
    report["lzp"] = metrics(hits, total, "lzp")
    report["lzp"]["build_seconds"] = round(time.perf_counter() - t, 1)
    print("lzp", report["lzp"], flush=True)
    del lzp

    # --- Recency cache + copy, per excerpt ---
    cache_hits = [0, 0, 0]
    copy_hits = [0, 0, 0]
    total = 0
    for ex in dev_excerpts:
        tokens = raw_words(ex)
        ids = [vocab.get(w, 0) for w in tokens]
        recent: list[int] = []
        for target in ids:
            if recent:
                scores = defaultdict(float)
                for age, prev in enumerate(reversed(recent[-50:])):
                    scores[prev] += math.exp(-age / 18.0)
                ranking = [w for w, _ in sorted(
                    scores.items(), key=lambda kv: (-kv[1], kv[0]))[:UNI]]
                add_hit(cache_hits, target, ranking)
                seen = []
                for prev in reversed(recent[-200:]):
                    if prev not in seen:
                        seen.append(prev)
                add_hit(copy_hits, target, seen)
                total += 1
            recent.append(target)
    report["cache"] = metrics(cache_hits, total, "cache")
    report["copy"] = metrics(copy_hits, total, "copy")
    print("cache", report["cache"], "copy", report["copy"], flush=True)

    # --- Decayed transition memory (STDP-like) ---
    t = time.perf_counter()
    W = np.zeros((V, V), dtype=np.float32)
    for a, b in zip(train_ids[:-1], train_ids[1:]):
        W[a, b] += 1.0
    # L1-normalize each source row.
    row = W.sum(axis=1, keepdims=True)
    W /= np.maximum(row, 1.0)
    hits = [0, 0, 0]
    total = 0
    decay = 0.75
    for i in range(1, len(dev_ids_all)):
        target = dev_ids_all[i]
        score = np.zeros(V, dtype=np.float32)
        weight = 1.0
        for j in range(min(8, i), 0, -1):
            score += weight * W[dev_ids_all[i - j]]
            weight *= decay
        ranking = np.argpartition(-score, UNI)[:UNI]
        ranking = ranking[np.argsort(-score[ranking])].tolist()
        ranking = ranking + [w for w in uni_top if w not in ranking]
        add_hit(hits, target, ranking)
        union5[i].update(ranking[:5])
        union20[i].update(ranking[:20])
        total += 1
    report["stdp_transition"] = metrics(hits, total, "stdp")
    report["stdp_transition"]["build_seconds"] = round(
        time.perf_counter() - t, 1)
    print("stdp", report["stdp_transition"], flush=True)
    del W

    # --- HRR: random permuted word vectors + ridge readout ---
    t = time.perf_counter()
    rng = np.random.default_rng(2026)
    dim = 256
    E = rng.normal(0, 1, (V, dim)).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    # Context feature: sum decay^j * roll(E[prev_j], j*13)
    def hrr_feature(ids):
        h = np.zeros(dim, dtype=np.float32)
        weight = 1.0
        for j in range(min(8, len(ids)), 0, -1):
            wid = ids[-j]
            h += weight * np.roll(E[wid], j * 13)
            weight *= 0.8
        return h
    n_train = min(50000, len(train_ids) - 9)
    sample_idx = rng.choice(np.arange(8, len(train_ids)), n_train, replace=False)
    H = np.zeros((n_train, dim + 1), dtype=np.float32)
    H[:, -1] = 1.0
    A = np.zeros((dim + 1, dim + 1), dtype=np.float64)
    B = np.zeros((dim + 1, V), dtype=np.float64)
    for row_i, idx in enumerate(sample_idx):
        h = hrr_feature(train_ids[idx - 8:idx])
        H[row_i, :dim] = h
        A += np.outer(H[row_i], H[row_i])
        B[:, train_ids[idx]] += H[row_i]
    A += np.eye(dim + 1) * 1e-2
    Wr = np.linalg.solve(A, B).astype(np.float32)
    hits = [0, 0, 0]
    total = 0
    for i in range(1, len(dev_ids_all)):
        h = np.zeros(dim + 1, dtype=np.float32)
        h[:dim] = hrr_feature(dev_ids_all[max(0, i - 8):i])
        h[-1] = 1.0
        score = Wr.T @ h
        ranking = np.argpartition(-score, UNI)[:UNI]
        ranking = ranking[np.argsort(-score[ranking])].tolist()
        add_hit(hits, dev_ids_all[i], ranking)
        union5[i].update(ranking[:5])
        union20[i].update(ranking[:20])
        total += 1
    report["hrr"] = metrics(hits, total, "hrr")
    report["hrr"]["build_seconds"] = round(time.perf_counter() - t, 1)
    print("hrr", report["hrr"], flush=True)

    # --- Immune/kNN over HRR context keys ---
    t = time.perf_counter()
    n_proto = min(5000, n_train)
    proto_idx = rng.choice(n_train, n_proto, replace=False)
    K = H[proto_idx, :dim].copy()
    K /= np.linalg.norm(K, axis=1, keepdims=True) + 1e-9
    proto_next = np.array([train_ids[int(sample_idx[int(i)])]
                           for i in proto_idx], dtype=np.int32)
    hits = [0, 0, 0]
    total = 0
    for i in range(1, len(dev_ids_all)):
        q = hrr_feature(dev_ids_all[max(0, i - 8):i])
        q /= np.linalg.norm(q) + 1e-9
        sims = K @ q
        top = np.argpartition(-sims, 16)[:16]
        scores = Counter()
        for j in top:
            scores[proto_next[j]] += float(max(0.0, sims[j])) ** 2
        ranking = [w for w, _ in scores.most_common(UNI)]
        ranking += [w for w in uni_top if w not in ranking]
        add_hit(hits, dev_ids_all[i], ranking)
        union5[i].update(ranking[:5])
        union20[i].update(ranking[:20])
        total += 1
    report["immune_knn"] = metrics(hits, total, "immune")
    report["immune_knn"]["build_seconds"] = round(time.perf_counter() - t, 1)
    print("immune", report["immune_knn"], flush=True)
    seen = sum(1 for i, t in enumerate(dev_ids_all)
               if i > 0 and t in union5[i])
    seen20 = sum(1 for i, t in enumerate(dev_ids_all)
                 if i > 0 and t in union20[i])
    report["union"] = {"top5": seen / max(len(dev_ids_all) - 1, 1),
                        "top20": seen20 / max(len(dev_ids_all) - 1, 1),
                        "note": "LZP+STDP+HRR+immune; cache/copy excluded"}
    report["total_seconds"] = round(time.perf_counter() - t0, 1)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
