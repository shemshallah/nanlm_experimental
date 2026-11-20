#!/usr/bin/env python3
"""Offline word top-5 ablation suite.

Tests tokenization, vocabulary coverage, PPM/CTW-style context memory, cache
and copy predictors on the 16 MB pilot. No gradient training and no 100M model
forward: every experiment is a deterministic count/retrieval probe.
"""
from __future__ import annotations

import gc
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/marimo/nanlm_full2")
PILOT = Path("/marimo/broad7h/runs/trsi_10m_smoke")
CORPUS = Path("/marimo/broad7h/terminal_rsi_corpus_6h")
OUT = Path("/marimo/word_offline_results.json")

sys.path.insert(0, str(ROOT))
from terminal_rsi.unit_heads import _word_tokens  # noqa: E402

MAX_ORDER = 8
UNI = 20
WORD_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_'’\-]*")


def raw_words(text: bytes) -> list[str]:
    return [w.decode("utf-8", "ignore") for _s, _e, w in _word_tokens(text)]


def norm_words(text: bytes) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(
        text.decode("utf-8", "ignore"))]


def build_vocab(tokens: list[str], size: int) -> dict[str, int]:
    counts = Counter(tokens)
    return {word: index + 1
            for index, (word, _count) in enumerate(counts.most_common(size - 1))}


def eval_ppm(train_ids: list[int], dev_ids: list[int], vocab_size: int,
             max_order: int) -> dict:
    t0 = time.perf_counter()
    counts = [None] + [defaultdict(Counter) for _ in range(max_order)]
    unigram = Counter()
    for i, wid in enumerate(train_ids):
        unigram[wid] += 1
        for k in range(1, max_order + 1):
            if i >= k:
                counts[k][tuple(train_ids[i - k:i])][wid] += 1
    build_s = time.perf_counter() - t0
    uni_top = [w for w, _ in unigram.most_common(UNI)]
    uni_total = sum(unigram.values())

    hits = {name: [0, 0, 0] for name in ("longest", "interp", "kt", "union")}
    total = 0
    order_use = Counter()
    for i in range(1, len(dev_ids)):
        target = dev_ids[i]
        rows = []
        for k in range(1, min(max_order, i) + 1):
            row = counts[k].get(tuple(dev_ids[i - k:i]))
            if row:
                rows.append((k, row))
        chosen = None
        chosen_order = 0
        for k, row in reversed(rows):
            if len(row) >= 2:
                chosen = [w for w, _ in row.most_common(UNI)]
                chosen_order = k
                break
        if chosen is None and rows:
            k, row = rows[-1]
            chosen = [w for w, _ in row.most_common(UNI)]
            chosen_order = k
        if chosen is None:
            chosen = uni_top
        chosen = chosen + [w for w in uni_top if w not in chosen]
        # Interpolated backoff, higher orders get more weight.
        scores = defaultdict(float)
        for k, row in rows:
            total_k = sum(row.values())
            weight = 1.25 ** (k - 1)
            for w, c in row.items():
                scores[w] += weight * c / total_k
        for w, c in unigram.items():
            scores[w] += 0.05 * c / uni_total
        interp = [w for w, _ in sorted(
            scores.items(), key=lambda item: (-item[1], item[0]))[:UNI]]
        # KT/CTW-style probabilistic mixture.
        kt_scores = defaultdict(float)
        for k, row in rows:
            total_k = sum(row.values())
            weight = 0.5 ** k
            for w, c in row.items():
                kt_scores[w] += weight * (c + 0.5) / (
                    total_k + 0.5 * vocab_size)
        kt_scores[0] += 1e-9
        kt = [w for w, _ in sorted(
            kt_scores.items(), key=lambda item: (-item[1], item[0]))[:UNI]]
        union = set(chosen[:5]) | set(interp[:5]) | set(kt[:5])
        total += 1
        for name, ranking in (("longest", chosen), ("interp", interp),
                              ("kt", kt)):
            if ranking and ranking[0] == target:
                hits[name][0] += 1
            if target in ranking[:5]:
                hits[name][1] += 1
            if target in ranking[:20]:
                hits[name][2] += 1
        if target in union:
            hits["union"][1] += 1
        order_use[chosen_order] += 1
    del counts
    gc.collect()
    result = {"units": total, "build_seconds": round(build_s, 1),
              "order_use": dict(sorted(order_use.items()))}
    for name, (h1, h5, h20) in hits.items():
        result[name] = {"top1": h1 / max(total, 1),
                        "top5": h5 / max(total, 1),
                        "top20": h20 / max(total, 1)}
    return result


def eval_excerpt_cache(dev_tokens: list[str], vocab: dict[str, int]) -> dict:
    cache_hits = [0, 0, 0]
    copy_hits = [0, 0]
    total = 0
    recent: list[str] = []
    for token in dev_tokens:
        if total:
            # Recency cache with exponential decay over the last 50 tokens.
            scores = defaultdict(float)
            for age, prev in enumerate(reversed(recent[-50:])):
                scores[prev] += math.exp(-age / 18.0)
            ranking = [w for w, _ in sorted(
                scores.items(), key=lambda item: (-item[1], item[0]))[:UNI]]
            if token in ranking[:5]:
                cache_hits[1] += 1
            if token in ranking[:20]:
                cache_hits[2] += 1
            if ranking and ranking[0] == token:
                cache_hits[0] += 1
            # Copy: distinct recent words, most recent first.
            seen = []
            for prev in reversed(recent[-200:]):
                if prev not in seen:
                    seen.append(prev)
            if token in seen[:5]:
                copy_hits[0] += 1
            if token in seen[:20]:
                copy_hits[1] += 1
            total += 1
        recent.append(token)
    return {"units": total,
            "cache": {"top1": cache_hits[0] / max(total, 1),
                      "top5": cache_hits[1] / max(total, 1),
                      "top20": cache_hits[2] / max(total, 1)},
            "copy": {"top5": copy_hits[0] / max(total, 1),
                     "top20": copy_hits[1] / max(total, 1)}}


def main() -> int:
    train_raw = (PILOT / "train_cpu_pilot.bin").read_bytes()
    manifest = json.loads((PILOT / "win_dev.manifest.json").read_text())
    excerpts = []
    for book in manifest["selected_books"]:
        raw = (CORPUS / book["relative_path"]).read_bytes()
        start = int(book.get("excerpt_start", 0))
        length = int(book.get("length", 512))
        excerpts.append(raw[start:start + length])

    configs = [
        ("raw4k", raw_words, 4096),
        ("norm4k", norm_words, 4096),
        ("norm16k", norm_words, 16384),
    ]
    report = {"train_bytes": len(train_raw), "excerpts": len(excerpts),
              "configs": {}}
    for name, tokenizer, vocab_size in configs:
        t0 = time.perf_counter()
        train_tokens = tokenizer(train_raw)
        vocab = build_vocab(train_tokens, vocab_size)
        train_ids = [vocab.get(w, 0) for w in train_tokens]
        dev_tokens = []
        for excerpt in excerpts:
            dev_tokens.extend(tokenizer(excerpt))
        dev_ids = [vocab.get(w, 0) for w in dev_tokens]
        oov = sum(1 for i in dev_ids if i == 0) / max(len(dev_ids), 1)
        print("config", name, "vocab", len(vocab), "train_words",
              len(train_ids), "dev_words", len(dev_ids), "oov", round(oov, 4),
              flush=True)
        result = eval_ppm(train_ids, dev_ids, vocab_size, MAX_ORDER)
        result["vocab_size"] = len(vocab)
        result["train_words"] = len(train_ids)
        result["dev_words"] = len(dev_ids)
        result["oov_rate"] = oov
        result["prep_seconds"] = round(time.perf_counter() - t0, 1)
        # Cache/copy is per excerpt, then aggregated.
        cache_total = 0
        cache_hits = [0, 0, 0]
        copy_hits = [0, 0]
        for excerpt in excerpts:
            tokens = tokenizer(excerpt)
            part = eval_excerpt_cache(tokens, vocab)
            n = part["units"]
            cache_total += n
            for idx, key in enumerate(("top1", "top5", "top20")):
                cache_hits[idx] += int(round(part["cache"][key] * n))
            copy_hits[0] += int(round(part["copy"]["top5"] * n))
            copy_hits[1] += int(round(part["copy"]["top20"] * n))
        result["cache"] = {
            "units": cache_total,
            "top1": cache_hits[0] / max(cache_total, 1),
            "top5": cache_hits[1] / max(cache_total, 1),
            "top20": cache_hits[2] / max(cache_total, 1),
        }
        result["copy"] = {
            "units": cache_total,
            "top5": copy_hits[0] / max(cache_total, 1),
            "top20": copy_hits[1] / max(cache_total, 1),
        }
        report["configs"][name] = result
        print(json.dumps(result, indent=2), flush=True)
        del train_tokens, train_ids, dev_tokens, dev_ids, vocab
        gc.collect()
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
