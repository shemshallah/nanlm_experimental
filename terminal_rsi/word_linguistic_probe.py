#!/usr/bin/env python3
"""Evaluate the rule-based linguistic experts on the 16 MB dev stream."""
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
OUT = Path("/marimo/word_linguistic_results.json")
sys.path.insert(0, str(ROOT))
from terminal_rsi.unit_heads import _word_tokens  # noqa: E402
from terminal_rsi.word_linguistic import LinguisticPrior  # noqa: E402

V = 4096


def words(text: bytes) -> list[str]:
    return [w.decode("utf-8", "ignore") for _s, _e, w in _word_tokens(text)]


def main() -> int:
    t0 = time.perf_counter()
    train_words = words((PILOT / "train_cpu_pilot.bin").read_bytes())
    counter = Counter(train_words)
    vocab = {w: i + 1 for i, (w, _n) in enumerate(counter.most_common(V - 1))}
    id_to_word = {i: w for w, i in vocab.items()}
    prior = LinguisticPrior(vocab_size=V)
    prior.build(train_words, word_to_id=vocab, id_to_word=id_to_word)
    print("built", round(time.perf_counter() - t0, 1), flush=True)

    manifest = json.loads((PILOT / "win_dev.manifest.json").read_text())
    totals = Counter()
    source_hits = Counter()
    source_units = Counter()
    for book in manifest["selected_books"]:
        raw = (CORPUS / book["relative_path"]).read_bytes()
        start = int(book.get("excerpt_start", 0))
        length = int(book.get("length", 512))
        tokens = words(raw[start:start + length])
        ids = [vocab.get(w, 0) for w in tokens]
        for i in range(1, len(ids)):
            target = ids[i]
            result = prior.query(tokens[:i])
            ranking = result.ids
            totals["units"] += 1
            if ranking and ranking[0] == target:
                totals["top1"] += 1
            if target in ranking[:5]:
                totals["top5"] += 1
            if target in ranking[:20]:
                totals["top20"] += 1
            for name, ids_src in result.sources.items():
                source_units[name] += 1
                if target in ids_src[:5]:
                    source_hits[name] += 1
        print("book", book.get("gutenberg_id"), "units", totals["units"],
              "top5", round(totals["top5"] / max(totals["units"], 1), 4),
              flush=True)
    report = {
        "units": totals["units"],
        "top1": totals["top1"] / max(totals["units"], 1),
        "top5": totals["top5"] / max(totals["units"], 1),
        "top20": totals["top20"] / max(totals["units"], 1),
        "sources": {
            name: source_hits[name] / max(source_units[name], 1)
            for name in source_units
        },
        "seconds": round(time.perf_counter() - t0, 1),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
