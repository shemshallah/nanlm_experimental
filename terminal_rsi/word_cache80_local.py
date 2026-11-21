#!/usr/bin/env python3
"""Correct 80 MB cache: local dev labels, local previous words, local targets."""
from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/marimo/nanlm_full2")
PILOT = Path("/marimo/broad7h/runs/trsi_80m_pilot")
CORPUS = Path("/marimo/broad7h/terminal_rsi_corpus_6h")
CKPT = Path("/marimo/trsi_ckpt_6h/last.pt")
CACHE = Path("/marimo/word_dev_cache80.npz")
sys.path.insert(0, str(ROOT))

from terminal_rsi.legacy_v103_reference import (  # noqa: E402
    PARITY_CONTEXT, build_train_features, read_bpe_ranks)
from terminal_rsi.unit_heads import _word_tokens  # noqa: E402
from terminal_rsi.unified import device_for  # noqa: E402
from terminal_rsi.verbatim_v103 import VerbatimV103RSI  # noqa: E402

V = 4096
MAX_ORDER = 5
TOPK = 100
BOOKS = 64
POSITIONS = 256


def top_word_vocab(train_raw: bytes) -> dict[bytes, int]:
    words = [w for _s, _e, w in _word_tokens(train_raw)]
    counts = Counter(words)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {w: i + 1 for i, (w, _c) in enumerate(ordered[:V - 1])}


def ppm_rank(ids, i, counts, uni_top):
    ranking = None
    for k in range(min(MAX_ORDER, i), 0, -1):
        row = counts[k].get(tuple(ids[i - k:i]))
        if row and len(row) >= 2:
            ranking = [w for w, _ in row.most_common(TOPK)]
            break
    if ranking is None:
        ranking = list(uni_top)
    return ranking + [w for w in uni_top if w not in ranking]


def cache_rank(recent):
    scores = defaultdict(float)
    for age, prev in enumerate(reversed(recent[-50:])):
        scores[prev] += math.exp(-age / 18.0)
    return [w for w, _ in sorted(
        scores.items(), key=lambda kv: (-kv[1], kv[0]))[:TOPK]]


def copy_rank(recent):
    seen = []
    for prev in reversed(recent[-200:]):
        if prev not in seen:
            seen.append(prev)
    return seen[:TOPK]


def pad(seq, n=TOPK):
    seq = list(seq)[:n]
    return seq + [-1] * (n - len(seq))


def main() -> int:
    t0 = time.perf_counter()
    device = device_for("blackwell")
    train_raw = (PILOT / "train_cpu_pilot.bin").read_bytes()
    bpe = ROOT / "terminal_rsi/reference/nancm_v103_bpe.txt"
    ranks = read_bpe_ranks(bpe)
    vocab = top_word_vocab(train_raw)
    train_tokens = [w.decode("utf-8", "ignore")
                    for _s, _e, w in _word_tokens(train_raw)]
    train_ids = [vocab.get(w.encode("utf-8"), 0) for w in train_tokens]
    freq = Counter(train_ids)
    uni_top = [w for w, _ in freq.most_common(TOPK)]
    counts = [None] + [defaultdict(Counter) for _ in range(MAX_ORDER)]
    for i, wid in enumerate(train_ids):
        for k in range(1, MAX_ORDER + 1):
            if i >= k:
                counts[k][tuple(train_ids[i - k:i])][wid] += 1
    print("vocab", len(vocab), "ppm", round(time.perf_counter() - t0, 1),
          flush=True)

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    model = VerbatimV103RSI("blackwell").to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.set_word_vocab(vocab)
    model.set_word_ngram(train_raw)
    model.eval()
    print("model", round(time.perf_counter() - t0, 1), flush=True)

    manifest = json.loads((PILOT / "win_dev.manifest.json").read_text())
    records = []
    for book_idx, book in enumerate(manifest["selected_books"][:BOOKS]):
        base = int(book.get("excerpt_start", 0))
        length = int(book.get("length", 1024))
        excerpt = (CORPUS / book["relative_path"]).read_bytes()[
            base:base + length]
        features, _ = build_train_features(excerpt, ranks)
        aux = torch.from_numpy(features[PARITY_CONTEXT:].copy()).to(
            dtype=torch.int32)
        toks = _word_tokens(excerpt)
        words = [w.decode("utf-8", "ignore") for _s, _e, w in toks]
        ids = [vocab.get(w.encode("utf-8"), 0) for w in words]
        positions = [idx for idx, (start, _e, _w) in enumerate(toks)
                     if 256 <= start < 256 + POSITIONS]
        recent = []
        for start in range(0, len(positions), 64):
            part = positions[start:start + 64]
            pbytes = [int(toks[idx][0]) for idx in part]
            windows = torch.stack([
                torch.frombuffer(bytearray(
                    excerpt[p - 256:p]), dtype=torch.uint8).long()
                for p in pbytes]).to(device)
            targets = torch.tensor([excerpt[p] for p in pbytes],
                                   device=device, dtype=torch.long)
            aux_rows = aux[torch.tensor(
                [p - PARITY_CONTEXT for p in pbytes])].to(device)
            _, _, _, out = model(
                windows, targets, aux_rows,
                return_aux=True, return_hierarchy=True)
            heads = out["unit_logits"]["word"].float().cpu()
            prev = torch.zeros((len(part), 3), dtype=torch.long)
            for row, idx in enumerate(part):
                history = ids[max(0, idx - 3):idx]
                history = [0] * (3 - len(history)) + history
                prev[row] = torch.tensor(history[-3:])
            prev = prev.to(device)
            bridge = model.word_bridge(prev).float().cpu()
            prior = model.word_ngram_prior(
                prev[:, 0], prev[:, 1], prev[:, 2]).float().cpu()
            for row, idx in enumerate(part):
                ppm = ppm_rank(ids, idx, counts, uni_top)
                cache = cache_rank(recent)
                copy = copy_rank(recent)
                records.append({
                    "book": book_idx, "pos": pbytes[row],
                    "target": int(ids[idx]) if idx < len(ids) else 0,
                    "prev": prev[row].cpu().tolist(),
                    "head": heads[row].numpy().astype(np.float16),
                    "bridge": bridge[row].numpy().astype(np.float16),
                    "prior": prior[row].numpy().astype(np.float16),
                    "ppm": pad(ppm), "cache": pad(cache), "copy": pad(copy),
                })
                if idx < len(ids):
                    recent.append(ids[idx])
        if book_idx % 8 == 0:
            print("book", book_idx, len(records),
                  round(time.perf_counter() - t0, 1), flush=True)
    np.savez_compressed(
        CACHE,
        head=np.stack([r["head"] for r in records]),
        bridge=np.stack([r["bridge"] for r in records]),
        prior=np.stack([r["prior"] for r in records]),
        prev=np.array([r["prev"] for r in records], dtype=np.int32),
        target=np.array([r["target"] for r in records], dtype=np.int32),
        book=np.array([r["book"] for r in records], dtype=np.int16),
        pos=np.array([r["pos"] for r in records], dtype=np.int32),
        ppm=np.array([r["ppm"] for r in records], dtype=np.int32),
        cache=np.array([r["cache"] for r in records], dtype=np.int32),
        copy=np.array([r["copy"] for r in records], dtype=np.int32),
    )
    print("wrote", CACHE, "records", len(records),
          "seconds", round(time.perf_counter() - t0, 1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
