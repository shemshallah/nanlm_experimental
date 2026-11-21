#!/usr/bin/env python3
"""Build the 80 MB word-boundary cache from the 6-hour checkpoint."""
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
from terminal_rsi.unit_heads import _word_tokens, build_unit_labels  # noqa: E402
from terminal_rsi.unified import device_for  # noqa: E402
from terminal_rsi.verbatim_v103 import VerbatimV103RSI  # noqa: E402

V = 4096
MAX_ORDER = 5
TOPK = 100
BOOKS = 64
POSITIONS = 256


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
    labels = build_unit_labels(train_raw, bpe)
    print("labels", round(time.perf_counter() - t0, 1), flush=True)
    train_tokens = [w for _s, _e, w in _word_tokens(train_raw)]
    train_ids = [labels.word_vocab.get(w, 0) for w in train_tokens]
    freq = Counter(train_ids)
    uni_top = [w for w, _ in freq.most_common(TOPK)]
    counts = [None] + [defaultdict(Counter) for _ in range(MAX_ORDER)]
    for i, wid in enumerate(train_ids):
        for k in range(1, MAX_ORDER + 1):
            if i >= k:
                counts[k][tuple(train_ids[i - k:i])][wid] += 1
    print("ppm", round(time.perf_counter() - t0, 1), flush=True)

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    model = VerbatimV103RSI("blackwell").to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.set_word_vocab(labels.word_vocab)
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
        tokens = _word_tokens(excerpt)
        token_starts = {int(start): idx for idx, (start, _e, _w) in
                        enumerate(tokens)}
        dev_ids = [labels.word_vocab.get(w, 0) for _s, _e, w in tokens]
        recent = []
        positions = [p for p in range(256, min(256 + POSITIONS, len(excerpt)))
                     if bool(labels.next_word_start[p - 1])
                     and int(labels.next_word[p - 1]) != -100]
        with torch.no_grad():
            for start in range(0, len(positions), 64):
                part = positions[start:start + 64]
                windows = torch.stack([
                    torch.frombuffer(bytearray(excerpt[p - 256:p]),
                                     dtype=torch.uint8).long()
                    for p in part]).to(device)
                targets = torch.tensor([excerpt[p] for p in part],
                                       device=device, dtype=torch.long)
                aux_rows = aux[torch.tensor(
                    [p - PARITY_CONTEXT for p in part])].to(device)
                _, _, _, out = model(
                    windows, targets, aux_rows,
                    return_aux=True, return_hierarchy=True)
                heads = out["unit_logits"]["word"].float().cpu()
                prev1 = labels.prev_word_1[torch.tensor(
                    [p - 1 for p in part])].long().to(device)
                prev2 = labels.prev_word_2[torch.tensor(
                    [p - 1 for p in part])].long().to(device)
                prev3 = labels.prev_word_3[torch.tensor(
                    [p - 1 for p in part])].long().to(device)
                bridge = model.word_bridge(
                    torch.stack((prev1, prev2, prev3), dim=-1)).float().cpu()
                prior = model.word_ngram_prior(
                    prev1, prev2, prev3).float().cpu()
                for row, p in enumerate(part):
                    t_idx = token_starts.get(p)
                    ppm = (ppm_rank(dev_ids, t_idx, counts, uni_top)
                           if t_idx is not None else list(uni_top))
                    cache = cache_rank(recent)
                    copy = copy_rank(recent)
                    records.append({
                        "book": book_idx, "pos": p,
                        "target": int(labels.next_word[p - 1]),
                        "prev": [int(prev1[row]), int(prev2[row]),
                                 int(prev3[row])],
                        "head": heads[row].numpy().astype(np.float16),
                        "bridge": bridge[row].numpy().astype(np.float16),
                        "prior": prior[row].numpy().astype(np.float16),
                        "ppm": pad(ppm), "cache": pad(cache),
                        "copy": pad(copy),
                    })
                    if t_idx is not None and t_idx < len(dev_ids):
                        recent.append(dev_ids[t_idx])
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
