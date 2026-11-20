#!/usr/bin/env python3
"""Cache model word features and test candidate-union recall + listwise rerank.

Dev books are split 8/8: fit on the first half, report on the second half.
Candidate sources: word head, GRU bridge, bigram/4-gram prior, PPM, cache, copy.
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/marimo/nanlm_full2")
PILOT = Path("/marimo/broad7h/runs/trsi_10m_smoke")
CORPUS = Path("/marimo/broad7h/terminal_rsi_corpus_6h")
CKPT = Path("/marimo/broad7h/runs/trsi_10m_smoke_seq2/last.pt")
OUT = Path("/marimo/word_rerank_results.json")
CACHE = Path("/marimo/word_dev_cache.npz")
sys.path.insert(0, str(ROOT))

from terminal_rsi.legacy_v103_reference import (  # noqa: E402
    PARITY_CONTEXT, build_train_features, read_bpe_ranks)
from terminal_rsi.unit_heads import _word_tokens, build_unit_labels  # noqa: E402
from terminal_rsi.unified import device_for  # noqa: E402
from terminal_rsi.verbatim_v103 import VerbatimV103RSI  # noqa: E402

V = 4096
MAX_ORDER = 8
TOPK = 20


def build_ppm(train_ids, max_order=MAX_ORDER):
    counts = [None] + [defaultdict(Counter) for _ in range(max_order)]
    for i, wid in enumerate(train_ids):
        for k in range(1, max_order + 1):
            if i >= k:
                counts[k][tuple(train_ids[i - k:i])][wid] += 1
    return counts


def ppm_rank(ids, i, counts, uni_top, max_order=MAX_ORDER):
    ranking = None
    for k in range(min(max_order, i), 0, -1):
        row = counts[k].get(tuple(ids[i - k:i]))
        if row and len(row) >= 2:
            ranking = [w for w, _ in row.most_common(TOPK)]
            break
    if ranking is None:
        ranking = list(uni_top)
    return ranking + [w for w in uni_top if w not in ranking]


def cache_rank(recent):
    if not recent:
        return []
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


def main() -> int:
    t0 = time.perf_counter()
    device = device_for("blackwell")
    train_raw = (PILOT / "train_cpu_pilot.bin").read_bytes()
    bpe = ROOT / "terminal_rsi/reference/nancm_v103_bpe.txt"
    ranks = read_bpe_ranks(bpe)
    labels = build_unit_labels(train_raw, bpe)
    word_vocab = labels.word_vocab
    id_to_bytes = {i: w for w, i in word_vocab.items()}
    train_tokens = [w for _s, _e, w in _word_tokens(train_raw)]
    train_ids = [word_vocab.get(w, 0) for w in train_tokens]
    freq = Counter(train_ids)
    uni_top = [w for w, _ in freq.most_common(TOPK)]
    print("labels", len(word_vocab), "train_words", len(train_ids),
          "seconds", round(time.perf_counter() - t0, 1), flush=True)

    print("ppm build", flush=True)
    counts = build_ppm(train_ids)
    print("ppm built", round(time.perf_counter() - t0, 1), flush=True)

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    model = VerbatimV103RSI("blackwell").to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.set_word_vocab(word_vocab)
    model.set_word_ngram(train_raw)
    model.eval()
    print("model loaded", round(time.perf_counter() - t0, 1), flush=True)

    manifest = json.loads((PILOT / "win_dev.manifest.json").read_text())
    records = []
    for book_idx, book in enumerate(manifest["selected_books"]):
        excerpt = (CORPUS / book["relative_path"]).read_bytes()[
            int(book.get("excerpt_start", 0)):
            int(book.get("excerpt_start", 0)) + int(book.get("length", 512))]
        features, _ = build_train_features(excerpt, ranks)
        aux = torch.from_numpy(features[PARITY_CONTEXT:].copy()).to(
            dtype=torch.int32)
        tokens = _word_tokens(excerpt)
        token_starts = {int(start): idx for idx, (start, _e, _w) in
                        enumerate(tokens)}
        dev_ids = [word_vocab.get(w, 0) for _s, _e, w in tokens]
        recent: list[int] = []
        positions = [p for p in range(max(256, PARITY_CONTEXT), len(excerpt))
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
                    # Cache/copy are computed from the token prefix.
                    cache = cache_rank(recent)
                    copy = copy_rank(recent)
                    records.append({
                        "book": book_idx,
                        "pos": p,
                        "target": int(labels.next_word[p - 1]),
                        "prev": [int(prev1[row]), int(prev2[row]),
                                 int(prev3[row])],
                        "head": heads[row].numpy().astype(np.float16),
                        "bridge": bridge[row].numpy().astype(np.float16),
                        "prior": prior[row].numpy().astype(np.float16),
                        "ppm": ppm[:TOPK],
                        "cache": cache[:TOPK],
                        "copy": copy[:TOPK],
                    })
                    if t_idx is not None and t_idx < len(dev_ids):
                        recent.append(dev_ids[t_idx])
        if book_idx % 4 == 0:
            print("book", book_idx, "records", len(records),
                  "seconds", round(time.perf_counter() - t0, 1), flush=True)

    def pad(seq, n=TOPK):
        seq = list(seq)[:n]
        return seq + [-1] * (n - len(seq))

    # Persist a compact cache for later experiments.
    np.savez_compressed(
        CACHE,
        head=np.stack([r["head"] for r in records]),
        bridge=np.stack([r["bridge"] for r in records]),
        prior=np.stack([r["prior"] for r in records]),
        prev=np.array([r["prev"] for r in records], dtype=np.int32),
        target=np.array([r["target"] for r in records], dtype=np.int32),
        book=np.array([r["book"] for r in records], dtype=np.int16),
        ppm=np.array([pad(r["ppm"]) for r in records], dtype=np.int32),
        cache=np.array([pad(r["cache"]) for r in records], dtype=np.int32),
        copy=np.array([pad(r["copy"]) for r in records], dtype=np.int32),
    )
    print("cache written", CACHE, flush=True)

    # Candidate union and features.
    def source_rank(ranking, candidate):
        try:
            return ranking.index(candidate)
        except ValueError:
            return -1

    def build_examples(book_filter):
        examples = []
        for r in records:
            if not book_filter(r["book"]):
                continue
            sources = {
                "head": list(np.argsort(-r["head"].astype(np.float32))[:TOPK]),
                "bridge": list(np.argsort(-r["bridge"].astype(np.float32))[:TOPK]),
                "prior": list(np.argsort(-r["prior"].astype(np.float32))[:TOPK]),
                "ppm": r["ppm"],
                "cache": r["cache"],
                "copy": r["copy"],
            }
            candidates = []
            for ids in sources.values():
                for w in ids:
                    w = int(w)
                    if w not in candidates:
                        candidates.append(w)
            target = int(r["target"])
            if target not in candidates:
                continue
            feats = []
            for c in candidates:
                f = []
                for name in ("head", "bridge", "prior", "ppm", "cache", "copy"):
                    rank = source_rank(sources[name], c)
                    f.append(1.0 / (rank + 1) if rank >= 0 else 0.0)
                f += [float(r["head"][c]) / 10.0,
                      float(r["bridge"][c]) / 10.0,
                      float(r["prior"][c]) / 10.0,
                      math.log1p(freq.get(c, 0)),
                      len(id_to_bytes.get(c, b"")),
                      1.0 if c in r["cache"] else 0.0,
                      1.0 if c in r["copy"] else 0.0,
                      1.0 if c == r["prev"][0] else 0.0,
                      1.0 if c == r["prev"][1] else 0.0,
                      1.0 if c == r["prev"][2] else 0.0]
                feats.append(f)
            examples.append({
                "candidates": candidates,
                "features": torch.tensor(feats, dtype=torch.float32),
                "target_index": candidates.index(target),
                "target": target,
                "sources": sources,
            })
        return examples

    train_ex = build_examples(lambda b: b < 8)
    test_ex = build_examples(lambda b: b >= 8)
    print("examples", len(train_ex), len(test_ex), flush=True)

    def recall(examples, k):
        hit = 0
        union_hit = 0
        for ex in examples:
            union = set()
            for ids in ex["sources"].values():
                union.update(int(w) for w in ids[:k])
            if ex["target"] in union:
                hit += 1
        return hit / max(len(examples), 1)

    def oracle_top5(examples):
        hit = 0
        for ex in examples:
            union = set()
            for ids in ex["sources"].values():
                union.update(int(w) for w in ids[:5])
            if ex["target"] in union:
                hit += 1
        return hit / max(len(examples), 1)

    report = {
        "records": len(records), "train_examples": len(train_ex),
        "test_examples": len(test_ex),
        "recall@20": recall(test_ex, 20),
        "recall@50": recall(test_ex, 50),
        "oracle_top5": oracle_top5(test_ex),
    }
    print("union", json.dumps(report), flush=True)

    class Reranker(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, 64), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    dim = len(train_ex[0]["features"][0]) if train_ex else 16
    model_r = Reranker(dim).to(device)
    opt = torch.optim.AdamW(model_r.parameters(), lr=1e-3, weight_decay=1e-2)
    for epoch in range(120):
        model_r.train()
        total = 0.0
        for ex in train_ex:
            feats = ex["features"].to(device)
            scores = model_r(feats)
            loss = F.cross_entropy(scores[None, :], torch.tensor(
                [ex["target_index"]], device=device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())
        if (epoch + 1) % 20 == 0:
            print("epoch", epoch + 1, "loss", round(total / len(train_ex), 4),
                  flush=True)

    @torch.no_grad()
    def evaluate(examples):
        model_r.eval()
        top1 = top5 = total = 0
        for ex in examples:
            scores = model_r(ex["features"].to(device))
            order = torch.argsort(-scores).tolist()
            ranking = [ex["candidates"][i] for i in order]
            target = ex["target"]
            total += 1
            if ranking[0] == target:
                top1 += 1
            if target in ranking[:5]:
                top5 += 1
        return {"units": total, "top1": top1 / max(total, 1),
                "top5": top5 / max(total, 1)}

    rerank = evaluate(test_ex)
    head_hits = 0
    ppm_hits = 0
    for ex in test_ex:
        head = ex["sources"]["head"][:5]
        ppm = ex["sources"]["ppm"][:5]
        if ex["target"] in head:
            head_hits += 1
        if ex["target"] in ppm:
            ppm_hits += 1
    report["reranker"] = rerank
    report["head_top5"] = head_hits / max(len(test_ex), 1)
    report["ppm_top5"] = ppm_hits / max(len(test_ex), 1)
    report["seconds"] = round(time.perf_counter() - t0, 1)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
