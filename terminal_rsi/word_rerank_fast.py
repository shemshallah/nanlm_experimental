#!/usr/bin/env python3
"""Fast rerank evaluation from the cached dev features.

Reports true candidate recall over ALL test records (not only covered ones),
conditional top-5, and overall top-5.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CACHE = Path("/marimo/word_dev_cache.npz")
OUT = Path("/marimo/word_rerank_fast_results.json")
TOPK = 20
TOPK_BIG = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> int:
    data = np.load(CACHE)
    head = data["head"].astype(np.float32)
    bridge = data["bridge"].astype(np.float32)
    prior = data["prior"].astype(np.float32)
    ppm = data["ppm"]
    cache = data["cache"]
    copy = data["copy"]
    target = data["target"]
    book = data["book"]
    prev = data["prev"]
    n = len(target)
    print("records", n, flush=True)

    head_big = np.argpartition(-head, TOPK_BIG, axis=1)[:, :TOPK_BIG]
    bridge_big = np.argpartition(-bridge, TOPK_BIG, axis=1)[:, :TOPK_BIG]
    prior_big = np.argpartition(-prior, TOPK_BIG, axis=1)[:, :TOPK_BIG]
    # Sort each row's big top-k descending for deterministic rank features.
    head_big = np.take_along_axis(
        head_big, np.argsort(-np.take_along_axis(head, head_big, 1), 1), 1)
    bridge_big = np.take_along_axis(
        bridge_big, np.argsort(-np.take_along_axis(bridge, bridge_big, 1), 1), 1)
    prior_big = np.take_along_axis(
        prior_big, np.argsort(-np.take_along_axis(prior, prior_big, 1), 1), 1)

    def sources_at(i, k=TOPK):
        return {
            "head": head_big[i, :k].tolist(),
            "bridge": bridge_big[i, :k].tolist(),
            "prior": prior_big[i, :k].tolist(),
            "ppm": [int(w) for w in ppm[i, :k] if w >= 0],
            "cache": [int(w) for w in cache[i, :k] if w >= 0],
            "copy": [int(w) for w in copy[i, :k] if w >= 0],
        }

    test = np.where(book >= 8)[0]
    train = np.where(book < 8)[0]

    def union_recall(indices, k=TOPK):
        hit = 0
        for i in indices:
            src = sources_at(i, k=k)
            union = set()
            for ids in src.values():
                union.update(int(w) for w in ids)
            if int(target[i]) in union:
                hit += 1
        return hit / max(len(indices), 1)

    def head_ppm_top5(indices):
        h = p = 0
        for i in indices:
            if int(target[i]) in head_big[i, :5]:
                h += 1
            if int(target[i]) in [int(w) for w in ppm[i, :5] if w >= 0]:
                p += 1
        return h / max(len(indices), 1), p / max(len(indices), 1)

    def oracle_top5(indices):
        hit = 0
        for i in indices:
            union = set()
            src = sources_at(i, k=TOPK)
            for ids in src.values():
                union.update(int(w) for w in ids[:5])
            if int(target[i]) in union:
                hit += 1
        return hit / max(len(indices), 1)

    def build_train_examples():
        examples = []
        for i in train:
            src = sources_at(i, k=TOPK)
            candidates = []
            for ids in src.values():
                for w in ids:
                    w = int(w)
                    if w not in candidates:
                        candidates.append(w)
            tgt = int(target[i])
            if tgt not in candidates:
                continue
            feats = []
            for c in candidates:
                f = []
                for name in ("head", "bridge", "prior", "ppm", "cache", "copy"):
                    try:
                        rank = src[name].index(c)
                    except ValueError:
                        rank = -1
                    f.append(1.0 / (rank + 1) if rank >= 0 else 0.0)
                f += [head[i, c] / 10.0, bridge[i, c] / 10.0,
                      prior[i, c] / 10.0, 1.0 if c in src["cache"] else 0.0,
                      1.0 if c in src["copy"] else 0.0,
                      1.0 if c == int(prev[i, 0]) else 0.0,
                      1.0 if c == int(prev[i, 1]) else 0.0,
                      1.0 if c == int(prev[i, 2]) else 0.0]
                feats.append(f)
            examples.append({
                "candidates": candidates,
                "features": torch.tensor(feats, dtype=torch.float32),
                "target_index": candidates.index(tgt),
                "target": tgt,
            })
        return examples

    class Reranker(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, 64), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    train_ex = build_train_examples()
    print("train covered", len(train_ex), "of", len(train), flush=True)
    dim = len(train_ex[0]["features"][0])
    model = Reranker(dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    for epoch in range(150):
        model.train()
        total = 0.0
        for ex in train_ex:
            scores = model(ex["features"].to(DEVICE))
            loss = F.cross_entropy(scores[None, :], torch.tensor(
                [ex["target_index"]], device=DEVICE))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())
        if (epoch + 1) % 30 == 0:
            print("epoch", epoch + 1, "loss", round(total / len(train_ex), 4),
                  flush=True)

    @torch.no_grad()
    def eval_all(indices, use_rerank=True):
        model.eval()
        top1 = top5 = covered = 0
        cond_top5 = 0
        for i in indices:
            src = sources_at(i, k=TOPK)
            candidates = []
            for ids in src.values():
                for w in ids:
                    w = int(w)
                    if w not in candidates:
                        candidates.append(w)
            tgt = int(target[i])
            if tgt not in candidates:
                continue
            covered += 1
            if use_rerank:
                feats = []
                for c in candidates:
                    f = []
                    for name in ("head", "bridge", "prior", "ppm", "cache", "copy"):
                        try:
                            rank = src[name].index(c)
                        except ValueError:
                            rank = -1
                        f.append(1.0 / (rank + 1) if rank >= 0 else 0.0)
                    f += [head[i, c] / 10.0, bridge[i, c] / 10.0,
                          prior[i, c] / 10.0,
                          1.0 if c in src["cache"] else 0.0,
                          1.0 if c in src["copy"] else 0.0,
                          1.0 if c == int(prev[i, 0]) else 0.0,
                          1.0 if c == int(prev[i, 1]) else 0.0,
                          1.0 if c == int(prev[i, 2]) else 0.0]
                    feats.append(f)
                scores = model(torch.tensor(feats, dtype=torch.float32).to(DEVICE))
                order = torch.argsort(-scores).tolist()
                ranking = [candidates[j] for j in order]
                if ranking[0] == tgt:
                    top1 += 1
                if tgt in ranking[:5]:
                    top5 += 1
                    cond_top5 += 1
            else:
                # Simple source-score sum baseline.
                score = {}
                for name, ids in src.items():
                    for rank, w in enumerate(ids):
                        score[int(w)] = score.get(int(w), 0.0) + 1.0 / (rank + 1)
                ranking = [w for w, _ in sorted(
                    score.items(), key=lambda kv: (-kv[1], kv[0]))]
                if ranking[0] == tgt:
                    top1 += 1
                if tgt in ranking[:5]:
                    top5 += 1
                    cond_top5 += 1
        return {"all": len(indices), "covered": covered,
                "coverage": covered / max(len(indices), 1),
                "top1_all": top1 / max(len(indices), 1),
                "top5_all": top5 / max(len(indices), 1),
                "top5_conditional": cond_top5 / max(covered, 1)}

    h5, p5 = head_ppm_top5(test)
    report = {
        "records": n, "train": len(train), "test": len(test),
        "recall@20": union_recall(test, 20),
        "recall@100": union_recall(test, 100),
        "oracle_union_top5": oracle_top5(test),
        "head_top5_all": h5, "ppm_top5_all": p5,
        "reranker": eval_all(test, True),
        "source_sum": eval_all(test, False),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
