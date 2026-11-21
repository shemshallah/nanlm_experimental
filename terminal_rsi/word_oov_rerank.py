#!/usr/bin/env python3
"""Add the UNK/OOV candidate to the cached union and measure true coverage."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CACHE = Path("/marimo/word_dev_cache.npz")
OUT = Path("/marimo/word_oov_results.json")
K = 100
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
    head_big = np.argpartition(-head, K, axis=1)[:, :K]
    bridge_big = np.argpartition(-bridge, K, axis=1)[:, :K]
    prior_big = np.argpartition(-prior, K, axis=1)[:, :K]
    head_big = np.take_along_axis(
        head_big, np.argsort(-np.take_along_axis(head, head_big, 1), 1), 1)
    bridge_big = np.take_along_axis(
        bridge_big, np.argsort(-np.take_along_axis(bridge, bridge_big, 1), 1), 1)
    prior_big = np.take_along_axis(
        prior_big, np.argsort(-np.take_along_axis(prior, prior_big, 1), 1), 1)
    test = np.where(book >= 8)[0]
    train = np.where(book < 8)[0]
    oov_rate = float(np.mean(target[test] == 0))
    print("records", n, "test", len(test), "oov_target", round(oov_rate, 4),
          flush=True)

    def sources(i):
        return {
            "head": head_big[i].tolist(),
            "bridge": bridge_big[i].tolist(),
            "prior": prior_big[i].tolist(),
            "ppm": [int(w) for w in ppm[i] if w >= 0],
            "cache": [int(w) for w in cache[i] if w >= 0],
            "copy": [int(w) for w in copy[i] if w >= 0],
        }

    def candidates(i, with_unk=True):
        cand = []
        for ids in sources(i).values():
            for w in ids:
                w = int(w)
                if w not in cand:
                    cand.append(w)
        if with_unk and 0 not in cand:
            cand.append(0)
        return cand

    def recall(indices, with_unk=True):
        hit = 0
        for i in indices:
            if int(target[i]) in candidates(i, with_unk):
                hit += 1
        return hit / max(len(indices), 1)

    def build(indices):
        out = []
        for i in indices:
            cand = candidates(i)
            tgt = int(target[i])
            if tgt not in cand:
                continue
            src = sources(i)
            feats = []
            for c in cand:
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
                      1.0 if c == 0 else 0.0,
                      1.0 if c == int(prev[i, 0]) else 0.0,
                      1.0 if c == int(prev[i, 1]) else 0.0,
                      1.0 if c == int(prev[i, 2]) else 0.0]
                feats.append(f)
            out.append({"candidates": cand,
                        "features": torch.tensor(feats, dtype=torch.float32),
                        "target_index": cand.index(tgt), "target": tgt})
        return out

    class Reranker(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, 64), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    train_ex = build(train)
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
    def evaluate(indices):
        model.eval()
        top1 = top5 = covered = cond5 = 0
        unk_top5 = 0
        for i in indices:
            cand = candidates(i)
            tgt = int(target[i])
            if tgt not in cand:
                continue
            covered += 1
            src = sources(i)
            feats = []
            for c in cand:
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
                      1.0 if c == 0 else 0.0,
                      1.0 if c == int(prev[i, 0]) else 0.0,
                      1.0 if c == int(prev[i, 1]) else 0.0,
                      1.0 if c == int(prev[i, 2]) else 0.0]
                feats.append(f)
            scores = model(torch.tensor(feats, dtype=torch.float32).to(DEVICE))
            order = torch.argsort(-scores).tolist()
            ranking = [cand[j] for j in order]
            if ranking[0] == tgt:
                top1 += 1
            if tgt in ranking[:5]:
                top5 += 1
                cond5 += 1
                if tgt == 0:
                    unk_top5 += 1
        return {"all": len(indices), "covered": covered,
                "coverage": covered / max(len(indices), 1),
                "top1_all": top1 / max(len(indices), 1),
                "top5_all": top5 / max(len(indices), 1),
                "top5_conditional": cond5 / max(covered, 1),
                "oov_top5_all": unk_top5 / max(len(indices), 1)}

    report = {
        "records": n, "oov_target_rate": oov_rate,
        "recall@100_no_unk": recall(test, False),
        "recall@100_with_unk": recall(test, True),
        "reranker": evaluate(test),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
