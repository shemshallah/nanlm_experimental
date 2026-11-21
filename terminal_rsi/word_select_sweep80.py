#!/usr/bin/env python3
"""Sweep candidate-set size K for a single-stage 80 MB word selector."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CACHE = Path("/marimo/word_dev_cache80.npz")
OUT = Path("/marimo/word_select_sweep80.json")
SPLIT = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main() -> int:
    t0 = time.perf_counter()
    data = np.load(CACHE)
    head = data["head"].astype(np.float32)
    bridge = data["bridge"].astype(np.float32)
    prior = data["prior"].astype(np.float32)
    ppm, cache, copy = data["ppm"], data["cache"], data["copy"]
    target, book = data["target"], data["book"]
    prev = data["prev"]
    train_idx = np.where(book < SPLIT)[0]
    test_idx = np.where(book >= SPLIT)[0]
    orders = {name: np.argsort(-arr, axis=1)
              for name, arr in (("head", head), ("bridge", bridge),
                                ("prior", prior))}
    print("records", len(target), "train", len(train_idx),
          "test", len(test_idx), flush=True)

    def sources(i, k):
        return {
            "head": orders["head"][i, :k].tolist(),
            "bridge": orders["bridge"][i, :k].tolist(),
            "prior": orders["prior"][i, :k].tolist(),
            "ppm": [int(w) for w in ppm[i] if w >= 0][:k],
            "cache": [int(w) for w in cache[i] if w >= 0][:k],
            "copy": [int(w) for w in copy[i] if w >= 0][:k],
        }

    def build(indices, k):
        examples = []
        for i in indices:
            src = sources(i, k)
            cand = []
            for ids in src.values():
                for w in ids:
                    w = int(w)
                    if w not in cand:
                        cand.append(w)
            if 0 not in cand:
                cand.append(0)
            tgt = int(target[i])
            if tgt not in cand:
                continue
            feats = []
            for c in cand:
                f = []
                for name in ("head", "bridge", "prior"):
                    try:
                        r = src[name].index(c)
                    except ValueError:
                        r = -1
                    f.append(1.0 / (r + 1) if r >= 0 else 0.0)
                    f.append(float(head[i, c] if name == "head"
                                   else (bridge[i, c] if name == "bridge"
                                         else prior[i, c])) / 10.0)
                for name in ("ppm", "cache", "copy"):
                    try:
                        r = src[name].index(c)
                    except ValueError:
                        r = -1
                    f.append(1.0 / (r + 1) if r >= 0 else 0.0)
                f += [1.0 if c == 0 else 0.0,
                      1.0 if c == int(prev[i, 0]) else 0.0,
                      1.0 if c == int(prev[i, 1]) else 0.0,
                      1.0 if c == int(prev[i, 2]) else 0.0,
                      1.0 if c in src["cache"] else 0.0,
                      1.0 if c in src["copy"] else 0.0]
                feats.append(f)
            examples.append({"candidates": cand,
                             "features": torch.tensor(feats, dtype=torch.float32),
                             "target": tgt,
                             "target_index": cand.index(tgt)})
        return examples

    report = {}
    for k in (50, 100, 200, 500, 1000):
        tr = build(train_idx, k)
        te = build(test_idx, k)
        model = Net(15).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        for epoch in range(25):
            model.train()
            total = 0.0
            for ex in tr:
                scores = model(ex["features"].to(DEVICE))
                loss = F.cross_entropy(scores[None, :], torch.tensor(
                    [ex["target_index"]], device=DEVICE))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss.detach())
        model.eval()
        top1 = top5 = 0
        with torch.no_grad():
            for ex in te:
                scores = model(ex["features"].to(DEVICE))
                order = torch.argsort(-scores).tolist()
                ranking = [ex["candidates"][j] for j in order]
                if ranking[0] == ex["target"]:
                    top1 += 1
                if ex["target"] in ranking[:5]:
                    top5 += 1
        coverage = len(te) / len(test_idx)
        report[f"K{k}"] = {
            "train_covered": len(tr), "test_covered": len(te),
            "coverage": coverage,
            "top1_all": top1 / len(test_idx),
            "top5_all": top5 / len(test_idx),
            "top5_conditional": top5 / max(len(te), 1),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        print(f"K{k}", json.dumps(report[f"K{k}"]), flush=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
