#!/usr/bin/env python3
"""Two-stage word selector over a 1,000-2,000 candidate union.

Stage 1: listwise MLP over head/bridge/prior top-1000 + PPM/cache/copy.
Stage 2: listwise MLP reranks the stage-1 shortlist.
Evaluates true overall top-5 (candidate misses count as failures).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CACHE = Path("/marimo/word_dev_cache.npz")
OUT = Path("/marimo/word_two_stage_results.json")
K_BIG = 1000
K_SMALL = 100
STAGE1_TOP = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> int:
    t0 = time.perf_counter()
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
    test = np.where(book >= 8)[0]
    train = np.where(book < 8)[0]
    print("records", n, "train", len(train), "test", len(test), flush=True)

    head_order = np.argsort(-head, axis=1)
    bridge_order = np.argsort(-bridge, axis=1)
    prior_order = np.argsort(-prior, axis=1)

    def build_example(i):
        src_lists = {
            "head": head_order[i, :K_BIG].tolist(),
            "bridge": bridge_order[i, :K_BIG].tolist(),
            "prior": prior_order[i, :K_BIG].tolist(),
            "ppm": [int(w) for w in ppm[i] if w >= 0][:K_SMALL],
            "cache": [int(w) for w in cache[i] if w >= 0][:K_SMALL],
            "copy": [int(w) for w in copy[i] if w >= 0][:K_SMALL],
        }
        rank = {name: {int(w): r for r, w in enumerate(ids)}
                for name, ids in src_lists.items()}
        cand = []
        for ids in src_lists.values():
            for w in ids:
                w = int(w)
                if w not in cand:
                    cand.append(w)
        if 0 not in cand:
            cand.append(0)
        tgt = int(target[i])
        if tgt not in cand:
            return None
        feats = []
        for c in cand:
            f = []
            for name in ("head", "bridge", "prior"):
                r = rank[name].get(c, -1)
                f.append(1.0 / (r + 1) if r >= 0 else 0.0)
                f.append(float(head[i, c] if name == "head"
                               else (bridge[i, c] if name == "bridge"
                                     else prior[i, c])) / 10.0)
            for name in ("ppm", "cache", "copy"):
                r = rank[name].get(c, -1)
                f.append(1.0 / (r + 1) if r >= 0 else 0.0)
            f += [1.0 if c == 0 else 0.0,
                  1.0 if c == int(prev[i, 0]) else 0.0,
                  1.0 if c == int(prev[i, 1]) else 0.0,
                  1.0 if c == int(prev[i, 2]) else 0.0,
                  1.0 if c in rank["cache"] else 0.0,
                  1.0 if c in rank["copy"] else 0.0]
            feats.append(f)
        return {"candidates": cand,
                "features": torch.tensor(feats, dtype=torch.float32),
                "target_index": cand.index(tgt), "target": tgt}

    train_ex = [ex for ex in (build_example(i) for i in train) if ex]
    test_ex = [ex for ex in (build_example(i) for i in test) if ex]
    print("examples", len(train_ex), len(test_ex),
          "seconds", round(time.perf_counter() - t0, 1), flush=True)

    class Net(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, 64), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    class LinearRanker(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc = nn.Linear(dim, 1)

        def forward(self, x):
            return self.fc(x).squeeze(-1)

    dim = len(train_ex[0]["features"][0])
    stage1 = Net(dim).to(DEVICE)
    opt1 = torch.optim.AdamW(stage1.parameters(), lr=1e-3, weight_decay=1e-2)
    for epoch in range(30):
        stage1.train()
        total = 0.0
        for ex in train_ex:
            scores = stage1(ex["features"].to(DEVICE))
            loss = F.cross_entropy(scores[None, :], torch.tensor(
                [ex["target_index"]], device=DEVICE))
            opt1.zero_grad(set_to_none=True)
            loss.backward()
            opt1.step()
            total += float(loss.detach())
        if (epoch + 1) % 10 == 0:
            print("stage1", epoch + 1, round(total / len(train_ex), 4),
                  flush=True)

    @torch.no_grad()
    def shortlist(examples, top=STAGE1_TOP):
        stage1.eval()
        out = []
        for ex in examples:
            scores = stage1(ex["features"].to(DEVICE))
            order = torch.argsort(-scores)[:top].tolist()
            cand = [ex["candidates"][j] for j in order]
            idx = torch.tensor(order)
            out.append({"candidates": cand,
                        "features": ex["features"][idx],
                        "target": ex["target"],
                        "target_index": (cand.index(ex["target"])
                                         if ex["target"] in cand else -1)})
        return out

    train_s = shortlist(train_ex)
    test_s = shortlist(test_ex)
    stage1_recall = sum(ex["target_index"] >= 0 for ex in test_s) / len(test_s)
    print("stage1 shortlist recall", round(stage1_recall, 4), flush=True)

    stage2 = Net(dim).to(DEVICE)
    opt2 = torch.optim.AdamW(stage2.parameters(), lr=1e-3, weight_decay=1e-2)
    train_s = [ex for ex in train_s if ex["target_index"] >= 0]
    for epoch in range(120):
        stage2.train()
        total = 0.0
        for ex in train_s:
            scores = stage2(ex["features"].to(DEVICE))
            loss = F.cross_entropy(scores[None, :], torch.tensor(
                [ex["target_index"]], device=DEVICE))
            opt2.zero_grad(set_to_none=True)
            loss.backward()
            opt2.step()
            total += float(loss.detach())
        if (epoch + 1) % 30 == 0:
            print("stage2", epoch + 1, round(total / len(train_s), 4),
                  flush=True)

    @torch.no_grad()
    def evaluate(examples):
        stage2.eval()
        top1 = top5 = 0
        cond = 0
        for ex in examples:
            if ex["target_index"] < 0:
                continue
            scores = stage2(ex["features"].to(DEVICE))
            order = torch.argsort(-scores).tolist()
            ranking = [ex["candidates"][j] for j in order]
            if ranking[0] == ex["target"]:
                top1 += 1
            if ex["target"] in ranking[:5]:
                top5 += 1
                cond += 1
        total = len(examples)
        return {"all": total, "covered": sum(
            ex["target_index"] >= 0 for ex in examples),
            "top1_all": top1 / total, "top5_all": top5 / total,
            "top5_conditional": cond / max(
                sum(ex["target_index"] >= 0 for ex in examples), 1)}

    report = {
        "records": n, "train_covered": len(train_ex),
        "test_covered": len(test_ex),
        "candidate_coverage": len(test_ex) / len(test),
        "stage1_shortlist_recall": stage1_recall,
        "selector": evaluate(test_s),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
