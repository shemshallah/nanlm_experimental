#!/usr/bin/env python3
"""Hierarchical cluster shortlist on the 80 MB word cache."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

CACHE = Path("/marimo/word_dev_cache80.npz")
CKPT = Path("/marimo/trsi_ckpt_6h/last.pt")
OUT = Path("/marimo/word_cluster_shortlist80.json")
SPLIT = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> int:
    t0 = time.perf_counter()
    data = np.load(CACHE)
    head = data["head"].astype(np.float32)
    bridge = data["bridge"].astype(np.float32)
    prior = data["prior"].astype(np.float32)
    ppm, cache, copy = data["ppm"], data["cache"], data["copy"]
    target, book = data["target"], data["book"]
    prev = data["prev"]
    n = len(target)
    test = np.where(book >= SPLIT)[0]
    train = np.where(book < SPLIT)[0]
    print("records", n, flush=True)

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    key = "upgrades.hierarchy.unit_heads.word.weight"
    if key not in ckpt["model"]:
        key = next(k for k in ckpt["model"]
                   if k.endswith("unit_heads.word.weight"))
    emb = ckpt["model"][key].float().numpy()
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    clusters = KMeans(n_clusters=128, n_init=4, random_state=2026).fit_predict(emb)
    order = np.argsort(clusters, kind="stable")
    sorted_clusters = clusters[order]
    starts = np.flatnonzero(np.r_[True, sorted_clusters[1:] != sorted_clusters[:-1]])
    members = np.split(order, starts[1:])
    print("clusters", len(members), "sizes",
          [len(m) for m in members[:5]], flush=True)

    def score_vector(i):
        score = head[i] / 10.0 + bridge[i] / 10.0 + prior[i] / 10.0
        for arr in (ppm, cache, copy):
            for rank, w in enumerate(arr[i]):
                if w >= 0:
                    score[int(w)] += 2.0 / (rank + 1)
        score -= score.max()
        return score

    def shortlist(i, n_clusters, mode):
        score = score_vector(i)
        cluster_scores = np.array([score[m].max() for m in members])
        if mode == "top2":
            cluster_scores = np.array([
                np.sort(score[m])[-2:].mean() for m in members])
        chosen = np.argsort(-cluster_scores)[:n_clusters]
        ids = np.concatenate([members[c] for c in chosen])
        rank = ids[np.argsort(-score[ids])[:50]]
        return rank

    recall = {}
    for mode in ("max", "top2"):
        for c in (4, 8, 16, 32, 64):
            hit = 0
            for i in test:
                if int(target[i]) in set(shortlist(i, c, mode)):
                    hit += 1
            recall[f"{mode}_C{c}"] = hit / len(test)
    print("recall", json.dumps(recall), flush=True)

    # Best config by recall; fit a small selector on the cluster shortlist.
    best = max(recall, key=recall.get)
    mode, cstr = best.split("_")
    n_clusters = int(cstr[1:])
    print("best", best, recall[best], flush=True)

    def features(i):
        score = score_vector(i)
        ids = shortlist(i, n_clusters, mode)
        src = {
            "head": np.argsort(-head[i])[:100].tolist(),
            "bridge": np.argsort(-bridge[i])[:100].tolist(),
            "prior": np.argsort(-prior[i])[:100].tolist(),
            "ppm": [int(w) for w in ppm[i] if w >= 0],
            "cache": [int(w) for w in cache[i] if w >= 0],
            "copy": [int(w) for w in copy[i] if w >= 0],
        }
        feats = []
        for w in ids:
            f = []
            for name in ("head", "bridge", "prior"):
                r = src[name].index(w) if w in src[name] else -1
                f += [1.0 / (r + 1) if r >= 0 else 0.0,
                      float(head[i, w] if name == "head"
                            else (bridge[i, w] if name == "bridge"
                                  else prior[i, w])) / 10.0]
            for name in ("ppm", "cache", "copy"):
                r = src[name].index(w) if w in src[name] else -1
                f.append(1.0 / (r + 1) if r >= 0 else 0.0)
            f += [1.0 if w == 0 else 0.0,
                  1.0 if w == int(prev[i, 0]) else 0.0,
                  1.0 if w == int(prev[i, 1]) else 0.0,
                  1.0 if w == int(prev[i, 2]) else 0.0,
                  float(score[w])]
            feats.append(f)
        tgt = int(target[i])
        return {"ids": ids, "features": torch.tensor(feats, dtype=torch.float32),
                "target": tgt, "target_index": ids.index(tgt)
                if tgt in ids else -1}

    tr = [features(i) for i in train]
    te = [features(i) for i in test]
    tr = [x for x in tr if x["target_index"] >= 0]
    print("train shortlist", len(tr), "test covered",
          sum(x["target_index"] >= 0 for x in te), flush=True)

    class Net(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, 64), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    dim = len(tr[0]["features"][0])
    model = Net(dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    for epoch in range(40):
        model.train()
        for ex in tr:
            scores = model(ex["features"].to(DEVICE))
            loss = F.cross_entropy(scores[None, :], torch.tensor(
                [ex["target_index"]], device=DEVICE))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    top1 = top5 = 0
    with torch.no_grad():
        for ex in te:
            if ex["target_index"] < 0:
                continue
            scores = model(ex["features"].to(DEVICE))
            order = torch.argsort(-scores).tolist()
            ranking = [ex["ids"][j] for j in order]
            if ranking[0] == ex["target"]:
                top1 += 1
            if ex["target"] in ranking[:5]:
                top5 += 1
    covered = sum(x["target_index"] >= 0 for x in te)
    report = {
        "records": n, "best": best, "shortlist_recall": recall[best],
        "shortlist_size": 50, "recalls": recall,
        "selector": {"all": len(te), "covered": covered,
                     "coverage": covered / len(te),
                     "top1_all": top1 / len(te),
                     "top5_all": top5 / len(te),
                     "top5_conditional": top5 / max(covered, 1)},
        "seconds": round(time.perf_counter() - t0, 1),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
