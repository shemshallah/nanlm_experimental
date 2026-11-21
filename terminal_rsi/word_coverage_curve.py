#!/usr/bin/env python3
"""True candidate coverage as a function of top-K from each source."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CACHE = Path("/marimo/word_dev_cache.npz")
OUT = Path("/marimo/word_coverage_curve.json")


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
    test = np.where(book >= 8)[0]
    big = {}
    for name, arr in (("head", head), ("bridge", bridge), ("prior", prior)):
        order = np.argsort(-arr, axis=1)
        big[name] = order
    report = {}
    for k in (20, 50, 100, 200, 500, 1000, 2000):
        hit = 0
        for i in test:
            union = set()
            for name in ("head", "bridge", "prior"):
                union.update(int(w) for w in big[name][i, :k])
            for arr in (ppm, cache, copy):
                union.update(int(w) for w in arr[i] if w >= 0)
            if int(target[i]) in union:
                hit += 1
        report[f"recall@{k}"] = hit / len(test)
    # Per-source coverage at 100 and 1000.
    for name in ("head", "bridge", "prior"):
        for k in (100, 1000):
            hit = sum(int(target[i]) in set(int(w) for w in big[name][i, :k])
                      for i in test)
            report[f"{name}@{k}"] = hit / len(test)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
