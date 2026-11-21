#!/usr/bin/env python3
"""Parse Terminal-RSI progress logs and extrapolate training curves."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

PROGRESS = re.compile(
    r"legacy_step=(\d+).*?elapsed=(\d+)s/.*?upgrade_byte_bpb=([0-9.]+)")
BIT = re.compile(r"legacy_step=(\d+).*?bit_acc=([0-9.]+)")


def parse(paths):
    points = {}
    bits = {}
    for path in paths:
        text = Path(path).read_text(errors="replace")
        for line in text.splitlines():
            m = PROGRESS.search(line)
            if m:
                step, elapsed, bpb = int(m.group(1)), int(m.group(2)), float(m.group(3))
                points[step] = (elapsed, bpb)
            b = BIT.search(line)
            if b:
                bits[int(b.group(1))] = float(b.group(2))
    rows = [(step, *points[step]) for step in sorted(points)]
    return rows, bits


def smooth(rows, window=25):
    if len(rows) < window:
        return rows
    steps = np.array([r[0] for r in rows])
    bpb = np.array([r[2] for r in rows])
    out = []
    for i in range(len(rows)):
        lo, hi = max(0, i - window // 2), min(len(rows), i + window // 2 + 1)
        out.append((steps[i], rows[i][1], float(np.median(bpb[lo:hi]))))
    return out


def fits(rows):
    x = np.array([r[0] for r in rows], dtype=float)
    y = np.array([r[2] for r in rows], dtype=float)
    x0 = x.min()
    result = {}
    try:
        from scipy.optimize import curve_fit

        def exp_model(xx, a, b, tau):
            return a + b * np.exp(-(xx - x0) / tau)

        popt, _ = curve_fit(
            exp_model, x, y, p0=(max(1.0, y.min() - 0.2), 1.0, 3000.0),
            bounds=([0.8, 0.0, 50.0], [2.6, 20.0, 200000.0]), maxfev=20000)
        pred = exp_model(x, *popt)
        result["exp"] = {"a": float(popt[0]), "b": float(popt[1]),
                         "tau": float(popt[2]),
                         "rmse": float(np.sqrt(np.mean((pred - y) ** 2)))}

        def pow_model(xx, a, b, c):
            return a + b * np.maximum(xx, 1.0) ** (-c)

        popt2, _ = curve_fit(
            pow_model, x, y, p0=(1.0, 100.0, 0.3),
            bounds=([0.5, 0.0, 0.01], [3.0, 1e7, 2.0]), maxfev=20000)
        pred2 = pow_model(x, *popt2)
        result["power"] = {"a": float(popt2[0]), "b": float(popt2[1]),
                           "c": float(popt2[2]),
                           "rmse": float(np.sqrt(np.mean((pred2 - y) ** 2)))}
    except Exception as exc:
        result["error"] = str(exc)
    return result


def predict(model, step):
    if "exp" in model:
        m = model["exp"]
        return m["a"] + m["b"] * math.exp(-(step - 2400) / m["tau"])
    return float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-step", type=int, default=2400)
    args = parser.parse_args()
    rows, bits = parse(args.log)
    rows = [r for r in rows if r[0] >= args.min_step]
    if not rows:
        raise SystemExit("no progress rows")
    smoothed = smooth(rows)
    model = fits(smoothed)
    last_step, last_elapsed, last_bpb = rows[-1]
    # Median observed update rate over the last ~200 points.
    tail = rows[-200:]
    if len(tail) >= 2:
        rate = (tail[-1][0] - tail[0][0]) / max(tail[-1][1] - tail[0][1], 1e-9)
    else:
        rate = 0.18
    durations = [("20 min", 1200), ("1 h", 3600), ("4 h", 14400),
                 ("6 h", 21600), ("12 h", 43200), ("24 h", 86400)]
    table = []
    for label, seconds in durations:
        extra = rate * seconds
        step = last_step + extra
        table.append({
            "duration": label,
            "additional_updates": int(round(extra)),
            "total_step": int(round(step)),
            "predicted_train_bpb": round(predict(model, step), 4),
            "wall_hours": round(seconds / 3600, 2),
            "checkpoints_10min": int(math.ceil(seconds / 600)),
        })
    observed = []
    for row in smoothed[::max(1, len(smoothed) // 20)]:
        observed.append({"step": int(row[0]), "elapsed_s": int(row[1]),
                         "train_bpb": round(row[2], 4)})
    report = {
        "last_step": last_step, "last_elapsed_s": last_elapsed,
        "last_train_bpb": last_bpb,
        "observed_bit_accuracy_last": bits.get(last_step),
        "observed_rate_updates_per_s": round(rate, 4),
        "updates_per_hour": int(round(rate * 3600)),
        "points": len(rows),
        "fit": model,
        "observed": observed,
        "extrapolation": table,
        "note": "Train-batch BPB is noisy; the fit is a trend, not a guarantee. "
                "DEV/WikiText are measured only at checkpoints.",
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
