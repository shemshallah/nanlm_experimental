#!/usr/bin/env python3
"""r31 10-minute v2: 151M hybrid, router both vs geometry-only.

Both arms: 220 steps x batch 32 x context 256 = 1,802,240 positions, same
data/seed. The already-completed plain 21M arm from r31-10min is the dense
reference (same steps/data/seed).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
CTX = 256
BATCH = 32
STEPS = 220


def first(pattern: str) -> Path:
    hits = sorted(INPUT.rglob(pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def run(cmd, log_path, timeout, gpu=None):
    print("\n$ " + " ".join(cmd), flush=True)
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["OMP_NUM_THREADS"] = "2"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    started = time.time()
    timed_out = False
    with log_path.open("w") as log:
        try:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=timeout, env=env)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            code = -9
    out = {"cmd": cmd, "returncode": code, "timed_out": timed_out,
           "seconds": round(time.time() - started, 2), "log": str(log_path)}
    print(json.dumps(out), flush=True)
    return out


def main() -> int:
    import torch

    code_src = first("probe_r31.py").parent
    data_dir = first("train.bin").parent
    code_dir = WORK / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    for name in ("poincare_frontier.py", "probe_r31.py", "gen_r31.py", "heldout_prompt.txt"):
        shutil.copy2(code_src / name, code_dir / name)
    sys.path.insert(0, str(code_dir))
    import poincare_frontier as pf

    ngpu = torch.cuda.device_count()
    print(f"gpu_count={ngpu} devices={[torch.cuda.get_device_name(i) for i in range(ngpu)]}", flush=True)
    if ngpu < 2:
        return 2

    common = [
        "--dim", "544", "--layers", "4", "--heads", "16",
        "--pairs", "128", "--topk", "2", "--expert-hidden", "235", "--moe-every", "2",
        "--geo", "1.0", "--balance", "0.01",
        "--moe-additive",
    ]
    base = [
        sys.executable, "-u", str(code_dir / "poincare_frontier.py"), "train",
        "--data-dir", str(data_dir), "--steps", str(STEPS),
        "--context", str(CTX), "--batch", str(BATCH), "--grad-accum", "1",
        "--future2", "0.3", "--future3", "0.15", "--future-mode", "rollout",
        "--lr", "3e-4", "--eval-every", "100", "--eval-batches", "2",
        "--save-every", "110", "--seed", "1337",
    ]
    jobs = {
        "hybrid_both": {"gpu": 1, "router": "both"},
        "hybrid_geo": {"gpu": 0, "router": "geo"},
    }
    for name, job in jobs.items():
        job["run"] = WORK / name
        shutil.rmtree(job["run"], ignore_errors=True)
        job["run"].mkdir(parents=True, exist_ok=True)
        job["cmd"] = base + ["--run-dir", str(job["run"])] + common + ["--router-mode", job["router"]]

    results = {}

    def worker(name):
        job = jobs[name]
        print(f"[gpu{job['gpu']}] starting {name}", flush=True)
        results[name] = {"train": run(job["cmd"], WORK / f"{name}.train.log", 780, job["gpu"])}
        ckpt = job["run"] / "latest.pt"
        results[name]["checkpoint_exists"] = ckpt.exists()
        if not ckpt.exists():
            return
        results[name]["probe_run"] = run(
            [sys.executable, "-u", str(code_dir / "probe_r31.py"),
             "--data-dir", str(data_dir), "--run-dir", str(job["run"]),
             "--context", str(CTX), "--batch", "8", "--batches", "32", "--book-windows", "4",
             "--out", str(WORK / f"{name}.probe.json")],
            WORK / f"{name}.probe.log", 300, job["gpu"])
        try:
            results[name]["probe"] = json.loads((WORK / f"{name}.probe.json").read_text())
        except Exception as exc:  # noqa: BLE001
            results[name]["probe_error"] = repr(exc)
        results[name]["gen_run"] = run(
            [sys.executable, "-u", str(code_dir / "gen_r31.py"), "--run-dir", str(job["run"]),
             "--prompt-file", str(code_dir / "heldout_prompt.txt"),
             "--max-new-tokens", "200", "--temperature", "0.8", "--topk-sample", "30",
             "--out", str(WORK / f"{name}.gen.json")],
            WORK / f"{name}.gen.log", 300, job["gpu"])
        try:
            results[name]["gen"] = json.loads((WORK / f"{name}.gen.json").read_text())
        except Exception as exc:  # noqa: BLE001
            results[name]["gen_error"] = repr(exc)

    threads = [threading.Thread(target=worker, args=(name,)) for name in jobs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    def bpb(name):
        p = results.get(name, {}).get("probe")
        return p["bpb"] if p else None

    summary = {
        "design": {
            "positions_per_arm": STEPS * BATCH * CTX,
            "steps": STEPS, "batch": BATCH, "context": CTX, "seed": 1337,
            "arms": "hybrid151 both-router vs geometry-only; params 151,005,954; active ~21.1M MAC/token",
            "dense_reference": "plain 21,388,032 from r31-10min, same 220 steps/data/seed",
        },
        "results": results,
        "comparison": {
            "geo_minus_both_bpb": (round(bpb("hybrid_geo") - bpb("hybrid_both"), 6)
                                   if bpb("hybrid_geo") is not None and bpb("hybrid_both") is not None else None),
        },
    }
    (WORK / "r31_10min_v2_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nR31 10MIN V2 SUMMARY\n" + json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
