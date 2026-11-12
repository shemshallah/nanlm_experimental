#!/usr/bin/env python3
"""R32 multidomain pilot: domain-conditioned hybrid vs dense plain.

16 domain releases, 4 MB sampled per domain, document-level 90/10 split.
220 steps x batch 32 x context 256 = 1.8M positions per arm.
Both arms see a per-document domain token; evaluation is per held-out domain.
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

    code_src = first("probe_r32.py").parent
    data_dir = first("train.bin").parent
    code_dir = WORK / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    for name in ("poincare_frontier.py", "probe_r32.py"):
        shutil.copy2(code_src / name, code_dir / name)
    sys.path.insert(0, str(code_dir))
    import poincare_frontier as pf

    meta = pf.load_meta(data_dir)
    names = meta["domain_names"]
    print(f"domains={len(names)} train={meta['train_tokens']:,} valid={meta['valid_tokens']:,}", flush=True)
    ngpu = torch.cuda.device_count()
    print(f"gpu_count={ngpu}", flush=True)
    if ngpu < 2:
        return 2

    base = [
        sys.executable, "-u", str(code_dir / "poincare_frontier.py"), "train",
        "--data-dir", str(data_dir), "--steps", str(STEPS),
        "--batch", str(BATCH), "--grad-accum", "1",
        "--future2", "0.3", "--future3", "0.15", "--future-mode", "rollout",
        "--lr", "3e-4", "--eval-every", "100", "--eval-batches", "2",
        "--save-every", "110", "--seed", "1337",
        "--context", "257", "--train-context", "256", "--eval-context", "256",
    ]
    jobs = {
        "hybrid151_domain": {
            "gpu": 1,
            "args": ["--dim", "544", "--layers", "4", "--heads", "16",
                     "--pairs", "128", "--topk", "2", "--expert-hidden", "235", "--moe-every", "2",
                     "--geo", "1.0", "--balance", "0.01", "--router-mode", "both",
                     "--moe-additive"],
        },
        "plain21_domain": {
            "gpu": 0,
            "args": ["--dim", "576", "--layers", "4", "--heads", "24",
                     "--pairs", "128", "--topk", "2", "--expert-hidden", "100", "--moe-every", "5",
                     "--geo", "0.0", "--balance", "0.0", "--router-mode", "linear"],
        },
    }
    for name, job in jobs.items():
        job["run"] = WORK / name
        shutil.rmtree(job["run"], ignore_errors=True)
        job["run"].mkdir(parents=True, exist_ok=True)
        job["cmd"] = base + ["--run-dir", str(job["run"])] + job["args"]

    results = {}

    def worker(name):
        job = jobs[name]
        print(f"[gpu{job['gpu']}] starting {name}", flush=True)
        results[name] = {"train": run(job["cmd"], WORK / f"{name}.train.log", 900, job["gpu"])}
        ckpt = job["run"] / "latest.pt"
        results[name]["checkpoint_exists"] = ckpt.exists()
        if not ckpt.exists():
            return
        results[name]["probe_run"] = run(
            [sys.executable, "-u", str(code_dir / "probe_r32.py"),
             "--data-dir", str(data_dir), "--run-dir", str(job["run"]),
             "--context", "256", "--batch", "8", "--batches", "4",
             "--out", str(WORK / f"{name}.probe.json")],
            WORK / f"{name}.probe.log", 600, job["gpu"])
        try:
            results[name]["probe"] = json.loads((WORK / f"{name}.probe.json").read_text())
        except Exception as exc:  # noqa: BLE001
            results[name]["probe_error"] = repr(exc)

    threads = [threading.Thread(target=worker, args=(name,)) for name in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    def bpb(name):
        p = results.get(name, {}).get("probe")
        return p["aggregate"]["bpb"] if p else None

    summary = {
        "design": {
            "domains": names,
            "positions_per_arm": STEPS * BATCH * CTX,
            "steps": STEPS, "batch": BATCH, "context": CTX, "seed": 1337,
            "hybrid151_domain": "151,005,954 total; additive 128-pair top-2 MoE; domain token; linear+geo router",
            "plain21_domain": "21,388,032 dense; domain token",
            "corpus": "16 releases x 4 MB sampled; document-level 90/10 split; 43.7M train / 4.7M valid tokens",
        },
        "results": results,
        "comparison": {
            "hybrid_minus_plain_bpb": (
                round(bpb("hybrid151_domain") - bpb("plain21_domain"), 6)
                if bpb("hybrid151_domain") is not None and bpb("plain21_domain") is not None else None),
        },
    }
    (WORK / "r32_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nR32 SUMMARY\n" + json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
