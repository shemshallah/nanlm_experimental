#!/usr/bin/env python3
"""r31: compute-matched hybrid vs dense, plus router-mode diagnostic.

Main comparison (10M positions, 1,220 steps x batch 32 x context 256):
  plain      21.4M total / 21.4M active
  hybrid151  151.0M total / ~21.1M active (dense trunk + additive 128-pair MoE)

Router diagnostic (2M positions, 244 steps x batch 32 x context 256):
  hybrid_both / hybrid_geo / hybrid_linear, all 151M total.
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


def first(pattern: str) -> Path:
    hits = sorted(INPUT.rglob(pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def run(cmd: list[str], log_path: Path, timeout: int, gpu: int | None = None) -> dict:
    print("\n$ " + " ".join(cmd), flush=True)
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["OMP_NUM_THREADS"] = "2"
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
    result = {"cmd": cmd, "returncode": code, "timed_out": timed_out,
              "seconds": round(time.time() - started, 2), "log": str(log_path)}
    print(json.dumps(result), flush=True)
    return result


def main() -> int:
    import torch

    code_src = first("probe_r31.py").parent
    data_dir = first("train.bin").parent
    prompt = (code_src / "heldout_prompt.txt").read_text(encoding="utf-8").strip()
    code_dir = WORK / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    for name in ("poincare_frontier.py", "probe_r31.py", "gen_r31.py", "heldout_prompt.txt"):
        shutil.copy2(code_src / name, code_dir / name)
    sys.path.insert(0, str(code_dir))
    import poincare_frontier as pf

    ngpu = torch.cuda.device_count()
    print(f"gpu_count={ngpu} devices={[torch.cuda.get_device_name(i) for i in range(ngpu)]}", flush=True)
    if ngpu == 0:
        return 2

    hybrid_common = [
        "--dim", "544", "--layers", "4", "--heads", "16",
        "--pairs", "128", "--topk", "2", "--expert-hidden", "235", "--moe-every", "2",
        "--geo", "1.0", "--balance", "0.01",
        "--moe-additive", "--dispatch-mode", "gather", "--gather-chunk", "64",
    ]
    plain_args = [
        "--dim", "576", "--layers", "4", "--heads", "24",
        "--pairs", "128", "--topk", "2", "--expert-hidden", "100", "--moe-every", "5",
        "--geo", "0.0", "--balance", "0.0", "--router-mode", "linear",
    ]

    def base(steps: int, save_every: int) -> list[str]:
        return [
            sys.executable, "-u", str(code_dir / "poincare_frontier.py"), "train",
            "--data-dir", str(data_dir), "--steps", str(steps),
            "--context", str(CTX), "--batch", str(BATCH), "--grad-accum", "1",
            "--future2", "0.3", "--future3", "0.15", "--future-mode", "rollout",
            "--lr", "3e-4", "--eval-every", "100", "--eval-batches", "4",
            "--save-every", str(save_every), "--seed", "1337",
        ]

    jobs = {
        "plain_long": {"steps": 1220, "save": 100, "args": plain_args, "timeout": 1500},
        "hybrid_both_long": {"steps": 1220, "save": 100,
                             "args": hybrid_common + ["--router-mode", "both"], "timeout": 3600},
        "hybrid_both_short": {"steps": 244, "save": 100,
                              "args": hybrid_common + ["--router-mode", "both"], "timeout": 1200},
        "hybrid_geo_short": {"steps": 244, "save": 100,
                             "args": hybrid_common + ["--router-mode", "geo"], "timeout": 1200},
        "hybrid_linear_short": {"steps": 244, "save": 100,
                                "args": hybrid_common + ["--router-mode", "linear"], "timeout": 1200},
    }
    for name, job in jobs.items():
        job["run_dir"] = WORK / name
        shutil.rmtree(job["run_dir"], ignore_errors=True)
        job["run_dir"].mkdir(parents=True, exist_ok=True)
        job["cmd"] = base(job["steps"], job["save"]) + ["--run-dir", str(job["run_dir"])] + job["args"]

    results: dict[str, dict] = {}

    def worker(gpu: int, names: list[str]) -> None:
        for name in names:
            job = jobs[name]
            print(f"[gpu{gpu}] starting {name}", flush=True)
            results[name] = {"train": run(job["cmd"], WORK / f"{name}.train.log",
                                          job["timeout"], gpu)}
            checkpoint = job["run_dir"] / "latest.pt"
            entry = results[name]
            entry["checkpoint_exists"] = checkpoint.exists()
            if not checkpoint.exists():
                print(f"[gpu{gpu}] no checkpoint for {name}", flush=True)
                continue
            probe = run(
                [sys.executable, "-u", str(code_dir / "probe_r31.py"),
                 "--data-dir", str(data_dir), "--run-dir", str(job["run_dir"]),
                 "--context", str(CTX), "--batch", "8", "--batches", "32",
                 "--book-windows", "4", "--out", str(WORK / f"{name}.probe.json")],
                WORK / f"{name}.probe.log", 600, gpu,
            )
            entry["probe_run"] = probe
            try:
                entry["probe"] = json.loads((WORK / f"{name}.probe.json").read_text())
            except Exception as exc:  # noqa: BLE001
                entry["probe_error"] = repr(exc)
            gen = run(
                [sys.executable, "-u", str(code_dir / "gen_r31.py"),
                 "--run-dir", str(job["run_dir"]), "--prompt-file", str(code_dir / "heldout_prompt.txt"),
                 "--max-new-tokens", "200", "--temperature", "0.8", "--topk-sample", "30",
                 "--out", str(WORK / f"{name}.gen.json")],
                WORK / f"{name}.gen.log", 600, gpu,
            )
            entry["gen_run"] = gen
            try:
                entry["gen"] = json.loads((WORK / f"{name}.gen.json").read_text())
            except Exception as exc:  # noqa: BLE001
                entry["gen_error"] = repr(exc)
            if name != "hybrid_both_long":
                shutil.rmtree(job["run_dir"], ignore_errors=True)
                print(f"[gpu{gpu}] removed checkpoint for {name}", flush=True)

    wave1 = [threading.Thread(target=worker, args=(0, ["plain_long", "hybrid_both_short", "hybrid_linear_short"])),
             threading.Thread(target=worker, args=(1, ["hybrid_both_long", "hybrid_geo_short"]))]
    for thread in wave1:
        thread.start()
    for thread in wave1:
        thread.join()

    summary = {
        "design": {
            "corpus": "NaNROUTER corpora-sentences-0001; 43 train / 5 held-out books",
            "context": CTX, "batch": BATCH,
            "main_steps": 1220, "main_positions": 1220 * BATCH * CTX,
            "router_steps": 244, "router_positions": 244 * BATCH * CTX,
            "seed": 1337,
            "plain": "21,388,032 total and active params",
            "hybrid151": "151,005,954 total; dense trunk + additive 128-pair top-2 MoE; ~21.1M active MAC/token estimate",
            "router_ablation": "same 151M hybrid; scores=linear, scores=-hyperbolic_distance, or both; strength learnable from 1.0; balance 0.01",
        },
        "parameter_counts": {
            "plain": sum(p.numel() for p in pf.MirrorLM(pf.Config(
                vocab_size=257, dim=576, layers=4, heads=24, pairs=128, topk_pairs=2,
                expert_hidden=100, moe_every=5, context=CTX, hyperbolic_strength=0.0,
                router_mode="linear")).parameters()),
            "hybrid151": sum(p.numel() for p in pf.MirrorLM(pf.Config(
                vocab_size=257, dim=544, layers=4, heads=16, pairs=128, topk_pairs=2,
                expert_hidden=235, moe_every=2, context=CTX, hyperbolic_strength=1.0,
                router_mode="both", moe_additive=True)).parameters()),
        },
        "results": results,
        "comparison": {},
    }
    def bpb(name):
        probe = results.get(name, {}).get("probe")
        return probe["bpb"] if probe else None
    summary["comparison"] = {
        "hybrid151_minus_plain_bpb": (round(bpb("hybrid_both_long") - bpb("plain_long"), 6)
                                      if bpb("hybrid_both_long") is not None and bpb("plain_long") is not None else None),
        "router_geo_minus_linear_bpb": (round(bpb("hybrid_geo_short") - bpb("hybrid_linear_short"), 6)
                                        if bpb("hybrid_geo_short") is not None and bpb("hybrid_linear_short") is not None else None),
        "router_both_minus_linear_bpb": (round(bpb("hybrid_both_short") - bpb("hybrid_linear_short"), 6)
                                         if bpb("hybrid_both_short") is not None and bpb("hybrid_linear_short") is not None else None),
    }
    (WORK / "r31_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nR31 SUMMARY\n" + json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
