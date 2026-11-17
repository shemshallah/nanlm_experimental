"""A/A noise audit and A/B experiment harness for Terminal-RSI v3/v4.

Examples
--------
A/A probe noise audit on an existing checkpoint::

    python -m terminal_rsi.run_rsi_ab aa \
        --checkpoint path/to/candidate.pt \
        --train-stream path/to/train_cpu_pilot.bin \
        --bpe-merges terminal_rsi/reference/nancm_v103_bpe.txt \
        --shards 4 --positions 256 --repeats 20

A/B training comparison::

    python -m terminal_rsi.run_rsi_ab ab \
        --base-run path/to/cpu_run \
        --arms off,v3,v4 --seeds 2026,2027,2028 \
        --updates 1000 --output ab_results

``--dry-run`` prints the exact commands without spawning them.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

import torch

from .legacy_v103_reference import (
    PARITY_CONTEXT, build_train_features, read_bpe_ranks,
)
from .rsi_v4 import ProbeShards, probe_noise_audit
from .verbatim_v103 import VerbatimV103RSI


def _mean_se(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return statistics.mean(values), statistics.stdev(values) / math.sqrt(len(values))


def _load_model(checkpoint: Path) -> VerbatimV103RSI:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model" in state:
        model_state = state["model"]
    else:
        model_state = state
    model = VerbatimV103RSI("cpu")
    model.load_state_dict(model_state, strict=True)
    model.eval()
    return model


def aa_audit(args) -> dict:
    raw = args.train_stream.read_bytes()
    ranks = read_bpe_ranks(args.bpe_merges)
    features, _ = build_train_features(raw, ranks)
    raw_tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    aux_data = torch.from_numpy(features[PARITY_CONTEXT:].copy()).to(dtype=torch.int32)
    model = _load_model(args.checkpoint)
    report = probe_noise_audit(
        model, raw_tensor, aux_data,
        history=args.context,
        n_shards=args.shards,
        positions_per_shard=args.positions,
        repeats=args.repeats,
        seed=args.seed,
        eval_chunk=args.eval_chunk,
    )
    report.update({
        "checkpoint": str(args.checkpoint),
        "train_stream": str(args.train_stream),
        "train_sha256": __import__("hashlib").sha256(raw).hexdigest(),
        "context": args.context,
    })
    return report


def _train_command(args, arm: str, seed: int, output: Path) -> list[str]:
    base = args.base_run
    bpe_merges = (args.bpe_merges if args.bpe_merges.is_absolute()
                  else args.repo / args.bpe_merges)
    command = [
        sys.executable, "-m", "terminal_rsi.train_verbatim",
        "--fresh-start", "--mode", "cpu", "--cpu-threads", str(args.cpu_threads),
        "--seed", str(seed), "--updates", str(args.updates), "--batch", str(args.batch),
        "--train-stream", str(base / "train_cpu_pilot.bin"),
        "--bpe-merges", str(bpe_merges.resolve()),
        "--dev-bin", str(base / "win_dev.bin"),
        "--dev-manifest", str(base / "win_dev.manifest.json"),
        "--output", str(output / "candidate.pt"),
        "--metrics-output", str(output / "metrics.json"),
        "--max-dev-positions", str(args.max_dev_positions),
        "--topk-probes", str(args.topk_probes),
        "--upgrade-context", str(args.upgrade_context),
        "--progress-every", str(args.progress_every),
        "--champion-dir", str(output / "champions"),
        "--rsi-version", arm,
    ]
    if arm == "v3":
        command += [
            "--rsi-cycle-updates", str(args.v3_cycle_updates),
            "--rsi-probe-positions", "16",
            "--rsi-min-bpb-gain", str(args.v3_min_gain),
        ]
    if arm == "v4":
        command += [
            "--rsi-v4-cycle-updates", str(args.v4_cycle_updates),
            "--rsi-v4-horizon", str(args.v4_horizon),
            "--rsi-v4-candidates", str(args.v4_candidates),
            "--rsi-v4-shards", str(args.v4_shards),
            "--rsi-v4-shard-positions", str(args.v4_shard_positions),
            "--rsi-v4-min-gain", str(args.v4_min_gain),
            "--rsi-v4-z", str(args.v4_z),
            "--rsi-v4-eval-chunk", str(args.v4_eval_chunk),
        ]
        holdout = args.rsi_holdout_stream or (base / "rsi_holdout.bin")
        if holdout.is_file():
            command += ["--rsi-holdout-stream", str(holdout)]
        if args.v4_meta:
            command += ["--rsi-v4-meta", str(args.v4_meta)]
    return command


def ab_experiment(args) -> dict:
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    invalid = [arm for arm in arms if arm not in ("off", "v3", "v4")]
    if invalid:
        raise SystemExit(f"Unknown arms: {invalid}")
    if not arms or not seeds:
        raise SystemExit("At least one arm and one seed are required")
    args.output.mkdir(parents=True, exist_ok=True)
    runs = []
    for arm in arms:
        for seed in seeds:
            output = args.output / f"{arm}_seed{seed}"
            output.mkdir(parents=True, exist_ok=True)
            command = _train_command(args, arm, seed, output)
            print("$", " ".join(command), flush=True)
            if args.dry_run:
                continue
            log_path = output / "train.log"
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.run(
                    command, cwd=args.repo, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True)
                log.write(process.stdout or "")
            metrics_path = output / "metrics.json"
            if process.returncode != 0 or not metrics_path.is_file():
                runs.append({
                    "arm": arm, "seed": seed, "status": "failed",
                    "returncode": process.returncode,
                    "log": str(log_path),
                })
                continue
            metrics = json.loads(metrics_path.read_text())
            runs.append({
                "arm": arm, "seed": seed, "status": "completed",
                "metrics": str(metrics_path), "log": str(log_path),
                "before_bpb": metrics["before"]["bpb"],
                "after_bpb": metrics["after"]["bpb"],
                "delta_bpb": metrics["after"]["bpb"] - metrics["before"]["bpb"],
                "updates": metrics["actual_byte_updates_this_round"],
                "updates_per_second": metrics["updates_per_second"],
                "training_seconds": metrics["training_duration_seconds"],
                "rsi_proposals": metrics.get("rsi_mutations_verified", 0),
                "rsi_accepted": metrics.get("rsi_mutations_accepted", 0),
            })

    summary = {}
    for arm in arms:
        completed = [run for run in runs
                     if run["arm"] == arm and run["status"] == "completed"]
        after = [run["after_bpb"] for run in completed]
        delta = [run["delta_bpb"] for run in completed]
        speed = [run["updates_per_second"] for run in completed]
        accepted = [run["rsi_accepted"] for run in completed]
        mean_after, se_after = _mean_se(after)
        mean_delta, se_delta = _mean_se(delta)
        mean_speed, se_speed = _mean_se(speed)
        summary[arm] = {
            "runs": len(completed),
            "mean_after_bpb": mean_after,
            "se_after_bpb": se_after,
            "mean_delta_bpb": mean_delta,
            "se_delta_bpb": se_delta,
            "mean_updates_per_second": mean_speed,
            "se_updates_per_second": se_speed,
            "mean_rsi_accepted": statistics.mean(accepted) if accepted else 0.0,
            "after_bpb": after,
            "delta_bpb": delta,
        }
    report = {"runs": runs, "summary": summary}
    (args.output / "ab_report.json").write_text(json.dumps(report, indent=2))
    lines = ["# RSI A/B report", "", f"Updates per run: {args.updates}", "",
             "| Arm | Runs | Final BPB | SE | Δ BPB | SE | Updates/s | RSI accepted |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in arms:
        item = summary[arm]
        lines.append(
            f"| {arm} | {item['runs']} | {item['mean_after_bpb']:.6f} | "
            f"{item['se_after_bpb']:.6f} | {item['mean_delta_bpb']:+.6f} | "
            f"{item['se_delta_bpb']:.6f} | {item['mean_updates_per_second']:.3f} | "
            f"{item['mean_rsi_accepted']:.2f} |")
    (args.output / "ab_report.md").write_text("\n".join(lines) + "\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    aa = sub.add_parser("aa", help="A/A probe-noise audit")
    aa.add_argument("--checkpoint", type=Path, required=True)
    aa.add_argument("--train-stream", type=Path, required=True)
    aa.add_argument("--bpe-merges", type=Path, required=True)
    aa.add_argument("--context", type=int, default=256)
    aa.add_argument("--shards", type=int, default=4)
    aa.add_argument("--positions", type=int, default=256)
    aa.add_argument("--repeats", type=int, default=20)
    aa.add_argument("--seed", type=int, default=2026)
    aa.add_argument("--eval-chunk", type=int, default=8)
    aa.add_argument("--output", type=Path)

    ab = sub.add_parser("ab", help="A/B training comparison")
    ab.add_argument("--base-run", type=Path, required=True,
                    help="Directory with train_cpu_pilot.bin, win_dev.bin and manifest")
    ab.add_argument("--repo", type=Path, default=Path.cwd())
    ab.add_argument("--bpe-merges", type=Path,
                    default=Path("terminal_rsi/reference/nancm_v103_bpe.txt"))
    ab.add_argument("--arms", default="off,v3,v4")
    ab.add_argument("--seeds", default="2026,2027,2028")
    ab.add_argument("--updates", type=int, default=1000)
    ab.add_argument("--batch", type=int, default=4)
    ab.add_argument("--cpu-threads", type=int, default=4, choices=(1, 2, 4, 8, 16))
    ab.add_argument("--max-dev-positions", type=int, default=1024)
    ab.add_argument("--topk-probes", type=int, default=32)
    ab.add_argument("--upgrade-context", type=int, default=256)
    ab.add_argument("--progress-every", type=int, default=10)
    ab.add_argument("--v3-cycle-updates", type=int, default=50)
    ab.add_argument("--v3-min-gain", type=float, default=0.002)
    ab.add_argument("--v4-cycle-updates", type=int, default=200)
    ab.add_argument("--v4-horizon", type=int, default=10)
    ab.add_argument("--v4-candidates", type=int, default=2)
    ab.add_argument("--v4-shards", type=int, default=4)
    ab.add_argument("--v4-shard-positions", type=int, default=256)
    ab.add_argument("--v4-min-gain", type=float, default=0.01)
    ab.add_argument("--v4-z", type=float, default=2.0)
    ab.add_argument("--v4-eval-chunk", type=int, default=8)
    ab.add_argument("--rsi-holdout-stream", type=Path)
    ab.add_argument("--v4-meta", type=Path)
    ab.add_argument("--output", type=Path, required=True)
    ab.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.mode == "aa":
        report = aa_audit(args)
        text = json.dumps(report, indent=2)
        print(text)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n")
        return 0
    report = ab_experiment(args)
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
