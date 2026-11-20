"""Detached 6-hour Terminal-RSI bootstrap.

Builds the mixed 80 MB pilot, starts the Blackwell trainer, starts the rolling
Kaggle uploader, and waits for the trainer so ``DONE`` triggers a final backup.
Designed to be launched with nohup/start_new_session from a short notebook cell.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .prepare_cpu_pilot import build_cpu_pilot


def _log(message: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def _alive(pid_file: Path) -> bool:
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _split_books(corpus: Path, target: int = 1600) -> None:
    files = sorted(corpus.glob("*.txt"))
    numeric = [p for p in files if p.name[:1].isdigit()]
    if len(numeric) >= target:
        return
    text = "\n\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in files if not p.name[:1].isdigit())
    if not text:
        raise RuntimeError("six-hour corpus is empty")
    size = max(4096, len(text) // target)
    for index in range(target):
        chunk = text[index * size:(index + 1) * size] if index < target - 1 else text[(target - 1) * size:]
        if len(chunk) < 4096:
            continue
        path = corpus / f"{70000 + index}--sixh_{index:05d}.txt"
        if not path.exists():
            path.write_text(chunk, encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--init-weights", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--bpe", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=21600.0)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--microbatch", type=int, default=256)
    parser.add_argument("--train-bytes", type=int, default=80_000_000)
    args = parser.parse_args(argv)

    run = args.run_dir
    run.mkdir(parents=True, exist_ok=True)
    if (run / "pid.txt").exists() and _alive(run / "pid.txt"):
        _log(f"trainer already alive in {run}")
        return 0
    if (run / "DONE").exists():
        _log(f"run already complete in {run}")
        return 0

    pilot = args.pilot_dir
    if not (pilot / "train_cpu_pilot.bin").exists():
        _split_books(args.corpus_dir)
        _log("building 80 MB mixed pilot")
        report = build_cpu_pilot(
            corpus_dir=args.corpus_dir, output_dir=pilot,
            train_bytes=args.train_bytes, dev_books=64, dev_positions=1024,
            seed=103)
        _log("pilot built: " + json.dumps({
            "train_bytes": report["train_bytes"],
            "train_sha256": report["train_sha256"],
            "train_books": report["train_books"],
        }))
    staging = Path("/marimo/trsi_upload_6h")
    staging.mkdir(parents=True, exist_ok=True)
    upload_log = open(run / "upload_driver.log", "a")
    uploader = subprocess.Popen(
        [sys.executable, "-m", "terminal_rsi.kaggle_rolling",
         "--run-dir", str(run), "--staging", str(staging),
         "--dataset", args.dataset, "--token", args.token,
         "--title", "TRSI 100M 6h diverse checkpoints",
         "--description", "Rolling 6-hour Terminal-RSI weights + logs on the "
                          "34-source broad7h byte corpus (no WikiText)."],
        cwd=str(Path(__file__).resolve().parents[1]),
        stdout=upload_log, stderr=subprocess.STDOUT, start_new_session=True)
    _log(f"uploader {uploader.pid}")

    cmd = [
        sys.executable, "-m", "terminal_rsi.train_verbatim",
        "--mode", "blackwell", "--fresh-start",
        "--init-weights", str(args.init_weights),
        "--init-core-lr-scale", "0.1",
        "--train-stream", str(pilot / "train_cpu_pilot.bin"),
        "--dev-bin", str(pilot / "win_dev.bin"),
        "--dev-manifest", str(pilot / "win_dev.manifest.json"),
        "--bpe-merges", str(args.bpe),
        "--seconds", str(args.seconds),
        "--batch", str(args.batch), "--microbatch", str(args.microbatch),
        "--no-gpu-window-cache", "--upgrade-context", "256",
        "--upgrade-loss-weight", "1.0", "--boundary-loss-weight", "0.05",
        "--unit-loss-weight", "0.2", "--unit-byte-loss-weight", "0.2",
        "--unit-byte-distill-weight", "0.0", "--distill-weight", "0.0",
        "--distill-samples", "8", "--moe-loss-weight", "0.01",
        "--top5-byte-loss-weight", "0.1",
        "--word-boundary-loss-weight", "0.2",
        "--word-top5-loss-weight", "0.1", "--word-ngram-weight", "0.5",
        "--ablation", "full", "--rsi-version", "off",
        "--rsi-fast-adapters", "0",
        "--max-dev-positions", "512", "--topk-probes", "8",
        "--hierarchical-eval-books", "2",
        "--hierarchical-eval-corpus-dir", str(args.corpus_dir),
        "--hierarchical-eval-max-positions", "128",
        "--hierarchical-eval-batch", "128",
        "--unit-eval-books", "8", "--unit-eval-positions", "512",
        "--generation-steps", "8",
        "--output", str(run / "last.pt"),
        "--metrics-output", str(run / "metrics.json"),
        "--save-every", "50", "--progress-every", "1",
        "--cpu-threads", "8",
    ]
    (run / "RUN_SUMMARY.md").write_text(
        "# TRSI 100M 6h diverse run\n\n"
        "- Init: fresh step-150 weights, then 6 hours on the 34-source mixed "
        "broad7h byte corpus (80 MB train stream; no WikiText).\n"
        "- Word-token bridge + hashed 4-gram prior + word decoder active.\n"
        "- Weights and logs are versioned to Kaggle every 10 minutes; "
        "`last.pt` is atomically overwritten.\n")
    with open(run / "train.log", "w") as train_log:
        trainer = subprocess.Popen(
            cmd, cwd=str(Path(__file__).resolve().parents[1]),
            stdout=train_log, stderr=subprocess.STDOUT, start_new_session=True)
    (run / "pid.txt").write_text(str(trainer.pid))
    (run / "launch.json").write_text(json.dumps(
        {"cmd": cmd, "pid": trainer.pid, "start": time.time()}, indent=2))
    _log(f"trainer {trainer.pid}; waiting")
    trainer.wait()
    (run / "DONE").write_text("done\n")
    _log("trainer exited; final upload")
    try:
        uploader.wait(timeout=7200)
    except subprocess.TimeoutExpired:
        _log("uploader timeout")
    _log("bootstrap done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
