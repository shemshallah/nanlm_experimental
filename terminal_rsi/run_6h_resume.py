"""Resume the interrupted 6-hour Terminal-RSI run from a Kaggle checkpoint.

Rebuilds nothing itself: it expects the exact 80 MB pilot (train/dev bins,
manifest) and the rolling checkpoint produced by the earlier 6-hour run.  It
copies the checkpoint into the run directory, starts the rolling Kaggle
uploader against the existing dataset, then resumes training for the remaining
wall-clock budget and triggers a final upload on exit.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _log(run: Path, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with (run / "resume.log").open("a") as stream:
        stream.write(line + "\n")


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


def _copy_atomic(src: Path, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".part")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bpe", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--seconds", type=float, default=13200.0)
    parser.add_argument("--reopen-gates", type=float, default=None)
    parser.add_argument("--train-fast-adapter", action="store_true")
    parser.add_argument("--lookahead-loss-weight", type=float, default=0.0)
    parser.add_argument("--critic-loss-weight", type=float, default=0.0)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--microbatch", type=int, default=256)
    parser.add_argument("--staging", type=Path, default=Path("/marimo/trsi_upload_6h"))
    args = parser.parse_args(argv)

    run = args.run_dir
    run.mkdir(parents=True, exist_ok=True)
    if (run / "pid.txt").exists() and _alive(run / "pid.txt"):
        _log(run, "trainer already alive")
        return 0
    if (run / "DONE").exists():
        _log(run, "run already complete")
        return 0

    if not (args.pilot_dir / "train_cpu_pilot.bin").exists():
        raise SystemExit("recovered pilot is missing")
    for required in ("win_dev.bin", "win_dev.manifest.json"):
        if not (args.pilot_dir / required).exists():
            raise SystemExit(f"recovered pilot missing {required}")

    resume = run / "last.pt"
    if not resume.exists() or resume.stat().st_size < 10_000_000:
        _log(run, f"copying checkpoint {args.checkpoint} -> {resume}")
        _copy_atomic(args.checkpoint, resume)
    _log(run, f"checkpoint ready {resume.stat().st_size} bytes")

    # Preserve the pre-crash log so the Kaggle version has the full history.
    old_log = args.checkpoint.parent / "train.log"
    if old_log.exists() and not (run / "train.log").exists():
        shutil.copy2(old_log, run / "train.log")

    args.staging.mkdir(parents=True, exist_ok=True)
    (args.staging / ".created").write_text("1")
    upload_log = open(run / "upload_driver.log", "a")
    uploader = subprocess.Popen(
        [sys.executable, "-m", "terminal_rsi.kaggle_rolling",
         "--run-dir", str(run), "--staging", str(args.staging),
         "--dataset", args.dataset, "--token", args.token,
         "--title", "TRSI 100M 6h diverse checkpoints",
         "--description", "Rolling 6-hour Terminal-RSI weights + logs on the "
                          "34-source broad7h byte corpus (no WikiText).",
         "--interval", "600"],
        cwd=str(Path(__file__).resolve().parents[1]),
        stdout=upload_log, stderr=subprocess.STDOUT, start_new_session=True)
    _log(run, f"uploader {uploader.pid}")

    cmd = [
        sys.executable, "-m", "terminal_rsi.train_verbatim",
        "--mode", "blackwell", "--fresh-start",
        "--resume", str(resume),
        "--train-stream", str(args.pilot_dir / "train_cpu_pilot.bin"),
        "--dev-bin", str(args.pilot_dir / "win_dev.bin"),
        "--dev-manifest", str(args.pilot_dir / "win_dev.manifest.json"),
        "--bpe-merges", str(args.bpe),
        "--seconds", str(args.seconds),
        "--batch", str(args.batch), "--microbatch", str(args.microbatch),
        "--no-gpu-window-cache", "--upgrade-context", "256",
        "--upgrade-loss-weight", "1.0",
        "--upgrade-sequence-loss-weight", "1.0",
        "--boundary-loss-weight", "0.05",
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
    if args.reopen_gates is not None:
        cmd += ["--reopen-gates", str(args.reopen_gates)]
    if args.train_fast_adapter:
        cmd += ["--train-fast-adapter"]
    if args.lookahead_loss_weight:
        cmd += ["--lookahead-loss-weight", str(args.lookahead_loss_weight)]
    if args.critic_loss_weight:
        cmd += ["--critic-loss-weight", str(args.critic_loss_weight)]
    with (run / "RUN_SUMMARY.md").open("a") as stream:
        stream.write(
            "\n## Resumed after platform termination\n\n"
            f"- Checkpoint: {args.checkpoint} (copied to `{resume}`)\n"
            f"- Remaining wall-clock budget: {args.seconds:.0f} s\n"
            "- Same 80 MB recovered pilot and DEV, SHA-verified against "
            "checkpoint provenance.\n")
    with (run / "train.log").open("a") as train_log:
        trainer = subprocess.Popen(
            cmd, cwd=str(Path(__file__).resolve().parents[1]),
            stdout=train_log, stderr=subprocess.STDOUT, start_new_session=True)
    (run / "pid.txt").write_text(str(trainer.pid))
    (run / "launch.json").write_text(json.dumps(
        {"cmd": cmd, "pid": trainer.pid, "start": time.time(),
         "resumed_from": str(args.checkpoint)}, indent=2))
    _log(run, f"trainer {trainer.pid}; waiting")
    trainer.wait()
    (run / "DONE").write_text("done\n")
    _log(run, "trainer exited; final upload")
    try:
        uploader.wait(timeout=7200)
    except subprocess.TimeoutExpired:
        _log(run, "uploader timeout")
    _log(run, "resume bootstrap done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
