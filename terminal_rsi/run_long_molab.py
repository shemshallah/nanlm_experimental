"""Supervised long CPU run for Terminal-RSI v4 on Molab.

The supervisor:

1. builds a diverse difficulty curriculum if it is not already present,
2. repeatedly runs ``train_verbatim --seconds <remaining>`` with a rolling
   checkpoint so a crash or Molab restart loses at most ``--save-every`` updates,
3. resumes from the rolling checkpoint after a failure,
4. writes a final candidate, a long-run summary and a SHA-256-indexed bundle.

Example (6 hours, 8 CPU threads)::

    python -m terminal_rsi.run_long_molab \
      --corpus-dir /path/to/nanrouter_batch0004 \
      --output-dir /marimo/terminal_rsi_molab_runs/long_6h \
      --seconds 21600 --cpu-threads 8 \
      --train-bytes 64000000 \
      --rsi-v4-cycle-updates 1000 --rsi-v4-horizon 10 \
      --rsi-v4-candidates 2 --rsi-v4-shards 4 \
      --rsi-v4-shard-positions 128 --rsi-v4-eval-chunk 8 \
      --rsi-v4-min-gain 0.01 --rsi-v4-z 2.0 \
      --rsi-v4-critic-steps 16 --rsi-v4-policy-steps 8 \
      --save-every 500
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from .prepare_curriculum import build_curriculum


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _log(path: Path, message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _train_command(args, output: Path, rolling: Path, metrics: Path,
                   remaining: float) -> list[str]:
    command = [
        sys.executable, "-m", "terminal_rsi.train_verbatim",
        "--mode", args.mode, "--cpu-threads", str(args.cpu_threads),
        "--batch", str(args.batch),
        "--train-stream", str(output / "train_curriculum.bin"),
        "--bpe-merges", str(args.bpe_merges),
        "--dev-bin", str(output / "win_dev.bin"),
        "--dev-manifest", str(output / "win_dev.manifest.json"),
        "--output", str(rolling), "--metrics-output", str(metrics),
        "--max-dev-positions", str(args.max_dev_positions),
        "--topk-probes", str(args.topk_probes),
        "--upgrade-context", "256", "--rsi-probe-positions", "16",
        "--progress-every", str(args.progress_every),
        "--save-every", str(args.save_every),
        "--seconds", f"{max(1.0, remaining):.3f}",
        "--rsi-version", "v4",
        "--rsi-holdout-stream", str(output / "rsi_holdout.bin"),
        "--rsi-v4-cycle-updates", str(args.v4_cycle),
        "--rsi-v4-horizon", str(args.v4_horizon),
        "--rsi-v4-candidates", str(args.v4_candidates),
        "--rsi-v4-shards", str(args.v4_shards),
        "--rsi-v4-shard-positions", str(args.v4_shard_positions),
        "--rsi-v4-eval-chunk", str(args.v4_eval_chunk),
        "--rsi-v4-min-gain", str(args.v4_min_gain),
        "--rsi-v4-z", str(args.v4_z),
        "--rsi-v4-critic-steps", str(args.v4_critic_steps),
        "--rsi-v4-policy-steps", str(args.v4_policy_steps),
        "--rsi-v4-entropy", str(args.v4_entropy),
        "--champion-dir", str(output / "champions"),
    ]
    command.append("--fresh-start")
    if rolling.is_file():
        command += ["--resume", str(rolling)]
    if args.v4_meta and args.v4_meta.is_file():
        command += ["--rsi-v4-meta", str(args.v4_meta)]
    return command


def _bundle(output: Path, summary_path: Path) -> Path:
    names = [
        "terminal_rsi_long_candidate.pt",
        "terminal_rsi_long_metrics.json",
        "long_run_summary.json",
        "RUN_SUMMARY.md",
        "curriculum_manifest.json",
        "train_curriculum.bin",
        "train_curriculum_book_ids.json",
        "rsi_holdout.bin",
        "rsi_holdout_book_ids.json",
        "win_dev.bin",
        "win_dev.manifest.json",
        "supervisor.log",
        "champions/best.json",
    ]
    sources = [output / name for name in names]
    sources += sorted((output / "segments").glob("segment_*.log"))
    sources += sorted((output / "segments").glob("segment_*_metrics.json"))
    sources = [path for path in sources if path.is_file()]
    checksums = {}
    for path in sources:
        checksums[path.relative_to(output).as_posix()] = (
            _sha256(path), path.stat().st_size)
    notes = (
        "Terminal-RSI long CPU run with difficulty curriculum and v4 RSI.\n"
        "Restore this whole archive before resuming. The rolling/final candidate\n"
        "contains model, all AdamW groups, sampler/RNG and RSI-v4 state.\n"
        "Verify with sha256sum -c SHA256SUMS.txt.\n"
    )
    sums = "".join(f"{sha}  {name}\n"
                   for name, (sha, _) in sorted(checksums.items()))
    bundle = output / "terminal_rsi_long_bundle.zip"
    pending = output / (bundle.name + ".partial")
    try:
        with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_STORED,
                             allowZip64=True) as archive:
            for path in sources:
                archive.write(path, path.relative_to(output).as_posix())
            archive.writestr("SHA256SUMS.txt", sums)
            archive.writestr("RECOVERY.txt", notes)
        pending.replace(bundle)
    finally:
        pending.unlink(missing_ok=True)
    return bundle


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=Path.cwd())
    p.add_argument("--bpe-merges", type=Path,
                   default=Path("terminal_rsi/reference/nancm_v103_bpe.txt"))
    p.add_argument("--seconds", type=float, default=21_600.0)
    p.add_argument("--mode", choices=("cpu", "t4", "blackwell"), default="cpu",
                   help="Execution profile for the same v4 graph; Blackwell uses BF16 upgrade autocast")
    p.add_argument("--cpu-threads", type=int, default=8, choices=(4, 8, 16))
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--train-bytes", type=int, default=64_000_000)
    p.add_argument("--dev-books", type=int, default=64)
    p.add_argument("--rsi-holdout-books", type=int, default=4)
    p.add_argument("--chunk-bytes", type=int, default=65_536)
    p.add_argument("--replay-fraction", type=float, default=0.25)
    p.add_argument("--max-books-per-author", type=int, default=6)
    p.add_argument("--max-dev-positions", type=int, default=1024)
    p.add_argument("--topk-probes", type=int, default=32)
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--min-segment-seconds", type=float, default=300.0)
    p.add_argument("--v4-cycle", type=int, default=1000)
    p.add_argument("--v4-horizon", type=int, default=10)
    p.add_argument("--v4-candidates", type=int, default=2)
    p.add_argument("--v4-shards", type=int, default=4)
    p.add_argument("--v4-shard-positions", type=int, default=128)
    p.add_argument("--v4-eval-chunk", type=int, default=8)
    p.add_argument("--v4-min-gain", type=float, default=0.01)
    p.add_argument("--v4-z", type=float, default=2.0)
    p.add_argument("--v4-critic-steps", type=int, default=16)
    p.add_argument("--v4-policy-steps", type=int, default=8)
    p.add_argument("--v4-entropy", type=float, default=0.01)
    p.add_argument("--v4-meta", type=Path)
    args = p.parse_args(argv)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    supervisor_log = output / "supervisor.log"
    _log(supervisor_log, "supervisor start")
    manifest_path = output / "curriculum_manifest.json"
    if not manifest_path.is_file():
        _log(supervisor_log, "building difficulty curriculum")
        report = build_curriculum(
            corpus_dir=args.corpus_dir,
            output_dir=output,
            train_bytes=args.train_bytes,
            dev_books=args.dev_books,
            dev_positions=1024,
            rsi_holdout_books=args.rsi_holdout_books,
            chunk_bytes=args.chunk_bytes,
            replay_fraction=args.replay_fraction,
            seed=103,
            max_books_per_author=args.max_books_per_author,
        )
        _log(supervisor_log, "curriculum built: " + json.dumps({
            "train_bytes": report["train_bytes"],
            "train_books": report["train_books"],
            "holdout_books": report["holdout_books"],
            "stages": report["stages"],
        }, sort_keys=True))
    else:
        report = json.loads(manifest_path.read_text())
        _log(supervisor_log, "reusing curriculum manifest")

    rolling = output / "terminal_rsi_long_rolling.pt"
    metrics = output / "terminal_rsi_long_metrics.json"
    segment_dir = output / "segments"
    segment_dir.mkdir(exist_ok=True)
    deadline = time.monotonic() + max(1.0, args.seconds)
    reports: list[dict] = []
    segment = 0
    start_wall = time.monotonic()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _log(supervisor_log, "time budget exhausted")
            break
        segment += 1
        command = _train_command(args, output, rolling, metrics, remaining)
        log_path = segment_dir / f"segment_{segment:03d}.log"
        _log(supervisor_log,
             f"segment {segment} start remaining={remaining:.1f}s "
             f"resume={rolling.is_file()}")
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write("COMMAND " + " ".join(command) + "\n")
            handle.flush()
            process = subprocess.run(
                command, cwd=args.repo, stdout=handle,
                stderr=subprocess.STDOUT, text=True)
        if metrics.is_file():
            shutil.copy2(metrics, segment_dir / f"segment_{segment:03d}_metrics.json")
            try:
                reports.append(json.loads(metrics.read_text()))
            except (OSError, ValueError):
                pass
        _log(supervisor_log,
             f"segment {segment} exit={process.returncode} "
             f"elapsed={time.monotonic() - start_wall:.1f}s")
        if process.returncode == 0:
            if deadline - time.monotonic() <= 1.0:
                break
            # A short segment may have run out of update work; keep going.
            continue
        if deadline - time.monotonic() < args.min_segment_seconds:
            _log(supervisor_log, "stopping after failure near deadline")
            break
        _log(supervisor_log, "segment failed; resuming from rolling checkpoint in 10s")
        time.sleep(10)

    candidate = output / "terminal_rsi_long_candidate.pt"
    if rolling.is_file():
        shutil.copy2(rolling, candidate)
    first = reports[0] if reports else {}
    last = reports[-1] if reports else {}
    final_metrics = last
    total_updates = sum(int(item.get("actual_byte_updates_this_round", 0))
                        for item in reports)
    total_training_seconds = sum(float(item.get("training_duration_seconds", 0.0))
                                 for item in reports)
    summary = {
        "schema": "terminal-rsi-long-run-v1",
        "mode": args.mode,
        "curriculum": report,
        "segments": len(reports),
        "wall_seconds": time.monotonic() - start_wall,
        "total_updates": total_updates,
        "total_training_seconds": total_training_seconds,
        "before": first.get("before"),
        "after": final_metrics.get("after"),
        "final_checkpoint": str(candidate) if candidate.is_file() else None,
        "final_checkpoint_sha256": _sha256(candidate) if candidate.is_file() else None,
        "rsi_v4": final_metrics.get("rsi_v4") if final_metrics else None,
        "segment_metrics": [
            {
                "updates": item.get("actual_byte_updates_this_round"),
                "training_seconds": item.get("training_duration_seconds"),
                "before_bpb": (item.get("before") or {}).get("bpb"),
                "after_bpb": (item.get("after") or {}).get("bpb"),
            }
            for item in reports
        ],
    }
    (output / "long_run_summary.json").write_text(json.dumps(summary, indent=2))
    before_bpb = (summary["before"] or {}).get("bpb")
    after_bpb = (summary["after"] or {}).get("bpb")
    lines = [
        "# Terminal-RSI long CPU run",
        "",
        f"- Segments: {summary['segments']}",
        f"- Total kept updates: {summary['total_updates']}",
        f"- Training-loop seconds: {summary['total_training_seconds']:.1f}",
        f"- Final DEV BPB: {after_bpb}",
        f"- Initial DEV BPB: {before_bpb}",
        f"- Final checkpoint: `{summary['final_checkpoint']}`",
        f"- Final checkpoint SHA-256: `{summary['final_checkpoint_sha256']}`",
        "",
        "## Curriculum",
        "",
        f"- TRAIN bytes: {report['train_bytes']}",
        f"- TRAIN books: {report['train_books']}",
        f"- Used chunks: {report['used_chunks']}",
        f"- RSI holdout books: {report['holdout_books']} ({report['holdout_bytes']} bytes)",
        f"- DEV books: {report['dev_books']}",
        f"- Stages: `{json.dumps(report['stages'], sort_keys=True)}`",
        f"- Genre counts: `{json.dumps(report['genre_counts'], sort_keys=True)}`",
        f"- Dedup: `{json.dumps(report['dedup'], sort_keys=True)}`",
        "",
        "## v4 ledger",
        "",
        f"`{json.dumps((summary['rsi_v4'] or {}), sort_keys=True)}`",
        "",
    ]
    (output / "RUN_SUMMARY.md").write_text("\n".join(lines) + "\n")
    bundle = _bundle(output, output / "long_run_summary.json")
    _log(supervisor_log, f"done bundle={bundle} size={bundle.stat().st_size}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
