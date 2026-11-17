"""Run the same real-book Terminal-RSI CPU experiment on Molab or local CPU.

No authentication, private weights or historical DEV is used. The original v103
model, features, RSI feedback and upgraded architecture are imported from this
repository. Outputs are disposable candidates, never promoted automatically.

Example:
    python -m terminal_rsi.run_molab_cpu_1000 --cpu-threads 4 --updates 1000
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

from terminal_rsi.prepare_cpu_pilot import build_cpu_pilot

RELEASE = "https://github.com/shemshallah/NaNROUTER/releases/download/corpora-short-stories-0004"
ARCHIVE = "short_stories-0004.tar.zst"
CHECKSUMS = "short_stories-0004.SHA256SUMS"


def download_if_absent(url: str, destination: Path):
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Avoid leaving a apparently finished archive after an interrupted transfer.
    temporary = destination.with_name(destination.name + ".partial")
    try:
        with urllib.request.urlopen(url, timeout=120) as source, temporary.open("wb") as sink:
            shutil.copyfileobj(source, sink)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def verify_release(archive: Path, sums: Path):
    rows = [s.split() for s in sums.read_text().splitlines() if s.strip()]
    target = [parts[0].lower() for parts in rows if len(parts) >= 2
              and parts[-1].lstrip("*") == archive.name]
    if len(target) != 1 or len(target[0]) != 64:
        raise ValueError("Unable to locate exactly one SHA-256 archive digest in NaNROUTER SUMS")
    digest = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
    if digest != target[0]:
        raise ValueError("NaNROUTER archive SHA-256 mismatch; refusing to extract")
    return digest


def source_dir(artifacts: Path, supplied: Path | None):
    if supplied is not None:
        if not supplied.is_dir():
            raise ValueError("Provided --corpus-dir is not an existing directory")
        return supplied
    local = artifacts / "nanrouter_batch0004"
    local.mkdir(parents=True, exist_ok=True)
    archive, sums = local / ARCHIVE, local / CHECKSUMS
    download_if_absent(f"{RELEASE}/{ARCHIVE}", archive)
    download_if_absent(f"{RELEASE}/{CHECKSUMS}", sums)
    print("Verified NaNROUTER archive SHA-256:", verify_release(archive, sums), flush=True)
    marker = local / "extracted.complete"
    if not marker.exists():
        if shutil.which("tar") is None or shutil.which("zstd") is None:
            raise RuntimeError("Requires GNU tar and zstd (e.g. apt-get install zstd)")
        subprocess.run(["tar", "--zstd", "-xf", str(archive), "-C", str(local)], check=True)
        marker.write_text("Archive verified against official SHA-256 list.\n")
    return local


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpu-threads", type=int, choices=(1, 2, 4), default=4)
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--output-dir", type=Path, default=Path("terminal_rsi_cpu_1000_out"))
    p.add_argument("--corpus-dir", type=Path, help="Previously extracted official NaNROUTER batch 0004")
    p.add_argument("--train-bytes", type=int, default=2_000_000)
    p.add_argument("--dev-books", type=int, default=64)
    p.add_argument("--max-dev-positions", type=int, default=1024)
    p.add_argument("--topk-probes", type=int, default=8)
    args = p.parse_args(argv)
    if args.updates < 1 or args.batch < 1 or args.max_dev_positions < 1 or args.topk_probes < 1:
        p.error("Positive updates, batch, DEV positions and topk probes are required")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    corpus = source_dir(output, args.corpus_dir)
    report = build_cpu_pilot(corpus_dir=corpus, output_dir=output,
                             train_bytes=args.train_bytes,
                             dev_books=args.dev_books, dev_positions=1024, seed=103)
    print("Fresh English-only TRAIN/DEV:", json.dumps(report, indent=2), flush=True)
    candidate = output / "terminal_rsi_cpu_1000_candidate.pt"
    metrics = output / "terminal_rsi_cpu_1000_metrics.json"
    if candidate.exists() or metrics.exists():
        p.error(f"Candidate or metrics already exist under {output}; choose a NEW --output-dir")
    cmd = [sys.executable, "-m", "terminal_rsi.train_verbatim",
           "--fresh-start", "--mode", "cpu", "--cpu-threads", str(args.cpu_threads),
           "--updates", str(args.updates), "--batch", str(args.batch),
           "--train-stream", str(output / "train_cpu_pilot.bin"),
           "--bpe-merges", "terminal_rsi/reference/nancm_v103_bpe.txt",
           "--dev-bin", str(output / "win_dev.bin"),
           "--dev-manifest", str(output / "win_dev.manifest.json"),
           "--output", str(candidate), "--metrics-output", str(metrics),
           "--max-dev-positions", str(args.max_dev_positions),
           "--topk-probes", str(args.topk_probes), "--upgrade-context", "256"]
    print("Executing exact-update v103 Terminal-RSI CPU run:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    receipt = json.loads(metrics.read_text())
    if receipt["actual_byte_updates_this_round"] != args.updates or not receipt["old_v103_logits_identical_at_gate_zero"]:
        raise AssertionError("Fresh 1000-step training/legacy parity receipt failed")
    print("NEW ENGLISH BOOK DISJOINT CPU SCORES (BEFORE):", receipt["before"], flush=True)
    print("NEW ENGLISH BOOK DISJOINT CPU SCORES (AFTER):", receipt["after"], flush=True)
    print("CPU LOOP SECONDS:", receipt["training_duration_seconds"], flush=True)
    print("UPDATES PER SECOND:", receipt["updates_per_second"], flush=True)
    print("RSI ACTOR FEEDBACK EVENTS:", len(receipt["terminal_events"]), flush=True)
    print("Unpromoted candidate checkpoint:", candidate, flush=True)
    print("Benchmark receipt:", metrics, flush=True)


if __name__ == "__main__":
    main()
