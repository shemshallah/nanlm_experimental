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
import zipfile

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
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
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
        if shutil.which("tar") and shutil.which("zstd"):
            subprocess.run(["tar", "--zstd", "-xf", str(archive), "-C", str(local)], check=True)
        else:
            # Molab Python environments often have no sudo, GNU tar or zstd
            # executable. Use the pip-installable zstandard module instead.
            import tarfile
            try:
                import zstandard
            except ImportError as exc:
                raise RuntimeError(
                    "Run: python -m pip install zstandard (no sudo required)"
                ) from exc
            with archive.open("rb") as stream:
                with zstandard.ZstdDecompressor().stream_reader(stream) as decoder:
                    with tarfile.open(fileobj=decoder, mode="r|") as tar:
                        root = local.resolve()
                        for member in tar:
                            target = (root / member.name).resolve()
                            if (not target.is_relative_to(root) or
                                    not (member.isfile() or member.isdir())):
                                raise ValueError("Unsafe path or member in verified NaNROUTER archive")
                            tar.extract(member, path=local, filter="data")
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
    p.add_argument("--topk-probes", type=int, default=32)
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
           "--topk-probes", str(args.topk_probes), "--upgrade-context", "256",
           "--rsi-probe-positions", "16",
           "--champion-dir", str(output / "champions")]
    print("Executing verified-BPB Terminal-RSI CPU run:", " ".join(cmd), flush=True)
    console = output / "terminal_rsi_cpu_1000_console.log"
    if console.exists():
        p.error(f"Training console already exists: {console}; choose a NEW --output-dir")
    # A live tee: preserve the complete subprocess output even when Molab's
    # notebook prevents selecting or copying the rendered cell output.
    with console.open("w", encoding="utf-8") as log:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, errors="replace", bufsize=1) as child:
            assert child.stdout is not None
            for line in child.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            status = child.wait()
            if status:
                raise subprocess.CalledProcessError(status, cmd)
    receipt = json.loads(metrics.read_text())
    if receipt["actual_byte_updates_this_round"] != args.updates or not receipt["old_v103_logits_identical_at_gate_zero"]:
        raise AssertionError("Fresh 1000-step training/legacy parity receipt failed")
    print("NEW ENGLISH BOOK DISJOINT CPU SCORES (BEFORE):", receipt["before"], flush=True)
    print("NEW ENGLISH BOOK DISJOINT CPU SCORES (AFTER):", receipt["after"], flush=True)
    print("CPU LOOP SECONDS:", receipt["training_duration_seconds"], flush=True)
    print("UPDATES PER SECOND:", receipt["updates_per_second"], flush=True)
    print("RSI REWARD MODE:", receipt["terminal_reward_mode"], flush=True)
    print("RSI VERIFIED MUTATIONS:", receipt["rsi_mutations_verified"], flush=True)
    print("RSI ACCEPTED MUTATIONS:", receipt["rsi_mutations_accepted"], flush=True)
    print("Unpromoted candidate checkpoint:", candidate, flush=True)
    print("Benchmark receipt:", metrics, flush=True)
    bundle = create_download_bundle(output, receipt)
    print("Downloadable complete checkpoint/metrics/log bundle:", bundle, flush=True)


def create_download_bundle(output: Path, receipt: dict) -> Path:
    """One no-nesting, SHA-256-indexed, recoverable Molab ZIP per CPU round."""
    output = Path(output)
    accepted = receipt.get("champion_guard") or {}
    checkpoint = Path(accepted.get("champion_checkpoint", ""))
    if not checkpoint.is_file() or not checkpoint.is_relative_to(output):
        raise ValueError("Cannot package a missing/unprotected RSI champion")
    names = (
        "terminal_rsi_cpu_1000_metrics.json",
        "terminal_rsi_cpu_1000_console.log",
        "cpu_pilot.manifest.json",
        "win_dev.manifest.json",
        "train_book_ids.json",
        "train_cpu_pilot.bin",
        "win_dev.bin",
        "champions/best.json",
    )
    sources = [output / name for name in names] + [checkpoint]
    for f in sources:
        if not f.is_file():
            raise ValueError(f"Missing recoverability input: {f}")
    entries = {}
    for f in sources:
        path_inside = f.relative_to(output).as_posix()
        with f.open("rb") as stream:
            entries[path_inside] = (hashlib.file_digest(stream, "sha256").hexdigest(),
                                    f.stat().st_size)
    sums = "".join(f"{sha}  {filename}\n" for filename, (sha, _) in sorted(entries.items()))
    notes = (
        "Terminal-RSI CPU exact-update recoverable round.\n"
        "Restore the entire archive; the winning model+all AdamW states is under champions/.\n"
        "Training used NaNROUTER English-only book-disjoint DEV; independent TEST sealed.\n"
        "Verify integrity with sha256sum -c SHA256SUMS.txt after extraction.\n"
    )
    bundle = output / "terminal_rsi_cpu_1000_bundle.zip"
    if bundle.exists():
        raise ValueError("Bundle already exists; choose a new output directory")
    pending = output / (bundle.name + ".partial")
    try:
        with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_STORED,
                             allowZip64=True) as archive:
            for f in sources:
                archive.write(f, f.relative_to(output).as_posix())
            archive.writestr("SHA256SUMS.txt", sums)
            archive.writestr("RECOVERY.txt", notes)
        pending.replace(bundle)
    finally:
        pending.unlink(missing_ok=True)
    return bundle


if __name__ == "__main__":
    main()
