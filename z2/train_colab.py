#!/usr/bin/env python3
"""Colab 30-minute v2 run on the mixed NaNROUTER diverse + Gutenberg corpus.

Run from a Colab GPU notebook after cloning:
  NaNROUTER            -> /content/NaNROUTER
  nanlm_experimental   -> /content/nanlm_experimental

Example:
  python /content/nanlm_experimental/z2/train_colab.py \
    --minutes 30 --out /content/drive/MyDrive/nanlmv3_z2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tarfile

import requests

REPO = pathlib.Path(os.environ.get("NANROUTER_ROOT", "/content/NaNROUTER"))
EXP = pathlib.Path(os.environ.get("NANLM_EXPERIMENTAL_ROOT", "/content/nanlm_experimental"))
BUILD = pathlib.Path(os.environ.get("Z2_BUILD", "/content/z2_build"))


def download(url: str, dest: pathlib.Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("downloading", url, flush=True)
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_zst(archive: pathlib.Path, dest: pathlib.Path) -> None:
    import zstandard

    dest.mkdir(parents=True, exist_ok=True)
    dctx = zstandard.ZstdDecompressor()
    with archive.open("rb") as fh:
        with dctx.stream_reader(fh) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tf:
                tf.extractall(dest)


def extract_tar_gz(archive: pathlib.Path, dest: pathlib.Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, mode="r:gz") as tf:
        tf.extractall(dest)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", type=float, default=30.0)
    p.add_argument("--bytes", type=int, default=200_000_000)
    p.add_argument("--out", default="/content/z2")
    p.add_argument("--checkpoint-url", default="",
                   help="Optional z1/nanlmv3 .pt checkpoint URL for continuation")
    p.add_argument("--checkpoint-path", default="",
                   help="Optional local .pt checkpoint (e.g. Drive best.pt)")
    p.add_argument("--resume-best", action="store_true", default=True,
                   help="Automatically use OUT/best.pt if present")
    p.add_argument("--no-resume-best", dest="resume_best", action="store_false")
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-batches", type=int, default=64)
    args = p.parse_args()

    if not REPO.exists():
        raise SystemExit(f"missing {REPO}; clone NaNROUTER first")
    if not (REPO / "nanlmv3" / "unified_v2.py").exists():
        raise SystemExit(f"missing {REPO}/nanlmv3/unified_v2.py")

    build = BUILD
    build.mkdir(parents=True, exist_ok=True)
    out = pathlib.Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    # Diverse corpus: newest paragraph/short-story/sentence batches.
    releases = [
        ("corpora-paragraphs-0010", "paragraphs-0010"),
        ("corpora-short-stories-0004", "short_stories-0004"),
        ("corpora-sentences-0010", "sentences-0010"),
    ]
    sources: list[str] = []
    rel = build / "release"
    rel.mkdir(parents=True, exist_ok=True)
    for tag, prefix in releases:
        base = f"https://github.com/shemshallah/NaNROUTER/releases/download/{tag}"
        tar = rel / f"{prefix}.tar.zst"
        sums = rel / f"{prefix}.SHA256SUMS"
        download(f"{base}/{prefix}.tar.zst", tar)
        download(f"{base}/{prefix}.SHA256SUMS", sums)
        expected = sums.read_text().split()[0]
        actual = sha256(tar)
        assert actual == expected, f"sha mismatch {prefix}: {actual}"
        dest = build / f"corpus_{prefix}"
        extract_zst(tar, dest)
        sources.append(str(dest))

    # Gutenberg 21-book reference.
    gz = rel / "hw5-gutenberg-c256-g1.tar.gz"
    download(
        "https://github.com/shemshallah/NaNROUTER/releases/download/hw5-gutenberg-c256-g1/hw5-gutenberg-c256-g1.tar.gz",
        gz,
    )
    gdest = build / "corpus_gutenberg"
    extract_tar_gz(gz, gdest)
    sources.append(str(gdest))

    corpus = build / "mixed_200m.txt"
    cmd = [
        sys.executable, str(REPO / "nanlmv3" / "prepare_corpus.py"),
        *sum((["--source", s] for s in sources), []),
        "--bytes", str(args.bytes),
        "--output", str(corpus),
        "--manifest", str(build / "mixed_200m.json"),
    ]
    subprocess.run(cmd, check=True)

    model_corpus = build / "model_mixed_200m.txt"
    subprocess.run(
        [
            sys.executable, str(REPO / "nanlmv3" / "sample_corpus.py"),
            "--source", str(corpus),
            "--output", str(model_corpus),
            "--bytes", str(args.bytes),
            "--chunks", "20", "--val-chunks", "2",
            "--manifest", str(build / "model_mixed_200m.json"),
        ],
        check=True,
    )

    # Reuse z1 byte-level BPE if present; otherwise train one on a 50MB slice.
    bpe = EXP / "z1" / "build" / "bpe32k_new.json"
    if not bpe.exists():
        bpe_sample = build / "bpe_50m.txt"
        subprocess.run(
            [
                sys.executable, str(REPO / "nanlmv3" / "sample_corpus.py"),
                "--source", str(corpus),
                "--output", str(bpe_sample),
                "--bytes", "50000000",
                "--chunks", "50", "--val-chunks", "0",
            ],
            check=True,
        )
        bpe = build / "bpe32k.json"
        subprocess.run(
            [
                sys.executable, str(REPO / "nanlmv3" / "train_bpe.py"),
                "--corpus", str(bpe_sample),
                "--vocab-size", "32000",
                "--output", str(bpe),
                "--meta", str(build / "bpe32k_meta.json"),
            ],
            check=True,
        )

    init_from = ""
    if args.checkpoint_path:
        init_from = args.checkpoint_path
    elif args.resume_best:
        candidate = out / "best.pt"
        if candidate.exists():
            init_from = str(candidate)
    if not init_from and args.checkpoint_url:
        init_from = str(build / "init.pt")
        download(args.checkpoint_url, pathlib.Path(init_from))
    if init_from:
        if not pathlib.Path(init_from).exists():
            raise SystemExit(f"checkpoint not found: {init_from}")
        print(f"[resume] {init_from}", flush=True)

    cmd = [
        sys.executable, str(REPO / "nanlmv3" / "unified_v2.py"),
        "--tokenizer", "bpe",
        "--bpe-model", str(bpe),
        "--corpus", str(model_corpus),
        "--ctx", "512",
        "--d-model", "256",
        "--layers", "8",
        "--heads", "8",
        "--swiglu", "1024",
        "--d-hidden", "512",
        "--pairs", "128",
        "--topk", "8",
        "--branches", "32",
        "--steps", "20000",
        "--max-seconds", str(args.minutes * 60),
        "--batch-size", "8",
        "--eval-every", str(args.eval_every),
        "--eval-batches", str(args.eval_batches),
        "--train-fraction", "0.95",
        "--prev-weight", "0.1",
        "--memory-size", "1048576",
        "--memory-tokens", "10000000",
        "--recluster-every", "1000",
        "--dropout", "0.1",
        "--weight-decay", "0.05",
        "--lr", "1e-4",
        "--lr-horizon", "time",
        "--min-lr-frac", "0.0",
        "--ema-decay", "0.999",
        "--ema-start-frac", "0.3",
        "--amp",
        "--amp-init-scale", "1024",
        "--device", "cuda",
        "--threads", "4",
        "--out", str(out),
    ]
    if init_from:
        cmd += ["--init-from", init_from, "--eval-before"]
    print("running", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    result = json.loads((out / "result.json").read_text())
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
