import hashlib
import os
import pathlib
import subprocess
import sys
import tarfile

import requests

REPO = "/kaggle/working/NaNROUTER"
BUILD = pathlib.Path("/kaggle/working/build")
BUILD.mkdir(parents=True, exist_ok=True)
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/shemshallah/NaNROUTER", REPO], check=True)
try:
    import torch  # noqa: F401
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "torch",
                    "--index-url", "https://download.pytorch.org/whl/cu121"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "tokenizers", "zstandard"], check=True)

import zstandard  # noqa: E402

RELEASES = [
    ("corpora-paragraphs-0010", "paragraphs-0010", "para"),
    ("corpora-short-stories-0004", "short_stories-0004", "short"),
]
REL = BUILD / "release"
REL.mkdir(parents=True, exist_ok=True)
for tag, prefix, name in RELEASES:
    base = f"https://github.com/shemshallah/NaNROUTER/releases/download/{tag}"
    for suffix in ("tar.zst", "SHA256SUMS"):
        fname = f"{prefix}.{suffix}"
        target = REL / fname
        if not target.exists():
            print("downloading", fname, flush=True)
            with requests.get(f"{base}/{fname}", stream=True, timeout=300) as r:
                r.raise_for_status()
                with target.open("wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
    expected = (REL / f"{prefix}.SHA256SUMS").read_text().split()[0]
    h = hashlib.sha256()
    with (REL / f"{prefix}.tar.zst").open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    assert h.hexdigest() == expected, f"sha mismatch {prefix}"

for tag, prefix, name in RELEASES:
    dest = BUILD / f"corpus_{name}"
    dest.mkdir(parents=True, exist_ok=True)
    dctx = zstandard.ZstdDecompressor()
    with (REL / f"{prefix}.tar.zst").open("rb") as fh:
        with dctx.stream_reader(fh) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tf:
                tf.extractall(dest)

subprocess.run([sys.executable, os.path.join(REPO, "nanlmv3", "prepare_corpus.py"),
                "--source", str(BUILD / "corpus_para"),
                "--source", str(BUILD / "corpus_short"),
                "--bytes", "100000000",
                "--output", str(BUILD / "corpus_new_100m.txt"),
                "--manifest", str(BUILD / "corpus_new_100m.json")], check=True)
subprocess.run([sys.executable, os.path.join(REPO, "nanlmv3", "sample_corpus.py"),
                "--source", str(BUILD / "corpus_new_100m.txt"),
                "--output", str(BUILD / "model_new_100m.txt"),
                "--bytes", "100000000", "--chunks", "20", "--val-chunks", "2",
                "--manifest", str(BUILD / "model_new_100m.json")], check=True)
subprocess.run([sys.executable, os.path.join(REPO, "nanlmv3", "sample_corpus.py"),
                "--source", str(BUILD / "corpus_new_100m.txt"),
                "--output", str(BUILD / "bpe_new_50m.txt"),
                "--bytes", "50000000", "--chunks", "50", "--val-chunks", "0"], check=True)
subprocess.run([sys.executable, os.path.join(REPO, "nanlmv3", "train_bpe.py"),
                "--corpus", str(BUILD / "bpe_new_50m.txt"),
                "--vocab-size", "32000",
                "--output", str(BUILD / "bpe32k_new.json"),
                "--meta", str(BUILD / "bpe32k_new_meta.json")], check=True)

cmd = [
    sys.executable, os.path.join(REPO, "nanlmv3", "unified_v2.py"),
    "--tokenizer", "bpe",
    "--bpe-model", str(BUILD / "bpe32k_new.json"),
    "--corpus", str(BUILD / "model_new_100m.txt"),
    "--ctx", "512",
    "--d-model", "256",
    "--layers", "8",
    "--heads", "8",
    "--swiglu", "1024",
    "--d-hidden", "512",
    "--pairs", "128",
    "--topk", "8",
    "--branches", "32",
    "--steps", "5000",
    "--max-seconds", "600",
    "--batch-size", "8",
    "--eval-every", "500",
    "--eval-batches", "128",
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
    "--out", str(BUILD / "v2_newcorpus10"),
]
print("running", " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
print((BUILD / "v2_newcorpus10" / "result.json").read_text(), flush=True)
