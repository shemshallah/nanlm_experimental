#!/usr/bin/env python3
"""Kaggle TPU entry point for the causal 84M Z4 run.

The notebook/script is run by Kaggle with the TpuV5E8 accelerator.  It
installs the pinned torch/torch_xla stack only if the TPU image lacks it,
clones the current repo, patches the corpus manifest to the mounted dataset,
and runs either a short smoke or the long causal training.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

MODE = (Path(__file__).with_name("mode.txt").read_text().strip()
        if Path(__file__).with_name("mode.txt").is_file() else "smoke")
REPO = Path("/kaggle/working/nanlm_experimental")
DATA = Path("/kaggle/input/z4-broad7h-bins")
OUT = Path("/kaggle/working/runs/tpu_causal_84m")


def sh(cmd, *, timeout=None, check=True, env=None):
    print("$", " ".join(str(x) for x in cmd), flush=True)
    r = subprocess.run(
        [str(x) for x in cmd], timeout=timeout, check=False, env=env
    )
    if check and r.returncode != 0:
        raise SystemExit(f"command failed rc={r.returncode}")
    return r


print("mode", MODE, flush=True)
print("image", os.environ.get("COLAB_IMAGE_TYPE"),
      "vfio", os.path.exists("/dev/vfio"),
      "PJRT_DEVICE", os.environ.get("PJRT_DEVICE"), flush=True)
print("input", DATA, sorted(p.name for p in DATA.iterdir())[:20]
      if DATA.is_dir() else "MISSING", flush=True)

# The TPU VM must expose /dev/vfio.  If it does, its torch_xla stack is the
# right one.  If not, this is not a TPU session; fail early instead of
# burning the quota on a CPU image.
if not os.path.exists("/dev/vfio"):
    print("ERROR: /dev/vfio missing; this is not a TPU VM", flush=True)
    raise SystemExit(3)

os.environ.setdefault("PJRT_DEVICE", "TPU")

try:
    import torch_xla  # noqa: F401
    print("torch_xla present", flush=True)
except Exception as exc:
    print("torch_xla missing, installing pinned TPU stack:", exc, flush=True)
    sh([sys.executable, "-m", "pip", "install", "-q", "torch==2.9.0",
        "--index-url", "https://download.pytorch.org/whl/cpu"], timeout=1800)
    sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
        "torch_xla==2.9.0",
        "-f", "https://storage.googleapis.com/libtpu-releases/index.html"],
       timeout=1800)
    sh([sys.executable, "-m", "pip", "install", "-q", "libtpu==0.0.21.1",
        "-f", "https://storage.googleapis.com/libtpu-releases/index.html"],
       timeout=1800)

sh([sys.executable, "-m", "pip", "install", "-q", "tokenizers", "numpy",
    "pyarrow"], timeout=1800)

if not (REPO / "z3" / "train_v3.py").is_file():
    sh(["git", "clone", "--depth", "1",
        "https://github.com/shemshallah/nanlm_experimental.git", str(REPO)],
       timeout=600)
else:
    sh(["git", "-C", str(REPO), "pull", "--ff-only"], timeout=300)
print("repo", subprocess.run(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"],
    capture_output=True, text=True).stdout.strip(), flush=True)

if not (DATA / "train.bin").is_file():
    raise SystemExit(f"corpus missing under {DATA}")

# Patch the manifest to the mounted dataset.  Hashes stay unchanged.
manifest = json.loads((DATA / "manifest.json").read_text())
for split in ("train", "valid", "test"):
    manifest["bins"][split]["path"] = str(DATA / f"{split}.bin")
manifest["layered_model"] = str(DATA / "tokenizer" / "layered_tokenizer.json")
for value in manifest.get("domains", {}).values():
    value.pop("path", None)
    value.pop("test_text", None)
    value.pop("valid_text", None)
manifest_path = Path("/kaggle/working/frozen_manifest.json")
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
tokenizer = DATA / "tokenizer" / "layered_tokenizer.json"
print("manifest", manifest_path, "tokenizer", tokenizer.exists(), flush=True)
OUT.mkdir(parents=True, exist_ok=True)

common = [
    "--preset", "kaggle10m_small",
    "--corpus", str(DATA / "train.bin"),
    "--val-corpus", str(DATA / "valid.bin"),
    "--data-manifest", str(manifest_path),
    "--layered-model", str(tokenizer),
    "--batch-size", "8",
    "--ctx", "512",
    "--lr", "4e-4",
    "--no-grad-checkpoint",
    "--no-ema",
    "--causal-concept",
    "--moe-dense-dispatch",
    "--threads", "8",
]

if MODE == "smoke":
    cmd = [sys.executable, str(REPO / "z3" / "train_v3.py"),
           "--device", "xla", "--amp",
           *common,
           "--steps", "2",
           "--lr-total-steps", "1000",
           "--max-seconds", "180",
           "--eval-every", "1000",
           "--save-every", "1000",
           "--best-every", "1000",
           "--out", str(OUT)]
    sh(cmd, timeout=900)
else:
    # 8 hours of training; the 9-hour Kaggle kernel timeout leaves room for
    # XLA compilation and the final checkpoint.
    cmd = [sys.executable, str(REPO / "z3" / "train_tpu.py"),
           "--nprocs", "8",
           "--out", str(OUT),
           "--amp",
           *common,
           "--steps", "60000",
           "--lr-total-steps", "60000",
           "--lr-horizon", "time",
           "--max-seconds", "28800",
           "--warmup-steps", "50",
           "--decay-frac", "0.1",
           "--eval-every", "2000",
           "--eval-batches", "1",
           "--save-every", "1000",
           "--best-every", "1000"]
    sh(cmd, timeout=32400)

result_path = OUT / "result.json"
if result_path.is_file():
    result = json.loads(result_path.read_text())
    print("TPU_RESULT", json.dumps({
        "mode": MODE,
        "steps": result.get("steps_completed"),
        "tokens": result.get("tokens_seen"),
        "wall": result.get("wall_seconds"),
        "status": result.get("status"),
        "params": result.get("params"),
        "final_bpb": (result.get("final") or {}).get("bits_per_byte"),
        "final_top1": (result.get("final") or {}).get("accuracy"),
        "best_bpb": ((result.get("best") or {}).get("metrics") or {}).get(
            "bits_per_byte"),
    }, indent=2), flush=True)
print("TPU_KERNEL_DONE", flush=True)
