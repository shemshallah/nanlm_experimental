#!/usr/bin/env python3
"""Short T4 AMP precision probe: bf16 vs fp16 throughput at batch 8."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path("/kaggle/working/nanlm_experimental")
DATA = Path("/kaggle/input/datasets/shem42/z4-broad7h-bins")
CKPT = Path("/kaggle/input/datasets/shem42/z4-causal-gpu6h-weights/final.pt")
OUT = Path("/kaggle/working/runs/probe")


def sh(cmd, timeout=None, check=True):
    print("$", " ".join(str(x) for x in cmd), flush=True)
    r = subprocess.run([str(x) for x in cmd], timeout=timeout, check=False)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed rc={r.returncode}")
    return r


print("image", os.environ.get("COLAB_IMAGE_TYPE"),
      "cuda", os.path.exists("/dev/nvidia0"), flush=True)
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
if not CKPT.is_file():
    raise SystemExit(f"checkpoint missing at {CKPT}")

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
OUT.mkdir(parents=True, exist_ok=True)

common = [
    sys.executable, str(REPO / "z3" / "train_v3.py"),
    "--device", "cuda", "--amp",
    "--preset", "kaggle10m_small",
    "--corpus", str(DATA / "train.bin"),
    "--val-corpus", str(DATA / "valid.bin"),
    "--data-manifest", str(manifest_path),
    "--layered-model", str(DATA / "tokenizer" / "layered_tokenizer.json"),
    "--batch-size", "8", "--ctx", "512", "--lr", "4e-4",
    "--resume", str(CKPT),
    "--steps", "100000", "--lr-total-steps", "100000",
    "--lr-horizon", "time", "--max-seconds", "240",
    "--warmup-steps", "50", "--decay-frac", "0.1",
    "--eval-every", "100000", "--eval-batches", "1",
    "--save-every", "100000", "--best-every", "100000",
    "--no-grad-checkpoint", "--no-ema",
    "--causal-concept", "--moe-dense-dispatch",
    "--chunked-ce", "--ce-chunk-tokens", "2048",
    "--threads", "8",
]

for tag, dtype in (("bf16", "bf16"), ("fp16", "fp16")):
    out = Path(f"/kaggle/working/runs/probe_{tag}")
    out.mkdir(parents=True, exist_ok=True)
    cmd = [*common, "--amp-dtype", dtype, "--out", str(out)]
    print("PROBE", tag, " ".join(str(x) for x in cmd), flush=True)
    rc = sh(cmd, timeout=1200, check=False).returncode
    print("PROBE_RC", tag, rc, flush=True)

print("PROBE_DONE", flush=True)

