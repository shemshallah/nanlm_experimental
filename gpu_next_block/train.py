#!/usr/bin/env python3
"""Kaggle GPU 6-hour causal continuation from the 175.76M-token checkpoint.

Uses fp16 (native T4 tensor cores) and a corrected low-peak WSD schedule:
1e-4 peak with decay-frac 0.5 instead of 4e-4 with decay-frac 0.1.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path("/kaggle/working/nanlm_experimental")
DATA = Path("/kaggle/input/notebooks/shem42/z4-corpus-150m-mirror/corpus")
CKPT = Path("/kaggle/input/notebooks/shem42/z4-causal-gpu12h/runs/causal_gpu12h/final.pt")
OUT = Path("/kaggle/working/runs/causal_gpu18h")


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

OUT.mkdir(parents=True, exist_ok=True)

base = [
    sys.executable, str(REPO / "z3" / "train_v3.py"),
    "--device", "cuda", "--amp", "--amp-dtype", "fp16",
    "--preset", "kaggle10m_small",
    "--corpus", str(DATA / "train.bin"),
    "--val-corpus", str(DATA / "valid.bin"),
    "--layered-model", str(DATA / "tokenizer" / "layered_tokenizer.json"),
    "--batch-size", "8", "--ctx", "512", "--lr", "1e-4",
    "--resume", str(CKPT),
    "--steps", "100000", "--lr-total-steps", "100000",
    "--lr-horizon", "time", "--max-seconds", "21600",
    "--warmup-steps", "50", "--decay-frac", "0.5",
    "--eval-every", "2000", "--eval-batches", "1",
    "--save-every", "1000", "--best-every", "1000",
    "--no-grad-checkpoint", "--no-ema",
    "--causal-concept", "--moe-dense-dispatch",
    "--chunked-ce", "--ce-chunk-tokens", "2048",
    "--threads", "8",
    "--out", str(OUT),
]
print("GPU18H_CMD", " ".join(str(x) for x in base), flush=True)
rc = sh(base, timeout=23400, check=False).returncode
print("train_rc", rc, flush=True)
if rc != 0:
    # One conservative retry at half the batch if the first attempt OOMed
    # before producing a result.
    if not (OUT / "result.json").is_file():
        retry = list(base)
        i = retry.index("--batch-size")
        retry[i + 1] = "4"
        print("retrying batch 4", flush=True)
        sh(retry, timeout=23400, check=False)

# PTB OOD eval.
if (OUT / "final.pt").is_file():
    ptb = Path("/kaggle/working/ptb.test.txt")
    if not ptb.is_file() or ptb.stat().st_size < 100000:
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt",
            ptb,
        )
    sh([sys.executable, str(REPO / "z3" / "eval_domains.py"),
        "--checkpoint", str(OUT / "final.pt"),
        "--corpus", f"ptb={ptb}",
        "--out", str(OUT / "eval_ptb.json"),
        "--device", "cuda", "--batch-size", "8", "--bootstrap", "100"],
       timeout=1800, check=False)

if (OUT / "result.json").is_file():
    result = json.loads((OUT / "result.json").read_text())
    print("GPU18H_RESULT", json.dumps({
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
print("GPU18H_DONE", flush=True)
