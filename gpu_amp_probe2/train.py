#!/usr/bin/env python3
"""Self-contained T4 AMP probe: bf16 vs fp16 on a generated corpus.

Large private dataset mounts are currently stuck in the Kaggle GPU queue,
so this kernel uses:
- kernel_sources for the 12h checkpoint (338 MB, mounts instantly)
- a corpus generated at runtime from repo text (no dataset mount)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path("/kaggle/working/nanlm_experimental")
CKPT = Path(
    "/kaggle/input/notebooks/shem42/z4-causal-gpu12h/runs/causal_gpu12h/final.pt"
)
TOK = REPO / "z3" / "tokenizers" / "layered_v1" / "layered_tokenizer.json"
TRAIN_BIN = Path("/kaggle/working/probe_train.bin")
VALID_BIN = Path("/kaggle/working/probe_valid.bin")


def sh(cmd, timeout=None, check=True):
    print("$", " ".join(str(x) for x in cmd), flush=True)
    r = subprocess.run([str(x) for x in cmd], timeout=timeout, check=False)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed rc={r.returncode}")
    return r


if not (REPO / "z3" / "train_v3.py").is_file():
    sh(["git", "clone", "--depth", "1",
        "https://github.com/shemshallah/nanlm_experimental.git", str(REPO)],
       timeout=600)
else:
    sh(["git", "-C", str(REPO), "pull", "--ff-only"], timeout=300)
sh([sys.executable, "-m", "pip", "install", "-q", "tokenizers", "numpy"],
   timeout=1800)

sys.path.insert(0, str(REPO / "z3"))
import numpy as np  # noqa: E402
from layered_tokenizer import load_artifact  # noqa: E402

tok = load_artifact(TOK)
texts = [p.read_text(errors="ignore") for p in sorted((REPO / "z3").glob("*.md"))]
texts.append((REPO / "README.md").read_text(errors="ignore"))
texts.extend(
    p.read_text(errors="ignore")
    for p in sorted((REPO / "z3").glob("*.py"))[:20]
)
ids: list[int] = []
for text in texts:
    ids.extend(tok.encode(text, add_document_frame=True).token_ids)
base = np.asarray(ids, dtype=np.uint16)
if len(base) < 100_000:
    raise SystemExit(f"generated corpus too small: {len(base)}")
target = 8_000_000
train = np.tile(base, target // len(base) + 1)[:target]
valid_target = 1_000_000
valid = np.tile(base, valid_target // len(base) + 1)[:valid_target]
train.tofile(TRAIN_BIN)
valid.tofile(VALID_BIN)
print("generated", len(base), "unique tokens; train", train.size,
      "valid", valid.size, flush=True)

common = [
    sys.executable, str(REPO / "z3" / "train_v3.py"),
    "--device", "cuda", "--amp",
    "--preset", "kaggle10m_small",
    "--corpus", str(TRAIN_BIN),
    "--val-corpus", str(VALID_BIN),
    "--layered-model", str(TOK),
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
    out = Path(f"/kaggle/working/runs/probe2_{tag}")
    out.mkdir(parents=True, exist_ok=True)
    cmd = [*common, "--amp-dtype", dtype, "--out", str(out)]
    print("PROBE", tag, " ".join(str(x) for x in cmd), flush=True)
    rc = sh(cmd, timeout=1200, check=False).returncode
    print("PROBE_RC", tag, rc, flush=True)

print("AMP_PROBE2_DONE", flush=True)
