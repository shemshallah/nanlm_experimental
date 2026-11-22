#!/usr/bin/env python3
"""Full 1M-token per-domain BPB for the 12h checkpoint, single-bundle mount."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path("/kaggle/working/nanlm_experimental")
DATA = Path("/kaggle/input/notebooks/shem42/z4-domain-eval-inputs-12h/inputs")
CKPT = DATA / "final.pt"
TOK = DATA / "tokenizer" / "layered_tokenizer.json"
CAP = 1_000_000

if not (REPO / "z3" / "train_v3.py").is_file():
    subprocess.run([
        "git", "clone", "--depth", "1",
        "https://github.com/shemshallah/nanlm_experimental.git", str(REPO),
    ], check=True)
else:
    subprocess.run(["git", "-C", str(REPO), "pull", "--ff-only"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tokenizers",
                "numpy"], check=True)

sys.path.insert(0, str(REPO / "z3"))
import numpy as np
import torch
import eval_v3
from train_v3 import evaluate
from concept_loop import SpanSidecar

torch.cuda.set_per_process_memory_fraction(0.9)
manifest = json.loads((DATA / "manifest.json").read_text())
domains = manifest["domains"]
order = sorted(domains)
test = np.memmap(DATA / "test.bin", dtype=np.uint16, mode="r")
starts = {}
off = 0
for key in order:
    n = int(domains[key].get("test_tokens", 0))
    starts[key] = (off, off + n)
    off += n
print("test_tokens", len(test), "offset_sum", off, flush=True)

groups = {
    "fineweb_edu": lambda k: k.startswith("fineweb_edu"),
    "code": lambda k: (k.startswith("glaive_python")
                       or k.startswith("stackoverflow_2020")
                       or k == "nanrouter_code_00"),
    "math": lambda k: (k.startswith("finemath_p1")
                       or k == "nanrouter_math_00"),
    "nanrouter_news": lambda k: k == "nanrouter_news_00",
    "cosmopedia": lambda k: k.startswith("cosmopedia_p1"),
    "arxiv": lambda k: (k.startswith("arxiv_abstracts")
                        or k == "nanrouter_arxiv_00"),
    "pubmed": lambda k: k == "nanrouter_pubmed_00",
    "ag_news": lambda k: k == "ag_news_00",
    "tinystories": lambda k: k == "tinystories_00",
    "fineweb70": lambda k: k == "fineweb70_00",
    "law": lambda k: k == "nanrouter_law_00",
    "patents": lambda k: k == "nanrouter_patents_00",
    "finance": lambda k: k == "nanrouter_finance_00",
    "security": lambda k: k == "nanrouter_security_00",
    "standards": lambda k: k == "nanrouter_standards_00",
    "chemistry": lambda k: k == "nanrouter_chemistry_00",
    "recipes": lambda k: k == "nanrouter_recipes_00",
    "qa": lambda k: k == "nanrouter_qa_00",
}

model, cfg, meta = eval_v3.load_checkpoint(CKPT)
model = model.cuda()
print("step", meta.get("step"), "params", model.parameter_count(),
      "causal", getattr(cfg, "causal_concept", None), flush=True)
one = eval_v3._make_sidecar_builder(TOK)


def builder(xb):
    return SpanSidecar.collate(
        [one(row.detach().cpu().tolist()) for row in xb],
        device=xb.device,
    )


def make_batches(ids):
    out = []
    n = (len(ids) // 512) * 512
    for i in range(0, n, 512 * 8):
        block = ids[i:min(i + 512 * 8, n)]
        rows = [torch.from_numpy(block[j:j + 512].astype(np.int64))
                for j in range(0, len(block), 512)]
        if rows:
            out.append(torch.stack(rows))
    return out


result = {}
for name, pred in groups.items():
    keys = [k for k in order if pred(k)]
    pieces = []
    for key in keys:
        a, b = starts[key]
        if b > a:
            pieces.append(np.asarray(test[a:b]))
    if not pieces:
        continue
    ids = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
    if len(ids) > CAP:
        ids = ids[:CAP]
    batches = make_batches(ids)
    if not batches:
        continue
    r = evaluate(
        model, batches, torch.device("cuda"), True, 0.0,
        sidecar_builder=builder, bytes_per_token=None,
        amp_dtype=torch.bfloat16,
    )
    result[name] = {
        "tokens": int(r.get("tokens") or 0),
        "bpb": r.get("bits_per_byte"),
        "top1": r.get("accuracy"),
        "top5": r.get("top5"),
    }
    print(name, json.dumps(result[name]), flush=True)

Path("/kaggle/working/per_domain_gpu12h.json").write_text(
    json.dumps(result, indent=2)
)
print("PER_DOMAIN_DONE", flush=True)
