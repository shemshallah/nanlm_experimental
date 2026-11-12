#!/usr/bin/env python3
"""r31 held-out generation with repetition diagnostics.

Samples 200 bytes at temperature 0.8 / top-k 30 (not greedy), and reports
n-gram repetition plus the longest repeated substring.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import poincare_frontier as pf  # noqa: E402


def repetition_metrics(text: str) -> dict:
    out = {}
    for n in (3, 4, 5):
        grams = [text[i:i + n] for i in range(max(0, len(text) - n + 1))]
        out[f"unique_{n}gram_fraction"] = round(len(set(grams)) / max(len(grams), 1), 6)
        out[f"repetition_{n}gram"] = round(1.0 - len(set(grams)) / max(len(grams), 1), 6)
    grams4 = [text[i:i + 4] for i in range(max(0, len(text) - 3))]
    if grams4:
        gram, count = Counter(grams4).most_common(1)[0]
        out["most_common_4gram"] = {"text": gram, "count": count}
    best = 0
    best_text = ""
    for i in range(len(text)):
        for j in range(i + 1, len(text)):
            k = 0
            while j + k < len(text) and text[i + k] == text[j + k]:
                k += 1
            if k > best:
                best = k
                best_text = text[i:i + k]
    out["longest_repeated_substring"] = {"length": best, "text": best_text}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--topk-sample", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    saved = torch.load(pf.checkpoint_path(args.run_dir), map_location="cpu", weights_only=False)
    cfg = pf.Config(**saved["config"])
    model = pf.MirrorLM(cfg).to(device)
    pf.load_v25_compatible(model, saved["model"])
    model.eval()
    codec = pf.TokenCodec(saved["data"]["tokenizer_model"])
    prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    ids = codec.encode(prompt)
    prompt_len = len(ids)
    torch.manual_seed(args.seed)
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            inp = torch.tensor([ids[-cfg.context:]], dtype=torch.long, device=device)
            logits, _, _ = model(inp)
            scores = logits[0, -1].float() / max(args.temperature, 1e-4)
            if args.topk_sample > 0:
                vals, top_ids = scores.topk(min(args.topk_sample, scores.numel()))
                p = vals.softmax(-1)
                idx = int(top_ids[torch.multinomial(p, 1)].item())
            else:
                idx = int(scores.argmax().item())
            ids.append(idx)
    generated = codec.decode(ids[prompt_len:])
    result = {
        "run_dir": str(Path(args.run_dir).resolve()),
        "checkpoint_step": int(saved["step"]),
        "prompt": prompt,
        "generated": generated,
        "generated_bytes": len(ids) - prompt_len,
        "temperature": args.temperature,
        "topk_sample": args.topk_sample,
        "repetition": repetition_metrics(generated),
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text, flush=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
