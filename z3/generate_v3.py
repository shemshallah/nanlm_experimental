#!/usr/bin/env python3
"""Generate text from a z3 V3 checkpoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import eval_v3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample text from a z3 checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--layered-model",
        default=str(eval_v3.LAYERED_TOKENIZER_PATH),
    )
    parser.add_argument("--feedback-progress", type=float, default=1.0)
    return parser


def _decode(tokenizer, ids: list[int]) -> str:
    layered = getattr(tokenizer, "layered", tokenizer)
    if hasattr(layered, "decode_token_ids"):
        return layered.decode_token_ids(ids)
    if hasattr(layered, "decode"):
        return layered.decode(ids)
    return str(ids)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prompts = args.prompt or ["Once upon a time"]
    device = eval_v3._resolve_device(args.device)
    model, _cfg, _metadata = eval_v3.load_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()
    if hasattr(model, "set_feedback_progress"):
        model.set_feedback_progress(float(args.feedback_progress))
    tokenizer = eval_v3.load_tokenizer(
        "layered", None, args.layered_model
    )
    # ``LayeredTokenizerAdapter`` only exposes flat IDs; pass the underlying
    # layered tokenizer so generation can rebuild sidecars and decode text.
    generation_tokenizer = getattr(tokenizer, "_layered", tokenizer)
    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        if not ids:
            print(f"[skip] empty prompt: {prompt!r}", file=sys.stderr)
            continue
        tokens = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            generated, payload = model.generate(
                tokens,
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                seed=int(args.seed),
                tokenizer=generation_tokenizer,
                top_k=int(args.top_k),
                top_p=float(args.top_p),
                repetition_penalty=float(args.repetition_penalty),
            )
        out_ids = generated[0].detach().cpu().tolist()
        print(f"\n=== PROMPT: {prompt}")
        print(_decode(generation_tokenizer, out_ids))
        if isinstance(payload, dict) and payload.get("composition"):
            print(f"[composition events: {len(payload['composition'])}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
