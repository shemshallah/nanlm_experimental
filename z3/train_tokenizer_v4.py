#!/usr/bin/env python3
"""Train a small BPE tokenizer v4 smoke on FineWeb-Edu text."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

SPECIALS = ["<pad>", "<bos>", "<eos>", "<doc>", "<now>", "<future_1>", "<future_2>", "<future_3>"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--config", default="sample-10BT")
    parser.add_argument("--train-bytes", type=int, default=20_000_000)
    parser.add_argument("--eval-bytes", type=int, default=2_000_000)
    parser.add_argument("--vocab-size", type=int, default=64000)
    parser.add_argument("--out", default="/kaggle/working/tokenizer_v4")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        from tokenizers import Tokenizer, trainers
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
    except ModuleNotFoundError as exc:
        raise SystemExit("install `datasets tokenizers` first") from exc

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    kwargs = {"split": "train", "streaming": True}
    if token:
        kwargs["token"] = token
    ds = load_dataset(args.dataset, name=args.config, **kwargs)
    ds = ds.shuffle(seed=20261113, buffer_size=10_000)

    train_texts: list[str] = []
    eval_texts: list[str] = []
    train_bytes = 0
    eval_bytes = 0
    for example in ds:
        text = example.get("text") or ""
        if not text:
            continue
        raw = len(text.encode("utf-8", errors="ignore"))
        if eval_bytes < args.eval_bytes:
            eval_texts.append(text)
            eval_bytes += raw
        elif train_bytes < args.train_bytes:
            train_texts.append(text)
            train_bytes += raw
        else:
            break
    if not train_texts or not eval_texts:
        raise SystemExit("tokenizer smoke did not collect enough text")

    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIALS,
        min_frequency=2,
        show_progress=True,
    )
    tokenizer.train_from_iterator(train_texts, trainer=trainer)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer_path = out / f"tokenizer_v4_{args.vocab_size // 1000}k.json"
    tokenizer.save(str(tokenizer_path))

    eval_tokens = 0
    for text in eval_texts:
        eval_tokens += len(tokenizer.encode(text).ids)
    eval_total_bytes = sum(len(t.encode("utf-8", errors="ignore")) for t in eval_texts)
    bytes_per_token = eval_total_bytes / max(1, eval_tokens)
    manifest = {
        "train_bytes": train_bytes,
        "eval_bytes": eval_total_bytes,
        "eval_tokens": eval_tokens,
        "bytes_per_token": bytes_per_token,
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": SPECIALS,
        "tokenizer": str(tokenizer_path),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    if bytes_per_token < 3.8:
        raise SystemExit(f"compression gate failed: {bytes_per_token:.3f} bytes/token < 3.8")


if __name__ == "__main__":
    main()
