"""Character/byte OOV fallback using the trained unit->byte decoder.

When the word head's top class is UNK, the byte path can still spell the word.
This module runs a small beam over the hierarchy's 256-way next-byte decoder
(``unit_byte_raw``), which is trained on the next byte at every position.
No target word bytes are ever fed back as context.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

WORD_BYTES = set(
    b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    b"_-'")
TERMINATORS = set(b" \t\n\r.,;:!?()[]{}\"")
MAX_CONTEXT = 256


def _allowed(byte: int) -> bool:
    return byte in WORD_BYTES


def _batch_contexts(contexts: list[bytes]) -> torch.Tensor:
    trimmed = [c[-MAX_CONTEXT:] for c in contexts]
    width = max((len(c) for c in trimmed), default=0)
    if width == 0:
        raise ValueError("empty OOV history")
    rows = []
    for context in trimmed:
        pad = width - len(context)
        rows.append([0] * pad + list(context))
    return torch.tensor(rows, dtype=torch.long)


@torch.no_grad()
def next_byte_logprobs(model, contexts: list[bytes],
                       device: torch.device | None = None) -> torch.Tensor:
    """Return log p(next byte) for each context, using the unit byte decoder."""
    device = device or next(model.parameters()).device
    x = _batch_contexts(contexts).to(device)
    out = model.upgrades(x)
    raw = out.get("unit_byte_raw")
    if raw is None:
        raise RuntimeError("model has no unit_byte_raw decoder")
    return F.log_softmax(raw.float(), dim=-1)


@torch.no_grad()
def spell_oov(model, history: bytes, *, beam: int = 8, max_len: int = 24,
              top_k: int = 5, length_penalty: float = 0.6,
              device: torch.device | None = None) -> list[tuple[str, float]]:
    """Beam-spell the next word surface from a history ending at word start."""
    if beam < 1 or max_len < 1 or top_k < 1:
        raise ValueError("beam, max_len and top_k must be positive")
    active: list[tuple[bytes, float]] = [(b"", 0.0)]
    finished: list[tuple[bytes, float]] = []
    for _step in range(max_len):
        if not active:
            break
        contexts = [history + prefix for prefix, _score in active]
        logp = next_byte_logprobs(model, contexts, device=device)
        expanded = []
        for row, (prefix, score) in enumerate(active):
            values = logp[row]
            # Allowed continuation bytes.
            for byte in sorted(WORD_BYTES):
                expanded.append((prefix + bytes([byte]),
                                 score + float(values[byte])))
            # Terminator bytes.
            for byte in sorted(TERMINATORS):
                finished.append((prefix, score + float(values[byte])))
        expanded.sort(key=lambda item: item[1], reverse=True)
        active = expanded[:max(1, beam)]
    candidates = finished + active
    ranked = []
    seen = set()
    for prefix, score in sorted(candidates, key=lambda item: item[1],
                                reverse=True):
        if not prefix or prefix in seen:
            continue
        seen.add(prefix)
        ranked.append((prefix.decode("utf-8", "replace"),
                       score / max(1, len(prefix)) ** length_penalty))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:top_k]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--history", required=True,
                        help="UTF-8 history ending at word start")
    parser.add_argument("--beam", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=24)
    parser.add_argument("--mode", default="blackwell")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    from .unified import device_for
    from .verbatim_v103 import VerbatimV103RSI

    device = device_for(args.mode)
    checkpoint = torch.load(args.checkpoint, map_location="cpu",
                            weights_only=True)
    vocab_size = int(checkpoint.get("word_vocab_size", 4096))
    model = VerbatimV103RSI(args.mode, word_vocab_size=vocab_size).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    result = spell_oov(model, args.history.encode("utf-8"),
                       beam=args.beam, max_len=args.max_len, device=device)
    report = {"history": args.history, "candidates": result}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
