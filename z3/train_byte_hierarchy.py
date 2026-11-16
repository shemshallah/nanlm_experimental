#!/usr/bin/env python3
"""Byte -> subword -> word hierarchical byte model on a GPU-resident pool.

This is the byte-model counterpart of the previous 112.9M byte run.  It keeps
the byte vocabulary and byte-level top-k accounting, adds a subword-ID
embedding and word/sentence-aligned concept feedback, and removes host-side
random-window sampling from the hot path.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .concept_loop import MATH_KINDS, EVENT_NAMES, SpanSidecar
    from .train_v3 import lm_mtp_loss
    from .v3_model import V3Config, V3LM
except ImportError:  # script execution from z3/
    from concept_loop import MATH_KINDS, EVENT_NAMES, SpanSidecar
    from train_v3 import lm_mtp_loss
    from v3_model import V3Config, V3LM


def build_byte_sidecar(raw: bytes, layered) -> tuple[torch.Tensor, SpanSidecar]:
    text = raw.decode("utf-8", errors="ignore")
    text_bytes = text.encode("utf-8")
    n = len(text_bytes)
    encoding = layered.encode(text, add_document_frame=False)
    spans = list(encoding.subword.spans)
    subword_ids = torch.zeros(n, dtype=torch.long)
    for (start, end), token_id in zip(spans, encoding.subword.ids):
        start = max(0, min(n, int(start)))
        end = max(start, min(n, int(end)))
        if end > start:
            subword_ids[start:end] = int(token_id)

    word_ids = torch.zeros(n, dtype=torch.long)
    word_vocab_ids = torch.zeros(n, dtype=torch.long)
    sentence_ids = torch.zeros(n, dtype=torch.long)
    math_kind = torch.zeros(n, dtype=torch.long)
    event_ids = torch.zeros(n, dtype=torch.long)
    event_multi = torch.zeros(n, len(EVENT_NAMES), dtype=torch.float32)

    def mark(index: int, event: int) -> None:
        if 0 <= index < n:
            event_ids[index] = event
            event_multi[index, event] = 1.0

    def fill(span_start: int, span_end: int, target: torch.Tensor, value: int):
        start = max(0, min(n, int(span_start)))
        end = max(start, min(n, int(span_end)))
        if end > start:
            target[start:end] = int(value)
        return start, end

    word_units = [unit for unit in encoding.word_units if unit.is_word]
    for word_index, unit in enumerate(word_units, start=1):
        start, end = fill(unit.start, unit.end, word_ids, word_index)
        if end <= start:
            continue
        if unit.word_id is not None:
            word_vocab_ids[start:end] = int(unit.word_id) + 1
        mark(start, EVENT_NAMES.index("<word_start>"))
        mark(end - 1, EVENT_NAMES.index("<word_end>"))
        if end - start > 1:
            mark(end - 1, EVENT_NAMES.index("<compose>"))

    for sent_index, (start, end) in enumerate(encoding.sentence_spans, start=1):
        start, end = fill(start, end, sentence_ids, sent_index)
        if end > start:
            mark(end - 1, EVENT_NAMES.index("<sentence_end>"))

    for span in getattr(encoding, "math_spans", []):
        name = f"<{span.kind}>"
        kind = MATH_KINDS.index(name) if name in MATH_KINDS else 0
        start, end = fill(span.start, span.end, math_kind, kind)
        if end > start:
            mark(start, EVENT_NAMES.index("<math_start>"))
            mark(end - 1, EVENT_NAMES.index("<math_end>"))

    sidecar = SpanSidecar(
        word_ids=word_ids,
        word_vocab_ids=word_vocab_ids,
        sentence_ids=sentence_ids,
        math_kind=math_kind,
        event_ids=event_ids,
        byte_start=torch.arange(n, dtype=torch.long),
        byte_len=torch.ones(n, dtype=torch.long),
        char_len=torch.ones(n, dtype=torch.long),
        first_byte=torch.tensor(list(text_bytes), dtype=torch.long),
        event_multi=event_multi,
        token_bytes=None,
        subword_ids=subword_ids,
    )
    return torch.tensor(list(text_bytes), dtype=torch.uint8), sidecar


def load_pool(path: Path, tokenizer, max_bytes: int):
    raw = path.read_bytes()[:max_bytes]
    cut = raw.rfind(b"\n")
    if cut > 0:
        raw = raw[:cut]
    return build_byte_sidecar(raw, tokenizer)


@torch.inference_mode()
def evaluate(model, pool_bytes, pool_sidecar, *, ctx, device, batches=8):
    model.eval()
    total = correct = top5 = 0
    loss_sum = 0.0
    n = int(pool_bytes.numel())
    starts = torch.arange(0, max(1, n - ctx - 1), max(1, (n - ctx - 1) // batches))
    starts = starts[:batches]
    for start in starts.tolist():
        xb = pool_bytes[start:start + ctx + 1].to(device, torch.long).unsqueeze(0)
        if xb.size(1) < 2:
            continue
        sc = SpanSidecar.collate(
            [pool_sidecar.slice(start, xb.size(1))], device=device
        )
        logits, _mtp, _aux = model(xb, sidecar=sc)
        logits = logits[:, :-1]
        targets = xb[:, 1:]
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        )
        loss_sum += float(ce.sum())
        total += int(targets.numel())
        correct += int((logits.argmax(-1) == targets).sum())
        top5 += int(
            (logits.topk(5, -1).indices == targets.unsqueeze(-1))
            .any(-1).sum()
        )
    model.train(True)
    return {
        "bytes": total,
        "bpb": loss_sum / max(1, total) / math.log(2),
        "top1": correct / max(1, total),
        "top5": top5 / max(1, total),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-text", required=True)
    parser.add_argument("--valid-text", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--valid-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--ctx", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--max-seconds", type=float, default=600.0)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--n-experts", type=int, default=64)
    parser.add_argument("--expert-hidden", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    from layered_tokenizer import load_artifact

    tokenizer = load_artifact(
        Path(__file__).resolve().with_name("tokenizers")
        / "layered_v1" / "layered_tokenizer.json"
    )
    train_bytes, train_side = load_pool(
        Path(args.train_text), tokenizer, args.train_bytes
    )
    valid_bytes, valid_side = load_pool(
        Path(args.valid_text), tokenizer, args.valid_bytes
    )
    train_bytes = train_bytes.to(device)
    valid_bytes = valid_bytes.to(device)
    train_side = train_side.to(device)
    valid_side = valid_side.to(device)
    print(
        f"[data] train_bytes={train_bytes.numel():,} "
        f"valid_bytes={valid_bytes.numel():,}",
        flush=True,
    )

    cfg = V3Config(
        vocab_size=273, ctx=args.ctx, max_ctx=8192,
        d_model=256, n_layers=8, n_heads=8, n_kv_heads=0,
        swiglu=1024, window=512, global_every=4,
        moe_layers=4, n_experts=args.n_experts, topk=2,
        shared_experts=2, expert_hidden=args.expert_hidden,
        moe_group_size=4, moe_max_rows=512,
        use_memory=False, use_ngram_memory=False,
        use_concept_loop=True, subword_embedding=True,
        subword_vocab_size=64060, composition_head=False,
        attn_gate=True, mtp=2, mtp_block=False,
        dropout=0.1,
    )
    model = V3LM(cfg).to(device)
    n_params = model.parameter_count()
    print(f"[model] params={n_params:,}", flush=True)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.05
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    arange = torch.arange(args.ctx + 1, device=device)
    n = int(train_bytes.numel())
    started = time.time()
    step_done = 0
    history = []
    model.train()

    while step_done < args.steps:
        if time.time() - started >= args.max_seconds:
            break
        starts = torch.randint(
            0, n - args.ctx - 1, (args.batch_size,), device=device
        )
        xb = train_bytes[starts[:, None] + arange].long()
        side = SpanSidecar.collate(
            [train_side.slice(int(s), args.ctx + 1)
             for s in starts.detach().cpu().tolist()],
            device=device,
        )
        lr = args.lr * min(1.0, (step_done + 1) / max(1, args.warmup_steps))
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits, mtp, aux = model(xb, sidecar=side)
            loss, parts = lm_mtp_loss(
                logits, mtp, xb, mtp_weight=0.3
            )
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step_done += 1
        if step_done % args.log_every == 0:
            elapsed = time.time() - started
            ce = float(parts["ce"].detach())
            row = {
                "step": step_done,
                "loss": float(loss.detach()),
                "ce": ce,
                "bpb": ce / math.log(2),
                "lr": lr,
                "bytes_seen": step_done * args.batch_size * args.ctx,
                "bytes_per_sec": step_done * args.batch_size * args.ctx / elapsed,
                "vram_gb": torch.cuda.memory_allocated() / 1e9,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            (out / "history.json").write_text(
                json.dumps(history, indent=2) + "\n"
            )

    eval_metrics = evaluate(
        model, valid_bytes, valid_side, ctx=args.ctx, device=device
    )
    print("[eval]", json.dumps(eval_metrics), flush=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": cfg,
            "steps": step_done,
            "params": n_params,
            "eval": eval_metrics,
        },
        out / "final.pt",
    )
    result = {
        "params": n_params,
        "steps": step_done,
        "tokens_seen": step_done * args.batch_size * args.ctx,
        "wall_seconds": time.time() - started,
        "history": history,
        "eval": eval_metrics,
        "config": cfg.__dict__,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
