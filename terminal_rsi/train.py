"""CPU-first training and evaluation of the unified Terminal-RSI model.

No Google Colab dependency. Run unchanged on CPU, T4, or 96GB Blackwell.
Book splits use NaNROUTER release text; no WordNet-only model promotion.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import torch

from unified import PRESETS, Preset, RSIController, TerminalRSI, device_for, loss_and_stats


class BookCorpus:
    def __init__(self, root: Path):
        files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".text"})
        if not files:
            raise ValueError(f"No .txt/.text books under {root}; extract NaNROUTER Release first")
        self.books = {"train": [], "dev": [], "test": []}
        digest = hashlib.sha256()
        for path in files:
            relative = path.relative_to(root).as_posix()
            raw = path.read_bytes()
            if len(raw) < 16:
                continue
            digest.update(relative.encode("utf-8") + b"\0")
            digest.update(hashlib.sha256(raw).digest())
            parts = path.relative_to(root).parts
            split = parts[0].lower() if len(parts) > 1 and parts[0].lower() in self.books else None
            if split is None:
                rank = int(hashlib.sha256(relative.encode()).hexdigest()[:8], 16) % 100
                split = "train" if rank < 90 else ("dev" if rank < 95 else "test")
            self.books[split].append((relative, raw))
        if not self.books["train"] or not self.books["dev"]:
            raise RuntimeError("Need distinct TRAIN and DEV books. Provide train/ and dev/ directories.")
        self.digest = digest.hexdigest()
        self.manifest = {
            "sha256": self.digest,
            "books": {split: [name for name, _ in books] for split, books in self.books.items()},
            "positions": {split: sum(len(data) for _, data in books) for split, books in self.books.items()},
        }
        # No file can occur in two splits.
        sets = [set(self.manifest["books"][split]) for split in self.books]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def batch(self, split: str, batch_size: int, context: int,
              rng: random.Random, device: torch.device):
        books = self.books[split]
        if not books:
            raise RuntimeError(f"Split {split} has no books")
        raw_all, mask_all = [], []
        window = context + 3
        for _ in range(batch_size):
            _, data = books[rng.randrange(len(books))]
            # All context comes from strictly within the same causal book.
            start = rng.randrange(max(1, len(data) - 3))
            fragment = data[start:start + window]
            if len(fragment) < 4:
                fragment = data[max(0, len(data) - window):]
            fragment = fragment[-window:]
            pad = window - len(fragment)
            raw_all.append([0] * pad + list(fragment))
            mask_all.append([False] * pad + [True] * len(fragment))
        return (torch.tensor(raw_all, device=device, dtype=torch.long),
                torch.tensor(mask_all, device=device, dtype=torch.bool))


@torch.no_grad()
def evaluate(model: TerminalRSI, corpus: BookCorpus, device: torch.device,
             context: int, windows: int = 8, batch_size: int = 1):
    was_training = model.training
    model.eval()
    rng = random.Random(0x534941)  # matched replayable DEV windows
    sums = {f"h{h}_{metric}": 0.0 for h in range(1, 4)
            for metric in ("bits", "top1", "top5")}
    totals = {h: 0 for h in range(1, 4)}
    eval_context = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if device.type == "cuda" and torch.cuda.is_bf16_supported()
                    else nullcontext())
    with eval_context:
        for _ in range(windows):
            raw, valid = corpus.batch("dev", batch_size, context, rng, device)
            outputs = model(raw[:, :-3])
            _, metrics = loss_and_stats(outputs, raw, valid)
            for h in range(1, 4):
                n = int(metrics[f"h{h}_bytes"].item())
                totals[h] += n
                for m in ("bits", "top1", "top5"):
                    sums[f"h{h}_{m}"] += float(metrics[f"h{h}_{m}"].item()) * n
    model.train(was_training)
    return {f"h{h}_{m}": sums[f"h{h}_{m}"] / max(1, totals[h])
            for h in range(1, 4) for m in ("bits", "top1", "top5")} | {
            "dev_bytes": totals[1]}


def save_checkpoint(path: Path, model: TerminalRSI, optimizer, controller: RSIController,
                    step: int, corpus: BookCorpus, best: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format": "terminal-rsi-unified-v1",
        "config": vars(model.cfg),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "controller": vars(controller).copy(),
        "step": step,
        "corpus_sha256": corpus.digest,
        "split_manifest": corpus.manifest,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    # Create new checkpoint atomically; never touch historical hybrid_latest.pt.
    tmp = path.with_name(path.name + ".incomplete")
    torch.save(state, tmp)
    os.replace(tmp, path)
    if best:
        print("Accepted matched-DEV candidate:", path, flush=True)


def restore_checkpoint(path: Path, model: TerminalRSI, optimizer,
                       controller: RSIController, corpus: BookCorpus):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != "terminal-rsi-unified-v1":
        raise RuntimeError("Legacy or unknown format; use explicit migration, NEVER silent reinit")
    if state["config"] != vars(model.cfg):
        raise RuntimeError("Preset/architecture changed; exact optimizer resume prohibited")
    if state["corpus_sha256"] != corpus.digest:
        raise RuntimeError("Training corpus changed; refuse checkpoint resume")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    controller.__dict__.update(state["controller"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return int(state["step"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=PRESETS, default="cpu")
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("terminal_rsi_unified_candidate.pt"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means wall-time only")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-windows", type=int, default=8)
    parser.add_argument("--context-override", type=int, default=0,
                        help="shorter training windows without changing model weights")
    args = parser.parse_args()

    torch.manual_seed(17011)
    rng = random.Random(17011)
    cfg = PRESETS[args.mode]
    device = device_for(cfg.name)
    if device.type == "cpu":
        torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.set_per_process_memory_fraction(0.90)
    corpus = BookCorpus(args.corpus_dir)
    model = TerminalRSI(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    controller = RSIController(cfg)
    first_step = restore_checkpoint(args.checkpoint, model, opt, controller, corpus) if args.resume else 0
    print(json.dumps({
        "mode": cfg.name, "parameters": model.parameter_count(),
        "device": str(device), "context": cfg.context,
        "train_books": len(corpus.books["train"]), "dev_books": len(corpus.books["dev"]),
        "train_bytes": corpus.manifest["positions"]["train"],
        "dev_bytes": corpus.manifest["positions"]["dev"],
        "corpus_sha256": corpus.digest,
        "previous_step": first_step,
    }), flush=True)

    chosen_context = cfg.context if args.context_override <= 0 else min(args.context_override, cfg.context)
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
                if device.type == "cuda" else nullcontext())
    uses_scaler = device.type == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=uses_scaler)
    started = time.monotonic()
    deadline = started + 60 * args.minutes
    step = first_step
    last_eval = None
    # Strict CPU-first launch: GPU presets require prior CPU CI acceptance.
    while time.monotonic() < deadline - 60 and (args.max_steps == 0 or step - first_step < args.max_steps):
        model.train()
        opt.zero_grad(set_to_none=True)
        try:
            for _ in range(cfg.grad_accum):
                batch, valid = corpus.batch("train", cfg.microbatch, chosen_context, rng, device)
                with autocast:
                    pred = model(batch[:, :-3], passes=controller.passes)
                    loss, metrics = loss_and_stats(pred, batch, valid)
                    loss = loss / cfg.grad_accum
                scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
        except torch.cuda.OutOfMemoryError:
            if device.type != "cuda" or chosen_context <= 256:
                raise
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            chosen_context = max(256, chosen_context // 2)
            print(f"CUDA memory autotune: reduced context to {chosen_context}", flush=True)
            continue
        if step == first_step + 1 or step % args.eval_every == 0:
            last_eval = evaluate(model, corpus, device, chosen_context,
                                 windows=args.eval_windows)
            changed = controller.observe(opt, step, last_eval["h1_bits"])
            print(json.dumps({
                "step": step, "elapsed_sec": round(time.monotonic() - started, 1),
                "train_loss_nats": float(loss.item() * cfg.grad_accum),
                "DEV": last_eval,
                "RSI_action": controller.history[-1],
                "gpu_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3) if device.type == "cuda" else 0,
                "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3) if device.type == "cuda" else 0,
            }), flush=True)
            if changed:
                save_checkpoint(args.checkpoint.with_name("terminal_rsi_unified_best.pt"),
                                model, opt, controller, step, corpus, best=True)
        if step % 100 == 0:
            save_checkpoint(args.checkpoint, model, opt, controller, step, corpus)
    save_checkpoint(args.checkpoint, model, opt, controller, step, corpus)
    print(json.dumps({"finished_step": step, "seconds": round(time.monotonic()-started,1),
                      "last_dev": last_eval, "checkpoint": str(args.checkpoint)}), flush=True)


if __name__ == "__main__":
    main()
