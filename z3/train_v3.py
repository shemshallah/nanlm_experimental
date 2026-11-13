#!/usr/bin/env python3
"""Training harness for the z3 V3LM prototype (``z3/v3_model.py``).

Text mode (BPE via the optional ``tokenizers`` package)::

    python z3/train_v3.py \
        --corpus data/a.txt data/b.txt \
        --bpe-model bpe32k.json \
        --ctx 1024 --steps 20000 --max-seconds 1800 \
        --out runs/v3

Smoke mode (random token IDs, no tokenizer or corpus, CPU, 3-5 steps)::

    python z3/train_v3.py --random-smoke --steps 3 --out /tmp/v3_smoke

The text objective is ordinary next-token cross entropy plus one auxiliary
cross entropy per MTP head.  Head ``i`` predicts ``t + 2 + i`` and receives
weight ``--mtp-weight / k`` (``k`` is the number of MTP heads).  MoE routing
gets Switch-style load-balance and router z-loss terms with small weights.
"""

from __future__ import annotations

import argparse
import codecs
import contextlib
import copy
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v3_model import V3Config, V3LM  # noqa: E402

IGNORE_INDEX = -100
READ_BYTES = 1_000_000


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite(x: Any) -> float | None:
    try:
        value = float(x.detach()) if isinstance(x, torch.Tensor) else float(x)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch without weights_only
        return torch.load(path, map_location="cpu")


def _to_cpu(value: Any) -> Any:
    """Recursively move tensors to CPU so XLA checkpoints are portable."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _save_cpu(path: Path, payload: Any) -> None:
    torch.save(_to_cpu(payload), path)


def _xla_sync(device: torch.device) -> None:
    if device.type == "xla":
        import torch_xla
        torch_xla.sync()


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    state = checkpoint
    if isinstance(state, dict):
        for key in ("model", "model_state_dict", "state_dict", "model_state"):
            candidate = state.get(key)
            if isinstance(candidate, dict):
                state = candidate
                break
    if not isinstance(state, dict) or not all(torch.is_tensor(v) for v in state.values()):
        raise SystemExit("checkpoint does not contain a model state dict")
    return {
        (key[len("module."):] if key.startswith("module.") else key): value
        for key, value in state.items()
    }


def _is_data_parallel(model: torch.nn.Module) -> bool:
    return isinstance(model, torch.nn.DataParallel)


def _model_core(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if _is_data_parallel(model) else model


def _model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return _model_core(model).state_dict()


def _aux_scalar(value):
    """Reduce a per-replica auxiliary tensor (e.g. DataParallel gather)."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return value.reshape(-1).mean()
    return value


def _amp_context(enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    return torch.amp.autocast("cuda", enabled=True)


def _capture_rng(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Any, device: torch.device) -> None:
    if not isinstance(state, dict):
        return
    try:
        if state.get("python") is not None:
            random.setstate(tuple(state["python"]))
        if state.get("numpy") is not None:
            np.random.set_state(state["numpy"])
        if state.get("torch") is not None:
            torch.set_rng_state(state["torch"])
        if device.type == "cuda" and state.get("cuda"):
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as exc:  # pragma: no cover - best effort reproducibility
        print(f"[resume] could not restore RNG exactly: {exc}", flush=True)


def _was_provided(argv: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(flag + "=") for arg in argv)


# ---------------------------------------------------------------------------
# Tokenizers and data sources
# ---------------------------------------------------------------------------


class ByteTokenizer:
    """Fallback byte-level tokenizer; useful when ``tokenizers`` is absent."""

    vocab_size = 256

    def __init__(self) -> None:
        self.separator = np.frombuffer(b"\n\n", dtype=np.uint8).astype(np.int32)

    def encode(self, text: str) -> np.ndarray:
        raw = text.encode("utf-8", errors="ignore")
        return np.frombuffer(raw, dtype=np.uint8).astype(np.int32)


class BPETokenizer:
    """Thin wrapper around ``tokenizers.Tokenizer`` with a lazy import."""

    def __init__(self, path: Path) -> None:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise SystemExit(
                "BPE text mode requires the optional `tokenizers` package "
                "(`pip install tokenizers`); use --random-smoke or "
                "--tokenizer bytes for a dependency-free fallback"
            ) from exc

        if not path.is_file():
            raise SystemExit(f"BPE tokenizer file not found: {path}")
        self._tokenizer = Tokenizer.from_file(str(path))
        self.vocab_size = self._tokenizer.get_vocab_size()
        self.separator = self._separator()

    def _separator(self) -> np.ndarray:
        for name in ("<|endoftext|>", "<eos>", "</s>"):
            token_id = self._tokenizer.token_to_id(name)
            if token_id is not None and token_id >= 0:
                return np.asarray([token_id], dtype=np.int32)
        return self.encode("\n\n")

    def encode(self, text: str) -> np.ndarray:
        if not text:
            return np.empty((0,), dtype=np.int32)
        return np.asarray(self._tokenizer.encode(text).ids, dtype=np.int32)


def build_tokenizer(args: argparse.Namespace) -> ByteTokenizer | BPETokenizer:
    if args.tokenizer == "bytes":
        return ByteTokenizer()
    if not args.bpe_model:
        raise SystemExit("--bpe-model is required when --tokenizer bpe")
    return BPETokenizer(Path(args.bpe_model).expanduser())


def load_text_ids(
    paths: list[str],
    tokenizer: ByteTokenizer | BPETokenizer,
    max_bytes: int,
) -> np.ndarray:
    """Encode corpus files in bounded chunks, with separators between files."""
    parts: list[np.ndarray] = []
    total_bytes = 0
    for raw_path in paths:
        if max_bytes > 0 and total_bytes >= max_bytes:
            break
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise SystemExit(f"corpus file not found: {path}")
        if parts:
            parts.append(tokenizer.separator)

        decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")
        with path.open("rb") as handle:
            while True:
                budget = READ_BYTES
                if max_bytes > 0:
                    budget = min(budget, max_bytes - total_bytes)
                if budget <= 0:
                    break
                chunk = handle.read(budget)
                if not chunk:
                    break
                total_bytes += len(chunk)
                text = decoder.decode(chunk)
                if text:
                    ids = tokenizer.encode(text)
                    if ids.size:
                        parts.append(ids)
            tail = decoder.decode(b"", final=True)
            if tail:
                ids = tokenizer.encode(tail)
                if ids.size:
                    parts.append(ids)
        print(f"[data] read {path} ({total_bytes:,} bytes total)", flush=True)
        if max_bytes > 0 and total_bytes >= max_bytes:
            break

    if not parts:
        raise SystemExit("corpus produced no tokens")
    ids = np.concatenate(parts)
    if ids.size == 0:
        raise SystemExit("corpus produced no tokens")
    return ids.astype(np.int32, copy=False)


def load_token_bin(path: str, dtype: str) -> np.ndarray:
    """Load pre-tokenized IDs from a raw binary file."""
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise SystemExit(f"token bin not found: {file_path}")
    dtype_map = {"uint8": np.uint8, "uint16": np.uint16, "uint32": np.uint32, "int32": np.int32}
    if dtype not in dtype_map:
        raise SystemExit(f"unsupported --token-dtype: {dtype}")
    ids = np.fromfile(file_path, dtype=dtype_map[dtype])
    if ids.size == 0:
        raise SystemExit(f"token bin is empty: {file_path}")
    if np.issubdtype(ids.dtype, np.signedinteger) and int(ids.min()) < 0:
        raise SystemExit(f"token bin contains negative IDs: {file_path}")
    print(f"[data] read {file_path} ({ids.size:,} tokens, {dtype})", flush=True)
    return ids.astype(np.int32, copy=False)


def _window_batch(ids: np.ndarray, positions: np.ndarray, ctx: int) -> torch.Tensor:
    rows = [ids[int(pos) - ctx:int(pos)] for pos in positions]
    return torch.from_numpy(np.stack(rows).astype(np.int64, copy=False))


class TextBatchSource:
    """Random training windows plus a fixed, deterministic validation set."""

    def __init__(
        self,
        ids: np.ndarray,
        split: int,
        ctx: int,
        batch_size: int,
        eval_batches: int,
        seed: int,
        val_ids: np.ndarray | None = None,
    ) -> None:
        self.ids = ids
        self.split = int(split)
        self.ctx = int(ctx)
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)
        self.val_ids = val_ids

        if val_ids is None:
            val_source = ids
            val_lo = max(self.split, self.ctx)
            val_tokens = len(ids) - int(split)
        else:
            val_source = val_ids
            val_lo = self.ctx
            val_tokens = len(val_ids)
        val_hi = len(val_source)
        if val_hi <= val_lo:
            raise SystemExit("validation split is empty; lower --train-fraction")
        eval_rng = np.random.default_rng(seed + 100_003)
        positions = eval_rng.integers(
            val_lo, val_hi, size=eval_batches * batch_size
        ).astype(np.int64)
        self.eval_batches = [
            _window_batch(val_source, positions[i:i + batch_size], self.ctx)
            for i in range(0, len(positions), batch_size)
        ]
        self.info = {
            "mode": "text",
            "tokens": int(len(ids)),
            "train_tokens": int(split),
            "val_tokens": int(val_tokens),
            "vocab_size": None,
            "val_windows": int(len(positions)),
            "separate_validation": val_ids is not None,
        }

    def train_batch(self) -> torch.Tensor:
        if self.split <= self.ctx:
            raise SystemExit("training split is too small for the context window")
        positions = self.rng.integers(
            self.ctx, self.split, size=self.batch_size, dtype=np.int64
        )
        return _window_batch(self.ids, positions, self.ctx)

    def state(self) -> dict[str, Any]:
        return {"kind": "text", "rng": self.rng.bit_generator.state}

    def load_state(self, state: Any) -> None:
        if isinstance(state, dict) and state.get("kind") == "text":
            try:
                self.rng.bit_generator.state = state["rng"]
            except Exception as exc:  # pragma: no cover
                print(f"[resume] could not restore data RNG: {exc}", flush=True)


class RandomBatchSource:
    """Tiny synthetic source for --random-smoke."""

    def __init__(
        self,
        vocab_size: int,
        ctx: int,
        batch_size: int,
        eval_batches: int,
        seed: int,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.ctx = int(ctx)
        self.batch_size = int(batch_size)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        val_generator = torch.Generator(device="cpu").manual_seed(seed + 100_003)
        self.eval_batches = [
            torch.randint(
                0,
                self.vocab_size,
                (self.batch_size, self.ctx),
                generator=val_generator,
                dtype=torch.long,
            )
            for _ in range(eval_batches)
        ]
        self.info = {
            "mode": "random",
            "tokens": None,
            "train_tokens": None,
            "val_tokens": None,
            "vocab_size": self.vocab_size,
            "val_windows": int(len(self.eval_batches) * self.batch_size),
        }

    def train_batch(self) -> torch.Tensor:
        return torch.randint(
            0,
            self.vocab_size,
            (self.batch_size, self.ctx),
            generator=self.generator,
            dtype=torch.long,
        )

    def state(self) -> dict[str, Any]:
        return {"kind": "random", "generator": self.generator.get_state()}

    def load_state(self, state: Any) -> None:
        if isinstance(state, dict) and state.get("kind") == "random":
            try:
                self.generator.set_state(state["generator"])
            except Exception as exc:  # pragma: no cover
                print(f"[resume] could not restore data RNG: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Losses and evaluation
# ---------------------------------------------------------------------------


def lm_mtp_loss(
    logits: torch.Tensor,
    mtp_logits: list[torch.Tensor],
    tokens: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    mtp_weight: float = 0.3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Main next-token CE plus masked MTP CEs for t+2 ... t+k.

    Head ``i`` predicts ``tokens[t + 2 + i]``.  Its natural mask drops the
    final positions with no target, and ``valid_mask`` can additionally mark
    padded positions.  Each head is weighted by ``mtp_weight / k``.
    """
    batch, seq_len, vocab = logits.shape
    if valid_mask is None:
        valid_mask = torch.ones_like(tokens, dtype=torch.bool)
    else:
        valid_mask = valid_mask.to(device=tokens.device, dtype=torch.bool)
        if valid_mask.shape != tokens.shape:
            raise ValueError("valid_mask must have the same shape as tokens")

    zero = logits.sum() * 0.0

    main_logits = logits[:, :-1]
    main_targets = tokens[:, 1:]
    main_mask = valid_mask[:, 1:] & (main_targets != IGNORE_INDEX)
    if main_mask.any():
        main_targets = main_targets.masked_fill(~main_mask, IGNORE_INDEX)
        main_ce = F.cross_entropy(
            main_logits.reshape(-1, vocab),
            main_targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
    else:
        main_ce = zero

    head_count = len(mtp_logits)
    head_losses: list[torch.Tensor] = []
    for head_index, head_logits in enumerate(mtp_logits):
        lag = 2 + head_index
        target_len = seq_len - lag
        if target_len <= 0:
            continue
        targets = tokens[:, lag:]
        mask = valid_mask[:, :target_len] & valid_mask[:, lag:]
        mask = mask & (targets != IGNORE_INDEX)
        if not mask.any():
            continue
        targets = targets.masked_fill(~mask, IGNORE_INDEX)
        head_losses.append(
            F.cross_entropy(
                head_logits[:, :target_len].reshape(-1, vocab),
                targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        )

    if head_losses:
        stacked = torch.stack(head_losses)
        mtp_ce = stacked.mean()
        mtp_weighted = stacked.sum() * (mtp_weight / max(1, head_count))
    else:
        mtp_ce = zero
        mtp_weighted = zero

    total = main_ce + mtp_weighted
    parts = {
        "ce": main_ce.detach(),
        "mtp": mtp_ce.detach(),
        "mtp_weighted": mtp_weighted.detach(),
    }
    return total, parts


@torch.no_grad()
def evaluate(
    model: V3LM,
    batches: list[torch.Tensor],
    device: torch.device,
    amp: bool,
    mtp_weight: float,
) -> dict[str, Any]:
    """Evaluate on fixed windows; returns token-weighted metrics."""
    was_training = model.training
    model.eval()

    main_ce_sum = 0.0
    main_tokens = 0
    correct = 0
    top5 = 0
    mtp_ce_sum = 0.0
    mtp_tokens = 0
    load_balance_sum = 0.0
    router_z_sum = 0.0
    aux_batches = 0

    for xb in batches:
        xb = xb.to(device, non_blocking=True)
        with _amp_context(amp):
            logits, mtp_logits, aux = model(xb)

        vocab = logits.size(-1)
        main_logits = logits[:, :-1]
        targets = xb[:, 1:]
        valid = targets != IGNORE_INDEX
        if valid.any():
            main_ce_sum += float(
                F.cross_entropy(
                    main_logits.reshape(-1, vocab),
                    targets.reshape(-1),
                    reduction="sum",
                    ignore_index=IGNORE_INDEX,
                )
            )
            main_tokens += int(valid.sum())
            predictions = main_logits.argmax(dim=-1)
            correct += int((predictions[valid] == targets[valid]).sum())
            k = min(5, vocab)
            top = main_logits.topk(k, dim=-1).indices
            top5 += int((top[valid] == targets[valid].unsqueeze(-1)).any(-1).sum())

        for head_index, head_logits in enumerate(mtp_logits):
            lag = 2 + head_index
            target_len = xb.size(1) - lag
            if target_len <= 0:
                continue
            head_targets = xb[:, lag:]
            head_mask = head_targets != IGNORE_INDEX
            if not head_mask.any():
                continue
            mtp_ce_sum += float(
                F.cross_entropy(
                    head_logits[:, :target_len].reshape(-1, vocab),
                    head_targets.reshape(-1),
                    reduction="sum",
                    ignore_index=IGNORE_INDEX,
                )
            )
            mtp_tokens += int(head_mask.sum())

        if aux:
            if aux.get("load_balance") is not None:
                load_balance_sum += float(_aux_scalar(aux["load_balance"]).detach())
            if aux.get("router_z") is not None:
                router_z_sum += float(_aux_scalar(aux["router_z"]).detach())
            aux_batches += 1

    main_loss = main_ce_sum / max(1, main_tokens)
    mtp_loss = mtp_ce_sum / mtp_tokens if mtp_tokens else 0.0
    total_loss = main_loss + mtp_weight * mtp_loss
    metrics = {
        "loss": main_loss,
        "total_loss": total_loss,
        "mtp_loss": mtp_loss,
        "bits_per_token": main_loss / math.log(2),
        "perplexity": math.exp(main_loss) if main_loss < 50 else None,
        "accuracy": correct / max(1, main_tokens),
        "top5": top5 / max(1, main_tokens),
        "tokens": int(main_tokens),
        "batches": len(batches),
        "load_balance": (
            load_balance_sum / aux_batches if aux_batches else None
        ),
        "router_z": router_z_sum / aux_batches if aux_batches else None,
    }
    _xla_sync(next(model.parameters()).device)
    model.train(was_training)
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train V3LM with next-token CE, masked MTP auxiliary losses, "
            "MoE routing losses, EMA, AMP, and time-based cosine LR."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data")
    data.add_argument(
        "--corpus", nargs="+", default=None,
        help="one or more UTF-8 text files (not used by --random-smoke)",
    )
    data.add_argument(
        "--tokenizer", choices=("bpe", "bytes", "bin"), default="bpe",
        help="text tokenizer; 'bin' reads pre-tokenized IDs from --corpus",
    )
    data.add_argument("--bpe-model", default="", help="tokenizer.json path for --tokenizer bpe")
    data.add_argument("--token-dtype", choices=("uint8", "uint16", "uint32", "int32"),
                      default="uint16", help="dtype for --tokenizer bin")
    data.add_argument("--val-corpus", default="",
                      help="separate validation token file for --tokenizer bin")
    data.add_argument("--max-bytes", type=int, default=0, help="cap corpus bytes (0 = all)")
    data.add_argument("--train-fraction", type=float, default=0.95,
                      help="head fraction used for training; the tail is fixed validation")
    data.add_argument("--ctx", type=int, default=1024,
                      help="context length; validated presets are 512, 1024, and 8192")

    model = parser.add_argument_group("model")
    model.add_argument("--vocab-size", type=int, default=32000,
                       help="vocabulary for --random-smoke (text mode uses tokenizer vocab)")
    model.add_argument("--d-model", type=int, default=384)
    model.add_argument("--n-layers", type=int, default=12)
    model.add_argument("--n-heads", type=int, default=12)
    model.add_argument("--n-kv-heads", type=int, default=0,
                       help="KV heads for GQA; 0 = full MHA")
    model.add_argument("--attn-gate", action="store_true",
                       help="sigmoid-gate the attention output")
    model.add_argument("--swiglu", type=int, default=1536)
    model.add_argument("--window", type=int, default=512)
    model.add_argument("--global-every", type=int, default=4)
    model.add_argument("--moe-layers", type=int, default=4)
    model.add_argument("--n-experts", type=int, default=32)
    model.add_argument("--topk", type=int, default=2)
    model.add_argument("--shared-experts", type=int, default=2)
    model.add_argument("--expert-hidden", type=int, default=1536)
    model.add_argument("--moe-group-size", type=int, default=4,
                       help="experts dispatched per batched einsum group")
    model.add_argument("--moe-max-rows", type=int, default=512,
                       help="token-row chunk size for MoE peak memory")
    model.add_argument("--mtp", type=int, default=2,
                       help="MTP horizon; V3LM creates mtp-1 heads predicting t+2..t+mtp")
    model.add_argument("--mtp-adapter-rank", type=int, default=0,
                       help="low-rank MTP adapter width; 0 disables")
    model.add_argument("--mtp-block", action="store_true",
                       help="use a one-block t+1-conditioned MTP branch")
    model.add_argument("--mtp-block-experts", type=int, default=8)
    model.add_argument("--mtp-block-hidden", type=int, default=1024)
    model.add_argument("--dropout", type=float, default=0.1)
    model.add_argument("--rope-base", type=float, default=10000.0)
    model.add_argument("--max-ctx", type=int, default=8192,
                       help="maximum RoPE position cache/context supported")
    model.add_argument("--rope-scaling", choices=("auto", "none", "ntk", "yarn"),
                       default="auto",
                       help="auto uses YaRN when --ctx exceeds --rope-original-ctx")
    model.add_argument("--rope-factor", type=float, default=8.0)
    model.add_argument("--rope-original-ctx", type=int, default=1024)
    model.add_argument("--rope-beta-fast", type=float, default=32.0)
    model.add_argument("--rope-beta-slow", type=float, default=1.0)
    model.add_argument("--rope-attention-factor", type=float, default=None)

    mem = parser.add_argument_group("memory")
    mem.add_argument("--use-memory", action="store_true",
                     help="enable learned memory cross-attention")
    mem.add_argument("--no-ngram-memory", action="store_true",
                     help="disable the sparse hashed n-gram logit bias")
    mem.add_argument("--memory-slots", type=int, default=256)
    mem.add_argument("--memory-heads", type=int, default=4,
                     help="attention heads for the memory cross-attention (0 = n_heads)")
    mem.add_argument("--memory-size", type=int, default=1 << 20,
                     help="number of hashed n-gram slots")
    mem.add_argument("--memory-order", type=int, default=4)
    mem.add_argument("--memory-min-count", type=int, default=5)
    mem.add_argument("--memory-alpha-init", type=float, default=0.1)
    mem.add_argument("--memory-max-tokens", type=int, default=5_000_000,
                     help="cap training tokens used to build the n-gram store")

    train = parser.add_argument_group("training")
    train.add_argument("--steps", type=int, default=2000)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=0.05)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--warmup-frac", type=float, default=0.03)
    train.add_argument("--warmup-steps", type=int, default=0,
                       help="overrides --warmup-frac when greater than zero")
    train.add_argument("--min-lr-frac", type=float, default=0.0,
                       help="floor as a fraction of --lr (0 reaches zero)")
    train.add_argument("--lr-horizon", choices=("auto", "time", "steps", "wsd"), default="auto",
                       help="auto = step-based; time is explicit opt-in; wsd = warmup-stable-decay")
    train.add_argument("--lr-total-steps", type=int, default=0,
                       help="LR schedule denominator (0 = --steps); use for checkpoint reconstruction")
    train.add_argument("--decay-frac", type=float, default=0.1,
                       help="fraction of a WSD run used for the final decay")
    train.add_argument("--max-seconds", type=float, default=0.0,
                       help="wall-clock budget per invocation (LR horizon is --steps unless --lr-horizon time)")
    train.add_argument("--ema-decay", type=float, default=0.999)
    train.add_argument("--ema-start-frac", type=float, default=0.3)
    train.add_argument("--no-ema", action="store_true")
    train.add_argument("--mtp-weight", type=float, default=0.3,
                       help="total MTP weight; each of k heads gets weight/k")
    train.add_argument("--load-balance-coef", type=float, default=0.01)
    train.add_argument("--router-z-coef", type=float, default=0.001)

    run = parser.add_argument_group("evaluation and checkpoints")
    run.add_argument("--eval-every", type=int, default=1000)
    run.add_argument("--eval-batches", type=int, default=16,
                     help="number of fixed validation batches")
    run.add_argument("--early-stop-patience", type=int, default=0,
                     help="stop after this many steps without a best-loss improvement (0 = off)")
    run.add_argument("--early-stop-min-delta", type=float, default=0.0,
                     help="minimum loss improvement to reset early-stop patience")
    run.add_argument("--log-every", type=int, default=50)
    run.add_argument("--eval-before", action="store_true",
                     help="evaluate once after init/restore, before training")
    run.add_argument("--save-every", type=int, default=0,
                     help="write last.pt every N steps (0 = only final/early-stop)")
    run.add_argument("--freeze-trunk", action="store_true",
                     help="freeze the trunk (MTP repair)")
    run.add_argument("--unfreeze-last-block", action="store_true",
                     help="with --freeze-trunk, also train the last transformer block")
    run.add_argument("--best-every", type=int, default=0,
                     help="write best.pt every N steps (0 = only final/early-stop)")
    run.add_argument("--init-from", default="",
                     help="checkpoint weights to initialize from")
    run.add_argument("--allow-new-params", action="store_true",
                     help="allow missing keys when initializing a model with new adapter parameters")
    run.add_argument("--resume", default="", metavar="PATH",
                     help="last.pt/checkpoint to resume; 'last' means OUT/last.pt")
    run.add_argument("--out", default="runs/v3")
    run.add_argument("--seed", type=int, default=3)
    run.add_argument("--threads", type=int, default=4)

    system = parser.add_argument_group("system")
    system.add_argument("--device", default="auto",
                        help="'auto', 'cpu', 'cuda', or e.g. 'cuda:1'")
    system.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                        help="AMP on CUDA (default: enabled on CUDA; CPU always fp32)")
    system.add_argument("--data-parallel", action="store_true",
                        help="wrap the model in nn.DataParallel when multiple CUDA GPUs exist")
    system.add_argument("--amp-init-scale", type=float, default=1024.0)
    system.add_argument("--random-smoke", action="store_true",
                        help="random token IDs, no tokenizer/corpus, 3-5 CPU steps")
    return parser


SMOKE_DEFAULTS: dict[str, Any] = {
    "vocab_size": 256,
    "ctx": 32,
    "d_model": 64,
    "n_layers": 2,
    "n_heads": 4,
    "swiglu": 128,
    "window": 16,
    "global_every": 2,
    "moe_layers": 1,
    "n_experts": 4,
    "topk": 2,
    "shared_experts": 1,
    "expert_hidden": 128,
    "mtp": 3,
    "dropout": 0.0,
    "rope_base": 10000.0,
    "batch_size": 2,
    "steps": 3,
    "lr": 3e-3,
    "weight_decay": 0.0,
    "warmup_frac": 0.0,
    "min_lr_frac": 0.0,
    "lr_horizon": "steps",
    "max_seconds": 0.0,
    "ema_decay": 0.9,
    "ema_start_frac": 0.0,
    "eval_every": 1,
    "eval_batches": 1,
    "log_every": 1,
    "save_every": 1,
    "best_every": 1,
    "train_fraction": 0.9,
    "threads": 2,
    "out": "runs/v3_smoke",
}


def apply_smoke_defaults(args: argparse.Namespace, argv: list[str]) -> None:
    for key, value in SMOKE_DEFAULTS.items():
        flag = "--" + key.replace("_", "-")
        if not _was_provided(argv, flag):
            setattr(args, key, value)
    if not _was_provided(argv, "--device"):
        args.device = "cpu"
    if args.amp is None:
        args.amp = False
    args.steps = max(3, min(5, int(args.steps)))


def validate_args(args: argparse.Namespace) -> None:
    if not args.random_smoke:
        if not args.corpus:
            raise SystemExit("--corpus is required unless --random-smoke is used")
        if not 0.0 < args.train_fraction < 1.0:
            raise SystemExit("--train-fraction must be strictly between 0 and 1")
    if args.steps < 1:
        raise SystemExit("--steps must be at least 1")
    if args.ctx < 4:
        raise SystemExit("--ctx must be at least 4")
    if args.batch_size < 1 or args.eval_batches < 1:
        raise SystemExit("--batch-size and --eval-batches must be positive")
    if args.eval_every < 1 or args.log_every < 1:
        raise SystemExit("--eval-every and --log-every must be positive")
    if args.early_stop_patience < 0 or args.early_stop_min_delta < 0:
        raise SystemExit("early-stop values must be non-negative")
    if args.best_every < 0 or args.save_every < 0:
        raise SystemExit("checkpoint intervals must be non-negative")
    if args.threads < 1:
        raise SystemExit("--threads must be positive")
    if args.lr <= 0 or args.grad_clip <= 0:
        raise SystemExit("--lr and --grad-clip must be positive")
    if not 0.0 <= args.min_lr_frac <= 1.0:
        raise SystemExit("--min-lr-frac must be in [0, 1]")
    if not 0.0 < args.decay_frac <= 1.0:
        raise SystemExit("--decay-frac must be in (0, 1]")
    if args.lr_total_steps < 0:
        raise SystemExit("--lr-total-steps must be non-negative")
    if not 0.0 <= args.warmup_frac < 1.0:
        raise SystemExit("--warmup-frac must be in [0, 1)")
    if args.warmup_steps < 0:
        raise SystemExit("--warmup-steps must be non-negative")
    if not 0.0 <= args.ema_decay < 1.0:
        raise SystemExit("--ema-decay must be in [0, 1)")
    if not 0.0 <= args.ema_start_frac <= 1.0:
        raise SystemExit("--ema-start-frac must be in [0, 1]")
    if args.mtp_weight < 0 or args.load_balance_coef < 0 or args.router_z_coef < 0:
        raise SystemExit("loss coefficients must be non-negative")
    if args.max_bytes < 0 or args.max_seconds < 0:
        raise SystemExit("--max-bytes and --max-seconds must be non-negative")
    if args.resume and args.init_from:
        raise SystemExit("use either --resume or --init-from, not both")
    if args.d_model % args.n_heads != 0:
        raise SystemExit("--d-model must be divisible by --n-heads")
    if args.n_layers < 1 or args.n_heads < 1:
        raise SystemExit("--n-layers and --n-heads must be positive")
    if args.n_kv_heads < 0 or args.n_kv_heads > args.n_heads:
        raise SystemExit("--n-kv-heads must be 0 or between 1 and --n-heads")
    if args.n_kv_heads and args.n_heads % args.n_kv_heads:
        raise SystemExit("--n-heads must be divisible by --n-kv-heads")
    if not 0 <= args.moe_layers <= args.n_layers:
        raise SystemExit("--moe-layers must be between 0 and --n-layers")
    if not 1 <= args.topk <= args.n_experts:
        raise SystemExit("--topk must be between 1 and --n-experts")
    if args.n_experts < 1 or args.shared_experts < 0:
        raise SystemExit("invalid expert counts")
    if not 1 <= args.moe_group_size <= args.n_experts:
        raise SystemExit("--moe-group-size must be between 1 and --n-experts")
    if args.moe_max_rows < 1:
        raise SystemExit("--moe-max-rows must be positive")
    if args.mtp < 1:
        raise SystemExit("--mtp must be at least 1")
    if args.mtp_adapter_rank < 0:
        raise SystemExit("--mtp-adapter-rank must be non-negative")
    if args.mtp_block_experts < 1 or args.mtp_block_hidden < 1:
        raise SystemExit("--mtp-block-experts and --mtp-block-hidden must be positive")
    if args.global_every < 1 or args.window < 1:
        raise SystemExit("--global-every and --window must be positive")
    if not 0.0 <= args.dropout < 1.0:
        raise SystemExit("--dropout must be in [0, 1)")
    if args.vocab_size < 1:
        raise SystemExit("--vocab-size must be positive")
    if args.max_ctx < args.ctx:
        raise SystemExit("--max-ctx must be >= --ctx")
    if args.rope_original_ctx < 2 or args.rope_factor < 1.0:
        raise SystemExit("invalid RoPE original context or factor")
    if args.rope_beta_slow <= 0 or args.rope_beta_fast <= args.rope_beta_slow:
        raise SystemExit("RoPE beta values must satisfy 0 < beta-slow < beta-fast")
    if args.rope_attention_factor is not None and args.rope_attention_factor <= 0:
        raise SystemExit("--rope-attention-factor must be positive")
    if args.memory_heads < 0:
        raise SystemExit("--memory-heads must be non-negative")
    mem_heads = args.memory_heads or args.n_heads
    if args.d_model % mem_heads != 0:
        raise SystemExit("--d-model must be divisible by --memory-heads")
    if args.memory_slots < 1 or args.memory_size < 1:
        raise SystemExit("memory slot/size counts must be positive")
    if args.memory_order < 1 or args.memory_min_count < 0:
        raise SystemExit("invalid memory order or min-count")
    if args.memory_max_tokens < 0:
        raise SystemExit("--memory-max-tokens must be non-negative")
    if args.random_smoke and not (3 <= args.steps <= 5):
        raise SystemExit("--random-smoke runs 3-5 steps")


def make_config(args: argparse.Namespace, vocab_size: int) -> V3Config:
    return V3Config(
        vocab_size=int(vocab_size),
        ctx=int(args.ctx),
        d_model=int(args.d_model),
        n_layers=int(args.n_layers),
        n_heads=int(args.n_heads),
        n_kv_heads=int(args.n_kv_heads),
        attn_gate=bool(args.attn_gate),
        swiglu=int(args.swiglu),
        window=int(args.window),
        global_every=int(args.global_every),
        moe_layers=int(args.moe_layers),
        n_experts=int(args.n_experts),
        topk=int(args.topk),
        shared_experts=int(args.shared_experts),
        expert_hidden=int(args.expert_hidden),
        moe_group_size=int(args.moe_group_size),
        moe_max_rows=int(args.moe_max_rows),
        mtp=int(args.mtp),
        mtp_adapter_rank=int(args.mtp_adapter_rank),
        mtp_block=bool(args.mtp_block),
        mtp_block_experts=int(args.mtp_block_experts),
        mtp_block_hidden=int(args.mtp_block_hidden),
        dropout=float(args.dropout),
        rope_base=float(args.rope_base),
        max_ctx=int(args.max_ctx),
        rope_scaling=args.rope_scaling,
        rope_factor=float(args.rope_factor),
        rope_original_ctx=int(args.rope_original_ctx),
        rope_beta_fast=float(args.rope_beta_fast),
        rope_beta_slow=float(args.rope_beta_slow),
        rope_attention_factor=args.rope_attention_factor,
        use_memory=bool(args.use_memory),
        use_ngram_memory=not bool(args.no_ngram_memory),
        memory_slots=int(args.memory_slots),
        memory_heads=int(args.memory_heads),
        memory_dropout=float(args.dropout),
        memory_size=int(args.memory_size),
        memory_order=int(args.memory_order),
        memory_min_count=int(args.memory_min_count),
        memory_alpha_init=float(args.memory_alpha_init),
    )


def resolve_resume_path(value: str, out: Path) -> Path | None:
    if not value:
        return None
    if value in ("last", "auto"):
        path = out / "last.pt"
    elif value == "best":
        path = out / "best.pt"
    else:
        path = Path(value).expanduser()
    if not path.is_file():
        raise SystemExit(f"resume checkpoint not found: {path}")
    return path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def lr_scale(
    progress: float,
    warmup_frac: float,
    min_lr_frac: float,
    schedule: str = "cosine",
    decay_frac: float = 0.1,
) -> float:
    """Warmup followed by cosine or WSD decay to ``min_lr_frac``."""
    p = min(1.0, max(0.0, float(progress)))
    warmup = min(0.99, max(0.0, float(warmup_frac)))
    if warmup > 0.0 and p < warmup:
        return p / warmup
    q = 0.0 if warmup >= 1.0 else (p - warmup) / (1.0 - warmup)
    q = min(1.0, max(0.0, q))
    if schedule == "wsd":
        stable_end = 1.0 - min(0.99, max(0.0, float(decay_frac)))
        if q <= stable_end:
            return 1.0
        q = (q - stable_end) / max(1e-9, 1.0 - stable_end)
    cosine = 0.5 * (1.0 + math.cos(math.pi * q))
    return min_lr_frac + (1.0 - min_lr_frac) * cosine


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)

    if args.random_smoke:
        apply_smoke_defaults(args, raw_argv)
    validate_args(args)

    if args.random_smoke:
        if args.device != "cpu":
            print(f"[smoke] forcing CPU (requested {args.device})", flush=True)
            args.device = "cpu"
    elif args.ctx not in (512, 1024, 8192):
        print(
            f"[warn] ctx={args.ctx}; the validated long-context presets are "
            "512, 1024, and 8192",
            flush=True,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(args.threads)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "xla":
        try:
            import torch_xla.core.xla_model as xm
        except ImportError as exc:
            raise SystemExit(
                "--device xla requires torch_xla; install the TPU wheel first"
            ) from exc
        device = xm.xla_device()
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    if args.amp is None:
        amp = device.type == "cuda"
    else:
        amp = bool(args.amp) and device.type == "cuda"
        if args.amp and device.type != "cuda":
            print("[amp] requested on non-CUDA device; running fp32", flush=True)
    if args.random_smoke or device.type == "xla":
        amp = False

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    resume_path = resolve_resume_path(args.resume, out)
    resume_checkpoint: Any = None
    if resume_path is not None:
        resume_checkpoint = _torch_load(resume_path)
        saved_args = (
            resume_checkpoint.get("args", {})
            if isinstance(resume_checkpoint, dict) else {}
        )
        if not _was_provided(raw_argv, "--steps") and saved_args.get("steps"):
            args.steps = int(saved_args["steps"])
        if (
            not _was_provided(raw_argv, "--max-seconds")
            and saved_args.get("max_seconds") is not None
        ):
            args.max_seconds = float(saved_args["max_seconds"])

    # Data -----------------------------------------------------------------
    if args.random_smoke:
        source: TextBatchSource | RandomBatchSource = RandomBatchSource(
            args.vocab_size,
            args.ctx,
            args.batch_size,
            args.eval_batches,
            args.seed,
        )
        vocab_size = args.vocab_size
    elif args.tokenizer == "bin":
        if len(args.corpus) != 1:
            raise SystemExit("--tokenizer bin takes exactly one --corpus file")
        if not args.val_corpus:
            raise SystemExit("--tokenizer bin requires --val-corpus")
        token_ids = load_token_bin(args.corpus[0], args.token_dtype)
        val_ids = load_token_bin(args.val_corpus, args.token_dtype)
        vocab_size = int(args.vocab_size)
        if token_ids.size and int(token_ids.max()) >= vocab_size:
            raise SystemExit(
                f"token ID {int(token_ids.max())} >= --vocab-size {vocab_size}"
            )
        if val_ids.size and int(val_ids.max()) >= vocab_size:
            raise SystemExit(
                f"validation token ID {int(val_ids.max())} >= --vocab-size {vocab_size}"
            )
        if len(token_ids) <= args.ctx + 2:
            raise SystemExit(
                f"token file has {len(token_ids)} tokens, need more than --ctx"
            )
        split = len(token_ids)
        source = TextBatchSource(
            token_ids,
            split,
            args.ctx,
            args.batch_size,
            args.eval_batches,
            args.seed,
            val_ids=val_ids,
        )
        source.info["vocab_size"] = vocab_size
        source.info["corpus"] = [str(Path(p).expanduser()) for p in args.corpus]
        source.info["val_corpus"] = str(Path(args.val_corpus).expanduser())
        source.info["tokenizer"] = "bin"
        print(
            f"[data] tokenizer=bin vocab={vocab_size} "
            f"train={len(token_ids):,} val={len(val_ids):,} ctx={args.ctx}",
            flush=True,
        )
    else:
        tokenizer = build_tokenizer(args)
        token_ids = load_text_ids(args.corpus, tokenizer, args.max_bytes)
        vocab_size = tokenizer.vocab_size
        args.vocab_size = int(vocab_size)
        n_tokens = len(token_ids)
        if n_tokens < args.ctx + 2:
            raise SystemExit(
                f"corpus has {n_tokens} tokens, need at least --ctx + 2 "
                f"({args.ctx + 2})"
            )
        split = int(n_tokens * args.train_fraction)
        split = min(max(split, args.ctx + 1), n_tokens - 1)
        if split <= args.ctx:
            raise SystemExit("training split is too small for the requested --ctx")
        source = TextBatchSource(
            token_ids,
            split,
            args.ctx,
            args.batch_size,
            args.eval_batches,
            args.seed,
        )
        source.info["vocab_size"] = vocab_size
        source.info["corpus"] = [str(Path(p).expanduser()) for p in args.corpus]
        source.info["tokenizer"] = args.tokenizer
        print(
            f"[data] tokenizer={args.tokenizer} vocab={vocab_size} "
            f"tokens={n_tokens:,} train={split:,} val={n_tokens - split:,} "
            f"ctx={args.ctx}",
            flush=True,
        )

    cfg = make_config(args, vocab_size)
    model_config = asdict(cfg)

    if resume_checkpoint is not None:
        saved_config = (
            resume_checkpoint.get("model_config")
            if isinstance(resume_checkpoint, dict) else None
        )
        if isinstance(saved_config, dict):
            diffs = {
                key: (saved_config[key], getattr(cfg, key))
                for key in model_config
                if key in saved_config and saved_config[key] != getattr(cfg, key)
            }
            if diffs:
                raise SystemExit(
                    "resume model config mismatch "
                    f"({', '.join(sorted(diffs))}); use --init-from to load "
                    "weights into a differently shaped model"
                )

    model = V3LM(cfg).to(device)
    parameter_count = model.parameter_count()
    print(
        f"[model] params={parameter_count:,} layers={cfg.n_layers} "
        f"d_model={cfg.d_model} heads={cfg.n_heads} ctx={cfg.ctx} "
        f"mtp_heads={len(_model_core(model).mtp_heads)} moe_layers={cfg.moe_layers}",
        flush=True,
    )

    if cfg.use_memory and cfg.use_ngram_memory and hasattr(source, "ids"):
        memory_tokens = source.ids[: source.split]
        max_memory = args.memory_max_tokens or None
        print(
            f"[memory] building hashed {cfg.memory_order}-gram store "
            f"(slots={cfg.memory_size:,}, cap={max_memory or 'all'} windows)",
            flush=True,
        )
        model.build_memory(memory_tokens, max_tokens=max_memory)
        memory_stats = model.memory_path.ngram.stats()
        source.info["memory"] = memory_stats
        print(
            f"[memory] entries={memory_stats['entries']:,} "
            f"fill={memory_stats['fill_ratio']:.3f} "
            f"collisions={memory_stats['collisions']:,} "
            f"bytes={memory_stats['nbytes']:,}",
            flush=True,
        )

    if args.init_from:
        init_path = Path(args.init_from).expanduser()
        if not init_path.is_file():
            raise SystemExit(f"init checkpoint not found: {init_path}")
        init_state = _extract_state_dict(_torch_load(init_path))
        try:
            if args.allow_new_params:
                missing, unexpected = model.load_state_dict(init_state, strict=False)
                if missing:
                    print(f"[init] new parameters: {', '.join(missing)}", flush=True)
                if unexpected:
                    print(f"[init] ignored checkpoint keys: {', '.join(unexpected)}", flush=True)
            else:
                model.load_state_dict(init_state, strict=True)
        except RuntimeError as exc:
            raise SystemExit(
                f"init-from checkpoint is incompatible with this model: {exc}"
            ) from exc
        print(f"[init] loaded weights from {init_path}", flush=True)

    if args.freeze_trunk:
        last_prefix = f"blocks.{cfg.n_layers - 1}."
        for name, param in model.named_parameters():
            keep = (
                name.startswith("mtp_heads.")
                or name.startswith("mtp_adapters.")
                or name.startswith("mtp_block.")
                or name == "mtp_now"
            )
            if args.unfreeze_last_block and name.startswith(last_prefix):
                keep = True
            if not keep:
                param.requires_grad_(False)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[freeze] trunk frozen; trainable MTP params={trainable:,}", flush=True)

    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(
            f"[model] DataParallel over {torch.cuda.device_count()} GPUs",
            flush=True,
        )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=(device.type == "cuda" and not amp),
    )
    scaler = (
        torch.amp.GradScaler("cuda", init_scale=args.amp_init_scale)
        if amp else None
    )

    ema_model = None
    if not args.no_ema:
        ema_model = copy.deepcopy(model)
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)
        ema_model.eval()

    start_step = 0
    elapsed_before = 0.0
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []

    if resume_checkpoint is not None:
        resume_state = _extract_state_dict(resume_checkpoint)
        try:
            _model_core(model).load_state_dict(resume_state, strict=True)
        except RuntimeError as exc:
            raise SystemExit(
                f"resume checkpoint is incompatible with this model: {exc}"
            ) from exc
        if ema_model is not None and resume_checkpoint.get("ema") is not None:
            try:
                _model_core(ema_model).load_state_dict(
                    resume_checkpoint["ema"], strict=True
                )
            except RuntimeError as exc:
                raise SystemExit(f"resume EMA is incompatible: {exc}") from exc
        if resume_checkpoint.get("optimizer") is not None:
            try:
                optimizer.load_state_dict(resume_checkpoint["optimizer"])
            except ValueError as exc:
                raise SystemExit(f"resume optimizer is incompatible: {exc}") from exc
        if scaler is not None and resume_checkpoint.get("scaler") is not None:
            scaler.load_state_dict(resume_checkpoint["scaler"])
        start_step = int(resume_checkpoint.get("step") or 0)
        elapsed_before = float(resume_checkpoint.get("elapsed") or 0.0)
        loaded_best = resume_checkpoint.get("best")
        if isinstance(loaded_best, dict):
            best = loaded_best
        loaded_history = resume_checkpoint.get("history")
        if isinstance(loaded_history, list):
            history = loaded_history
        source.load_state(resume_checkpoint.get("data_state"))
        _restore_rng(resume_checkpoint.get("rng"), device)
        print(
            f"[resume] {resume_path} step={start_step} "
            f"history={len(history)} best={best.get('loss') if best else None}",
            flush=True,
        )
        if start_step >= args.steps:
            print(
                f"[resume] checkpoint step {start_step} >= --steps {args.steps}; "
                "only final evaluation will run",
                flush=True,
            )

    if args.warmup_steps > 0:
        warmup_steps = args.warmup_steps
    else:
        warmup_steps = int(round(args.steps * args.warmup_frac))
    warmup_steps = max(0, min(warmup_steps, max(0, args.steps - 1)))
    effective_warmup_frac = warmup_steps / max(1, args.steps)
    horizon = args.lr_horizon
    if horizon == "auto":
        # Never tie the LR schedule to a wall-clock cutoff by default: a
        # preempted or short run must resume on the same step-based curve,
        # not finish with an already-dead LR.
        horizon = "steps"

    config_doc = dict(vars(args))
    config_doc["resolved"] = {
        "vocab_size": cfg.vocab_size,
        "params": parameter_count,
        "device": str(device),
        "amp": amp,
        "mtp_heads": len(_model_core(model).mtp_heads),
        "mtp_max_lag": cfg.mtp,
        "lr_horizon": horizon,
        "warmup_steps": warmup_steps,
        "warmup_frac": effective_warmup_frac,
        "data": source.info,
    }
    _write_json(out / "config.json", config_doc)
    _write_json(out / "history.json", history)

    ema_start_step = int(args.steps * args.ema_start_frac)
    t0 = time.time() - elapsed_before
    run_start = time.time()
    total_start = time.time()
    last_step = start_step
    last_lr = 0.0
    last_improve_step = (
        int(best.get("step", start_step)) if best is not None else start_step
    )
    early_stop = False
    last_train: dict[str, Any] = {}
    status = "completed"
    steps_run = 0

    lr_total_steps = int(args.lr_total_steps or args.steps)

    def current_lr(step: int, elapsed: float) -> float:
        if horizon == "time" and args.max_seconds > 0:
            progress = elapsed / args.max_seconds
        else:
            progress = step / max(1, lr_total_steps)
        scale = lr_scale(
            progress,
            effective_warmup_frac,
            args.min_lr_frac,
            schedule=horizon,
            decay_frac=args.decay_frac,
        )
        return args.lr * scale

    def save_last(step: int, metrics: dict[str, Any] | None) -> None:
        payload = {
            "model": _model_state(model),
            "ema": _model_state(ema_model) if ema_model is not None else None,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "step": step,
            "best": best,
            "history": history,
            "elapsed": time.time() - t0,
            "data_state": source.state(),
            "rng": _capture_rng(device),
            "model_config": model_config,
            "args": vars(args),
            "metrics": metrics,
        }
        _save_cpu(out / "last.pt", payload)

    def consider_best(
        metrics: dict[str, Any] | None,
        source_name: str,
        step: int,
        force: bool = False,
    ) -> None:
        nonlocal best
        if metrics is None:
            return
        loss = metrics.get("loss")
        if loss is None or not math.isfinite(loss):
            return
        if best is not None and loss >= best.get("loss", float("inf")):
            return
        best = {
            "step": step,
            "source": source_name,
            "loss": float(loss),
            "metrics": metrics,
        }
        if not force and step != args.steps:
            if args.best_every <= 0 or step % args.best_every != 0:
                return
        checkpoint_model = ema_model if source_name == "ema" else model
        _save_cpu(
            out / "best.pt",
            {
                "model": _model_state(checkpoint_model),
                "source": source_name,
                "step": step,
                "loss": float(loss),
                "metrics": metrics,
                "model_config": model_config,
                "args": vars(args),
            },
        )
        print(
            f"[best] step {step} {source_name} loss={loss:.4f} "
            f"bits={metrics['bits_per_token']:.4f}",
            flush=True,
        )

    print(
        f"[train] device={device} amp={amp} steps={args.steps} "
        f"batch={args.batch_size} ctx={args.ctx} lr={args.lr:.2e} "
        f"horizon={horizon} warmup_steps={warmup_steps}",
        flush=True,
    )

    eval_before = None
    if args.eval_before:
        eval_before = evaluate(
            model, source.eval_batches, device, amp, args.mtp_weight
        )
        print(
            f"[eval-before] model loss={eval_before['loss']:.4f} "
            f"bits={eval_before['bits_per_token']:.4f} "
            f"top1={eval_before['accuracy'] * 100:.2f}%",
            flush=True,
        )

    for step in range(start_step, args.steps):
        elapsed = time.time() - t0
        run_elapsed = time.time() - run_start
        if args.max_seconds > 0 and run_elapsed >= args.max_seconds:
            status = "max_seconds"
            print(
                f"[stop] max-seconds reached at step {step} "
                f"({run_elapsed:.1f}s this invocation, {elapsed:.1f}s total)",
                flush=True,
            )
            break

        lr = current_lr(step, elapsed)
        for group in optimizer.param_groups:
            group["lr"] = lr

        xb = source.train_batch().to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _amp_context(amp):
            logits, mtp_logits, aux = model(xb, mtp_teacher=True)
            loss, parts = lm_mtp_loss(
                logits,
                mtp_logits,
                xb,
                mtp_weight=args.mtp_weight,
            )
            load_balance = _aux_scalar(aux.get("load_balance")) if aux else None
            router_z = _aux_scalar(aux.get("router_z")) if aux else None
            if load_balance is not None:
                loss = loss + args.load_balance_coef * load_balance
            if router_z is not None:
                loss = loss + args.router_z_coef * router_z
            loss_value = float(loss.detach())

        amp_skip = False
        if scaler is not None:
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
            amp_skip = scaler.get_scale() < scale_before
        else:
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: {loss_value}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            )
            if device.type == "xla":
                import torch_xla.core.xla_model as xm
                xm.optimizer_step(optimizer)
            else:
                optimizer.step()

        step_done = step + 1
        last_step = step_done
        steps_run += 1
        last_lr = lr

        if ema_model is not None and step_done >= ema_start_step:
            decay = args.ema_decay
            with torch.no_grad():
                for ema_parameter, parameter in zip(
                    ema_model.parameters(), model.parameters()
                ):
                    ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)
                for ema_buffer, buffer in zip(
                    ema_model.buffers(), model.buffers()
                ):
                    if (
                        ema_buffer.shape == buffer.shape
                        and ema_buffer.dtype == buffer.dtype
                    ):
                        ema_buffer.copy_(buffer)

        _xla_sync(device)

        last_train = {
            "loss": _finite(loss_value),
            "ce": _finite(parts["ce"]),
            "mtp": _finite(parts["mtp"]),
            "mtp_weighted": _finite(parts["mtp_weighted"]),
            "load_balance": _finite(load_balance) if load_balance is not None else None,
            "router_z": _finite(router_z) if router_z is not None else None,
            "grad_norm": _finite(grad_norm),
            "amp_skip": bool(amp_skip),
            "tokens": int(args.batch_size * args.ctx),
        }

        if step_done % args.log_every == 0 or step_done == args.steps:
            print(
                f"step {step_done:6d}/{args.steps} loss={loss_value:.4f} "
                f"ce={float(parts['ce']):.4f} mtp={float(parts['mtp']):.4f} "
                f"lb={float(load_balance.detach()) if load_balance is not None else 0.0:.3f} "
                f"lr={lr:.2e}",
                flush=True,
            )

        should_eval = (
            step_done % args.eval_every == 0 or step_done == args.steps
        )
        if should_eval:
            val = evaluate(
                model, source.eval_batches, device, amp, args.mtp_weight
            )
            val_ema = (
                evaluate(
                    ema_model, source.eval_batches, device, amp, args.mtp_weight
                )
                if ema_model is not None else None
            )
            elapsed_now = time.time() - t0
            history.append(
                {
                    "step": step_done,
                    "phase": "eval",
                    "elapsed": elapsed_now,
                    "lr": lr,
                    "train": last_train,
                    "val": val,
                    "val_ema": val_ema,
                }
            )
            print(
                f"  EVAL {step_done}: model loss={val['loss']:.4f} "
                f"bits={val['bits_per_token']:.4f} "
                f"top1={val['accuracy'] * 100:.2f}% "
                f"top5={val['top5'] * 100:.2f}% mtp={val['mtp_loss']:.4f}",
                flush=True,
            )
            if val_ema is not None:
                print(
                    f"        ema   loss={val_ema['loss']:.4f} "
                    f"bits={val_ema['bits_per_token']:.4f} "
                    f"top1={val_ema['accuracy'] * 100:.2f}% "
                    f"top5={val_ema['top5'] * 100:.2f}% "
                    f"mtp={val_ema['mtp_loss']:.4f}",
                    flush=True,
                )
            previous_best = best.get("loss") if best is not None else float("inf")
            consider_best(val, "model", step_done)
            if val_ema is not None:
                consider_best(val_ema, "ema", step_done)
            if best is not None and best["loss"] < previous_best - args.early_stop_min_delta:
                last_improve_step = step_done
            _write_json(out / "history.json", history)
            if args.save_every > 0 and (
                step_done % args.save_every == 0 or step_done == args.steps
            ):
                save_last(step_done, val)
            if (
                args.early_stop_patience > 0
                and step_done - last_improve_step >= args.early_stop_patience
            ):
                status = "early_stop"
                early_stop = True
                save_last(step_done, val)
                print(
                    f"[early-stop] no best improvement for "
                    f"{step_done - last_improve_step} steps "
                    f"(best={best['loss']:.4f} at step {best['step']})",
                    flush=True,
                )
        elif args.save_every > 0 and step_done % args.save_every == 0:
            save_last(step_done, None)

        if early_stop:
            break

    # Final artifacts ------------------------------------------------------
    final_metrics = evaluate(
        model, source.eval_batches, device, amp, args.mtp_weight
    )
    final_ema_metrics = (
        evaluate(ema_model, source.eval_batches, device, amp, args.mtp_weight)
        if ema_model is not None else None
    )
    consider_best(final_metrics, "model", last_step, force=True)
    if final_ema_metrics is not None:
        consider_best(final_ema_metrics, "ema", last_step, force=True)

    elapsed_total = time.time() - t0
    _save_cpu(
        out / "final.pt",
        {
            "model": _model_state(model),
            "step": last_step,
            "metrics": final_metrics,
            "model_config": model_config,
            "args": vars(args),
        },
    )
    if ema_model is not None:
        _save_cpu(
            out / "ema.pt",
            {
                "model": _model_state(ema_model),
                "source": "ema",
                "step": last_step,
                "metrics": final_ema_metrics,
                "model_config": model_config,
                "args": vars(args),
            },
        )

    if not history or history[-1].get("step") != last_step:
        history.append(
            {
                "step": last_step,
                "phase": "final",
                "elapsed": elapsed_total,
                "lr": last_lr,
                "train": last_train,
                "val": final_metrics,
                "val_ema": final_ema_metrics,
            }
        )
    _write_json(out / "history.json", history)
    save_last(last_step, final_metrics)

    artifacts = {
        "best_weights": str(out / "best.pt"),
        "final_weights": str(out / "final.pt"),
        "ema_weights": str(out / "ema.pt") if ema_model is not None else None,
        "last_checkpoint": str(out / "last.pt"),
        "result": str(out / "result.json"),
        "history": str(out / "history.json"),
        "config": str(out / "config.json"),
    }
    result = {
        "status": status,
        "random_smoke": bool(args.random_smoke),
        "params": parameter_count,
        "steps_requested": args.steps,
        "steps_completed": last_step,
        "start_step": start_step,
        "steps_this_run": steps_run,
        "tokens_seen": int(steps_run * args.batch_size * args.ctx),
        "elapsed_seconds": elapsed_total,
        "wall_seconds": time.time() - total_start,
        "device": str(device),
        "amp": amp,
        "data": source.info,
        "objective": {
            "main_ce_weight": 1.0,
            "mtp_weight": args.mtp_weight,
            "mtp_heads": len(_model_core(model).mtp_heads),
            "mtp_lags": [2 + i for i in range(len(_model_core(model).mtp_heads))],
            "load_balance_coef": args.load_balance_coef,
            "router_z_coef": args.router_z_coef,
        },
        "mtp": {
            "enabled": bool(_model_core(model).mtp_heads),
            "heads": len(_model_core(model).mtp_heads),
            "max_lag": cfg.mtp,
            "weight": args.mtp_weight,
        },
        "lr_schedule": {
            "horizon": horizon,
            "warmup_steps": warmup_steps,
            "warmup_frac": effective_warmup_frac,
            "min_lr_frac": args.min_lr_frac,
            "last_lr": last_lr,
        },
        "eval_before": eval_before,
        "best": best,
        "final": final_metrics,
        "final_ema": final_ema_metrics,
        "last_train": last_train,
        "history_entries": len(history),
        "artifacts": artifacts,
        "config": config_doc,
    }
    _write_json(out / "result.json", result)
    final_ema_loss = (
        float("nan") if final_ema_metrics is None
        else final_ema_metrics["loss"]
    )
    print(
        f"[done] status={status} steps={last_step} "
        f"final_loss={final_metrics['loss']:.4f} "
        f"final_ema_loss={final_ema_loss:.4f} "
        f"best={best['loss'] if best else None}",
        flush=True,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
