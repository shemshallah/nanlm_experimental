#!/usr/bin/env python3
"""Training harness for the z3 V3LM prototype (``z3/v3_model.py``).

The default text path is the bundled v4 64k tokenizer.  Pre-tokenized uint16
bins produced by that tokenizer are selected automatically from ``*.bin``::

    python z3/train_v3.py \
        --corpus data/v4_train.bin --val-corpus data/v4_valid.bin \
        --ctx 1024 --steps 20000 --max-seconds 1800 \
        --out runs/v3

An explicit external BPE tokenizer remains available for byte-era ablations::

    python z3/train_v3.py \
        --corpus data/a.txt data/b.txt \
        --bpe-model bpe32k.json \
        --ctx 1024 --steps 20000 --max-seconds 1800 \
        --out runs/v3

Smoke mode (random token IDs, no tokenizer or corpus, CPU, 3-5 steps)::

    python z3/train_v3.py --random-smoke --steps 3 --out /tmp/v3_smoke

The fixed 115M model (same shape on T4/A100/Blackwell)::

    python z3/train_v3.py --preset 115m \
        --corpus data/z3_domains_115m/train.bin \
        --val-corpus data/z3_domains_115m/valid.bin \
        --tokenizer bin --out runs/z3_115m

The hashed n-gram memory is opt-in (``--ngram-memory``); the learned 256-slot
memory is on by default.  See ``z3/PRESET_115M.md`` for the document-framed
corpus build and the sidecar-aware full evaluator.

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
import hashlib
import json
import math
import random
import sys
import time
from collections import OrderedDict
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
from presets import (  # noqa: E402
    PRESETS,
    expected_parameter_count,
    get_preset,
)
from text_documents import iter_documents  # noqa: E402

IGNORE_INDEX = -100
READ_BYTES = 1_000_000
V4_TOKENIZER_PATH = (
    SCRIPT_DIR / "tokenizers" / "v4_64k" / "tokenizer_complete.json"
)
V4_VOCAB_SIZE = 64060
LAYERED_TOKENIZER_PATH = (
    SCRIPT_DIR / "tokenizers" / "layered_v1" / "layered_tokenizer.json"
)


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


def _hard_negative_margin(
    logits: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Margin loss against the highest-scoring incorrect object vertex.

    The usual CE already treats every other vertex as a negative.  This term
    puts explicit pressure on the *hardest* negative in each row, which keeps
    pointer predictions from becoming merely calibrated but locally ambiguous.
    """
    rows = torch.arange(logits.size(-1), device=logits.device)
    gold = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    negatives = logits.masked_fill(
        rows.unsqueeze(0) == targets.unsqueeze(-1), float("-inf")
    )
    hardest = negatives.max(dim=-1).values
    return torch.relu(float(margin) + hardest - gold).mean()


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
        if hasattr(torch_xla, "sync"):
            torch_xla.sync()
        else:
            import torch_xla.core.xla_model as xm
            xm.mark_step()


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


# Parameters added after the first concept checkpoints.  Resume is allowed to
# leave these at their initialized values; any other mismatch is still fatal.
_RESUME_NEW_PARAMS = (
    "concept_loop.carry_gate",
    "concept_loop.object_triple_pointer.weight",
    "feedback_progress",
)
_RESUME_NEW_SUFFIXES = (
    ".router_mix",
    ".object_id_embedding.weight",
    ".object_id_to_state.weight",
    ".object_id_gate",
    ".byte_sequence_probe.weight",
    ".token_feedback_gate",
    ".route_hidden_gate",
    ".route_concept_gate",
    ".depth_carry_gate",
    ".memory_feedback_gate",
    ".mtp_carry_gate",
)


def _load_resume_state(
    module: torch.nn.Module,
    state: dict[str, torch.Tensor],
    *,
    label: str,
) -> None:
    missing, unexpected = module.load_state_dict(state, strict=False)
    allowed = set(_RESUME_NEW_PARAMS)
    bad_missing = sorted(
        key for key in set(missing) - allowed
        if not key.endswith(_RESUME_NEW_SUFFIXES)
    )
    if bad_missing or unexpected:
        parts = []
        if bad_missing:
            parts.append(f"missing={bad_missing[:8]}")
        if unexpected:
            parts.append(f"unexpected={sorted(unexpected)[:8]}")
        raise RuntimeError(f"{label}: {', '.join(parts)}")


def _aux_scalar(value):
    """Reduce a per-replica auxiliary tensor (e.g. DataParallel gather)."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return value.reshape(-1).mean()
    return value


def _tpu_environment() -> bool:
    return bool(
        os.environ.get("TPU_NAME")
        or os.environ.get("COLAB_TPU_ADDR")
        or os.environ.get("PJRT_DEVICE", "").strip().upper() == "TPU"
    )


def _cuda_amp_dtype() -> torch.dtype:
    """fp16 for T4/V100, bf16 for Ampere and newer (A100/H100/Blackwell)."""
    if not torch.cuda.is_available():
        return torch.float16
    try:
        major, _minor = torch.cuda.get_device_capability()
    except Exception:
        return torch.float16
    return torch.bfloat16 if major >= 8 else torch.float16


def _amp_context(
    enabled: bool,
    device_type: str = "cuda",
    cuda_dtype: torch.dtype = torch.float16,
):
    if not enabled:
        return contextlib.nullcontext()
    if device_type == "xla":
        # TPU uses bf16 autocast.  Prefer the torch_xla context, then the
        # newer torch.autocast("xla") spelling, then plain fp32.
        try:
            from torch_xla.amp import autocast as xla_autocast
            return xla_autocast(enabled=True, dtype=torch.bfloat16)
        except Exception:
            try:
                return torch.autocast(
                    "xla", dtype=torch.bfloat16, enabled=True
                )
            except Exception:
                return contextlib.nullcontext()
    return torch.amp.autocast("cuda", dtype=cuda_dtype, enabled=True)


def _capture_rng(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    elif device.type == "xla":
        try:
            import torch_xla.core.xla_model as xm
            state["xla"] = xm.get_rng_state()
        except Exception:
            pass
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
        elif device.type == "xla" and state.get("xla") is not None:
            try:
                import torch_xla.core.xla_model as xm
                xm.set_rng_state(state["xla"])
            except Exception:
                pass
    except Exception as exc:  # pragma: no cover - best effort reproducibility
        print(f"[resume] could not restore RNG exactly: {exc}", flush=True)


def _was_provided(argv: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(flag + "=") for arg in argv)


def apply_preset(args: argparse.Namespace, argv: list[str]) -> None:
    """Fill unset arguments from a named preset.

    Explicit CLI flags always win.  This makes a preset a reproducible
    baseline rather than a lock-in.
    """
    name = str(getattr(args, "preset", "") or "").strip()
    if not name:
        return
    try:
        spec = get_preset(name)
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc
    for key, value in spec.items():
        flag = "--" + key.replace("_", "-")
        negative = "--no-" + key.replace("_", "-")
        if not _was_provided(argv, flag) and not _was_provided(argv, negative):
            setattr(args, key, value)
    print(f"[preset] applied {name} ({len(spec)} defaults)", flush=True)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_data_manifest(args: argparse.Namespace) -> dict[str, Any] | None:
    """Verify that train/valid bins and the tokenizer match the builder."""
    raw = str(getattr(args, "data_manifest", "") or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_file():
        raise SystemExit(f"data manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "z3-document-framed-domain-bins":
        raise SystemExit(
            f"unexpected data manifest kind: {manifest.get('kind')!r}"
        )
    expected_layered = manifest.get("layered_model_sha256")
    layered_path = Path(args.layered_model).expanduser()
    if expected_layered and layered_path.is_file():
        actual_layered = _sha256_file(layered_path)
        if actual_layered != expected_layered:
            raise SystemExit(
                "tokenizer does not match the data manifest: "
                f"{layered_path} sha256={actual_layered} "
                f"expected={expected_layered}"
            )
    if args.tokenizer == "bin":
        if len(args.corpus) != 1:
            raise SystemExit(
                "--data-manifest bin mode expects exactly one --corpus train bin"
            )
        if not args.val_corpus:
            raise SystemExit(
                "--data-manifest bin mode requires --val-corpus"
            )
        checks = (
            ("train", Path(args.corpus[0])),
            ("valid", Path(args.val_corpus)),
        )
        for split, bin_path in checks:
            entry = (manifest.get("bins") or {}).get(split) or {}
            expected = entry.get("sha256")
            if not expected:
                raise SystemExit(f"manifest has no {split} bin hash")
            actual = _sha256_file(bin_path)
            if actual != expected:
                raise SystemExit(
                    f"{split} bin does not match manifest: "
                    f"{bin_path} sha256={actual} expected={expected}"
                )
        manifest_vocab = manifest.get("vocab_size")
        if manifest_vocab is not None and int(args.vocab_size) != int(manifest_vocab):
            raise SystemExit(
                f"--vocab-size {args.vocab_size} does not match manifest "
                f"vocab_size {manifest_vocab}"
            )
    print(f"[data-manifest] verified {path}", flush=True)
    return manifest


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


class LayeredFlatTokenizer:
    """Flat subword stream from the tuned layered artifact.

    The bit/byte/word/sentence layers remain available on the artifact for
    aligned auxiliary objectives; the language-model trainer consumes the
    subword/token stream through the same interface as v4.
    """

    kind = "layered"

    def __init__(self, path: str | Path = LAYERED_TOKENIZER_PATH):
        from layered_tokenizer import load_artifact

        self.path = Path(path).expanduser()
        if not self.path.is_file():
            raise SystemExit(f"layered tokenizer not found: {self.path}")
        self.layered = load_artifact(self.path)
        self.vocab_size = self.layered.vocab_size
        self.separator = (
            self.layered.eos_id
            if self.layered.eos_id is not None
            else 2
        )

    def encode(self, text: str) -> np.ndarray:
        encoded = self.layered.encode(text, add_document_frame=False)
        return np.asarray(encoded.subword.ids, dtype=np.int32)

    def encode_document(self, text: str) -> np.ndarray:
        """Encode one document with explicit ``<doc>``/``<eos>`` framing."""
        encoded = self.layered.encode(text, add_document_frame=True)
        return np.asarray(encoded.token_ids, dtype=np.int32)


def build_tokenizer(
    args: argparse.Namespace,
) -> ByteTokenizer | BPETokenizer | LayeredFlatTokenizer:
    if args.tokenizer == "bytes":
        return ByteTokenizer()
    if args.tokenizer == "v4":
        return BPETokenizer(V4_TOKENIZER_PATH)
    if args.tokenizer == "layered":
        return LayeredFlatTokenizer(args.layered_model)
    if not args.bpe_model:
        raise SystemExit("--bpe-model is required when --tokenizer bpe")
    return BPETokenizer(Path(args.bpe_model).expanduser())


def load_text_ids(
    paths: list[str],
    tokenizer: ByteTokenizer | BPETokenizer | LayeredFlatTokenizer,
    max_bytes: int,
    *,
    document_framing: bool = False,
) -> np.ndarray:
    """Encode corpus files in bounded chunks, with separators between files.

    With ``document_framing=True`` blank-line-separated documents are encoded
    one at a time through ``encode_document`` when the tokenizer supports it.
    That is the training/evaluation contract used by the concept sidecar: a
    random window never has to guess where a document began.
    """
    parts: list[np.ndarray] = []
    total_bytes = 0
    for raw_path in paths:
        if max_bytes > 0 and total_bytes >= max_bytes:
            break
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise SystemExit(f"corpus file not found: {path}")

        if document_framing:
            remaining = 0 if max_bytes <= 0 else max(0, max_bytes - total_bytes)
            for document, raw_bytes in iter_documents(
                path, max_bytes=remaining
            ):
                total_bytes += int(raw_bytes)
                encode_document = getattr(tokenizer, "encode_document", None)
                ids = (
                    encode_document(document)
                    if encode_document is not None
                    else tokenizer.encode(document)
                )
                if ids.size:
                    parts.append(np.asarray(ids, dtype=np.int32))
                if max_bytes > 0 and total_bytes >= max_bytes:
                    break
            print(
                f"[data] read {path} ({total_bytes:,} bytes total, "
                "document framing)",
                flush=True,
            )
            if max_bytes > 0 and total_bytes >= max_bytes:
                break
            continue

        if parts:
            parts.append(np.atleast_1d(np.asarray(tokenizer.separator, dtype=np.int32)))

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


def _first_byte_map(
    token_bytes: Any,
    vocab_size: int,
) -> torch.Tensor:
    """Map token IDs to their first-byte class for distillation."""
    table = torch.zeros(int(vocab_size), dtype=torch.long)
    for index, raw in enumerate(list(token_bytes)[: int(vocab_size)]):
        if raw:
            table[index] = int(raw[0])
    return table


def _set_bytes_per_token(
    source: "TextBatchSource",
    subword: Any,
    ids: np.ndarray,
) -> float | None:
    """Attach corpus mean bytes/token from the tokenizer's byte table."""
    token_bytes = getattr(subword, "token_bytes", None)
    if token_bytes is None or ids.size == 0:
        return None
    counts = np.zeros(len(token_bytes), dtype=np.int64)
    # Process in bounded slices so a 60M-token corpus does not allocate a
    # second 500MB int64 copy just to count bytes.
    for start in range(0, ids.size, 1_000_000):
        chunk = np.asarray(ids[start:start + 1_000_000], dtype=np.int64)
        if chunk.size and int(chunk.max()) >= len(token_bytes):
            return None
        counts += np.bincount(chunk, minlength=len(token_bytes))
    total_bytes = sum(
        int(counts[index]) * len(token_bytes[index])
        for index in range(len(token_bytes))
    )
    if total_bytes <= 0:
        return None
    value = total_bytes / float(ids.size)
    source.info["bytes_per_token"] = float(value)
    return float(value)


def load_text_ids_with_tasks(
    paths: list[str],
    tokenizer: ByteTokenizer | BPETokenizer | LayeredFlatTokenizer,
    max_bytes: int,
    task_ids: list[int],
    *,
    document_framing: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode each corpus file separately with a token-aligned task id."""
    if len(paths) != len(task_ids):
        raise SystemExit("--corpus-task-ids must match --corpus length")
    parts: list[np.ndarray] = []
    token_tasks: list[np.ndarray] = []
    for index, (raw_path, task_id) in enumerate(zip(paths, task_ids)):
        ids = load_text_ids(
            [raw_path],
            tokenizer,
            max_bytes,
            document_framing=document_framing,
        )
        if parts:
            separator = np.atleast_1d(
                np.asarray(tokenizer.separator, dtype=np.int32)
            ).reshape(-1)
            parts.append(separator)
            token_tasks.append(
                np.full(separator.size, int(task_id), dtype=np.int32)
            )
        parts.append(ids)
        token_tasks.append(
            np.full(ids.size, int(task_id), dtype=np.int32)
        )
    if not parts:
        raise SystemExit("domain mixture produced no tokens")
    return (
        np.concatenate(parts).astype(np.int32, copy=False),
        np.concatenate(token_tasks).astype(np.int32, copy=False),
    )


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
        sidecar_builder: Any = None,
        token_task_ids: np.ndarray | None = None,
    ) -> None:
        self.ids = ids
        self.split = int(split)
        self.ctx = int(ctx)
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)
        self.val_ids = val_ids
        self.sidecar_builder = sidecar_builder
        self.token_task_ids = (
            np.asarray(token_task_ids, dtype=np.int32)
            if token_task_ids is not None else None
        )
        if (
            self.token_task_ids is not None
            and len(self.token_task_ids) != len(ids)
        ):
            raise SystemExit("token task ids must align with token ids")
        self.last_task_ids: torch.Tensor | None = None
        self.task_segments: dict[int, list[tuple[int, int]]] = {}
        if self.token_task_ids is not None:
            self.task_segments = self._build_task_segments()
        # Random training windows rarely repeat, but fixed eval batches and
        # --overfit-single-batch do.  Cache aligned rows on CPU, not device.
        self._sidecar_cache: OrderedDict[bytes, Any] = OrderedDict()
        self.sidecar_cache_size = 64

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
        self.eval_task_ids: list[torch.Tensor] | None = None
        if self.token_task_ids is not None and val_ids is None:
            task_rows = []
            for i in range(0, len(positions), batch_size):
                chunk = positions[i:i + batch_size]
                task_rows.append(
                    torch.from_numpy(
                        self.token_task_ids[
                            np.clip(chunk - 1, 0, len(self.token_task_ids) - 1)
                        ].astype(np.int64, copy=False)
                    )
                )
            self.eval_task_ids = task_rows
        self.info = {
            "mode": "text",
            "tokens": int(len(ids)),
            "train_tokens": int(split),
            "val_tokens": int(val_tokens),
            "vocab_size": None,
            "val_windows": int(len(positions)),
            "separate_validation": val_ids is not None,
            "tasks": (
                sorted(self.task_segments)
                if self.task_segments else None
            ),
        }

    def _build_task_segments(self) -> dict[int, list[tuple[int, int]]]:
        if self.token_task_ids is None:
            return {}
        end_limit = min(self.split, len(self.token_task_ids))
        segments: dict[int, list[tuple[int, int]]] = {}
        start = 0
        for index in range(1, end_limit + 1):
            if (
                index == end_limit
                or self.token_task_ids[index] != self.token_task_ids[start]
            ):
                if index - start > self.ctx:
                    task = int(self.token_task_ids[start])
                    segments.setdefault(task, []).append((start, index))
                start = index
        return segments

    def train_batch(self) -> torch.Tensor:
        if self.split <= self.ctx:
            raise SystemExit("training split is too small for the context window")
        if self.task_segments:
            tasks = list(self.task_segments)
            weights = np.asarray(
                [
                    sum(end - start for start, end in self.task_segments[t])
                    for t in tasks
                ],
                dtype=np.float64,
            )
            weights /= weights.sum()
            task = tasks[int(self.rng.choice(len(tasks), p=weights))]
            ranges = self.task_segments[task]
            lengths = np.asarray(
                [end - start for start, end in ranges], dtype=np.float64
            )
            lengths /= lengths.sum()
            start, end = ranges[
                int(self.rng.choice(len(ranges), p=lengths))
            ]
            positions = self.rng.integers(
                start + self.ctx, end, size=self.batch_size,
                dtype=np.int64,
            )
            self.last_task_ids = torch.full(
                (self.batch_size,), task, dtype=torch.long
            )
        else:
            positions = self.rng.integers(
                self.ctx, self.split, size=self.batch_size, dtype=np.int64
            )
            self.last_task_ids = None
        return _window_batch(self.ids, positions, self.ctx)

    def build_sidecar(self, ids: torch.Tensor):
        if self.sidecar_builder is None:
            return None
        from concept_loop import SpanSidecar

        rows = []
        for row in ids:
            cpu_row = row.detach().cpu()
            key = cpu_row.numpy().tobytes()
            cached = self._sidecar_cache.get(key)
            if cached is None:
                cached = self.sidecar_builder(cpu_row.tolist())
                self._sidecar_cache[key] = cached
                if len(self._sidecar_cache) > self.sidecar_cache_size:
                    self._sidecar_cache.popitem(last=False)
            else:
                self._sidecar_cache.move_to_end(key)
            rows.append(cached)
        return SpanSidecar.collate(rows, device=ids.device)

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
    aux_logits: torch.Tensor | list[torch.Tensor] | None = None,
    aux_weight: float = 0.0,
    aux_weights: list[float] | tuple[float, ...] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Main next-token CE plus masked MTP CEs for t+2 ... t+k.

    Head ``i`` predicts ``tokens[t + 2 + i]``.  Its natural mask drops the
    final positions with no target, and ``valid_mask`` can additionally mark
    padded positions.  Each head is weighted by ``mtp_weight / k``.

    ``aux_logits`` accepts one intermediate-layer tensor or a list of tensors
    for deep supervision.  Every deep-supervision tensor predicts t+2 and is
    combined with ``aux_weights`` (normalised); this is deliberately separate
    from the per-lag MTP heads.
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

    if aux_logits is not None:
        aux_list = (
            list(aux_logits)
            if isinstance(aux_logits, (list, tuple))
            else [aux_logits]
        )
    else:
        aux_list = []
    aux_terms: list[torch.Tensor] = []
    target_len = seq_len - 2
    if aux_list and target_len > 0:
        targets = tokens[:, 2:]
        mask = valid_mask[:, :target_len] & valid_mask[:, 2:]
        targets = targets.masked_fill(~mask, IGNORE_INDEX)
        if mask.any():
            for layer_logits in aux_list:
                aux_terms.append(
                    F.cross_entropy(
                        layer_logits[:, :target_len].reshape(-1, vocab),
                        targets.reshape(-1),
                        ignore_index=IGNORE_INDEX,
                    )
                )
    if aux_terms:
        if aux_weights is not None and len(aux_weights) == len(aux_terms):
            weights = torch.as_tensor(
                aux_weights, device=aux_terms[0].device, dtype=aux_terms[0].dtype
            )
            weights = weights / weights.sum().clamp_min(1e-9)
            aux_ce = torch.stack(aux_terms).mul(weights).sum()
        else:
            aux_ce = torch.stack(aux_terms).mean()
    else:
        aux_ce = zero

    total = main_ce + mtp_weighted + aux_weight * aux_ce
    parts = {
        "ce": main_ce.detach(),
        "mtp": mtp_ce.detach(),
        "mtp_weighted": mtp_weighted.detach(),
        "mtp_aux": aux_ce.detach(),
        "mtp_aux_layers": len(aux_terms),
    }
    return total, parts


@torch.no_grad()
def evaluate(
    model: V3LM,
    batches: list[torch.Tensor],
    device: torch.device,
    amp: bool,
    mtp_weight: float,
    sidecar_builder: Any = None,
    task_id: int = -1,
    bytes_per_token: float | None = None,
    eval_task_ids: list[torch.Tensor] | None = None,
    amp_dtype: torch.dtype = torch.float16,
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
    mtp_correct_sum = 0.0
    mtp_correct_tokens = 0
    mtp_wrong_sum = 0.0
    mtp_wrong_tokens = 0
    mtp_accept = 0
    mtp_accept_tokens = 0
    diag_branch_correct = diag_base_correct = diag_gate_correct = 0.0
    diag_branch_wrong = diag_base_wrong = diag_gate_wrong = 0.0
    diag_correct_tokens = diag_wrong_tokens = 0
    load_balance_sum = 0.0
    router_z_sum = 0.0
    aux_batches = 0
    object_sums: dict[str, float] = {}
    feedback_sums: dict[str, float] = {}

    composition_sum = 0.0
    composition_tokens = 0
    mtp_composition_sum = 0.0
    mtp_composition_tokens = 0
    mtp_pointer_sum = 0.0
    mtp_pointer_tokens = 0
    target_bytes = 0
    target_chars = 0
    byte_probe_ce = 0.0
    byte_probe_tokens = 0
    byte_probe_correct = 0
    byte_probe_top5 = 0
    byte_sequence_ce = 0.0
    byte_sequence_tokens = 0
    byte_sequence_correct = 0
    byte_sequence_top5 = 0
    word_probe_ce = 0.0
    word_probe_tokens = 0
    word_probe_correct = 0
    word_probe_top5 = 0
    triple_pointer_sum = 0.0
    triple_pointer_tokens = 0
    mtp_triple_pointer_sum = 0.0
    mtp_triple_pointer_tokens = 0
    task_stats: dict[int, dict[str, float]] = {}
    for batch_index, xb in enumerate(batches):
        sidecar = (
            sidecar_builder(xb) if sidecar_builder is not None else None
        )
        if sidecar is not None:
            sidecar = sidecar.to(device)
        xb = xb.to(device, non_blocking=True)
        if task_id >= 0:
            task_ids = torch.full(
                (xb.size(0),), int(task_id), device=device,
                dtype=torch.long,
            )
        elif eval_task_ids is not None and batch_index < len(eval_task_ids):
            task_ids = eval_task_ids[batch_index].to(
                device=device, dtype=torch.long
            )
        else:
            task_ids = None
        with _amp_context(amp, device.type, amp_dtype):
            logits, mtp_logits, aux = model(
                xb, return_diagnostics=True, sidecar=sidecar,
                task_ids=task_ids, compute_mtp_aux=False,
            )
        if sidecar is not None and aux.get("composition_logits") is not None:
            comp_logits = aux["composition_logits"]
            if getattr(sidecar, "event_multi", None) is not None:
                comp_targets = sidecar.event_multi.to(comp_logits.dtype)
                composition_sum += float(
                    F.binary_cross_entropy_with_logits(
                        comp_logits, comp_targets, reduction="sum"
                    )
                )
                composition_tokens += int(comp_targets.numel())
                for lag, future_comp_logits in zip(
                    aux.get("mtp_composition_lags") or [],
                    aux.get("mtp_composition_logits") or [],
                ):
                    target_len = xb.size(1) - int(lag)
                    if target_len <= 0:
                        continue
                    future_targets = comp_targets[:, int(lag):]
                    mtp_composition_sum += float(
                        F.binary_cross_entropy_with_logits(
                            future_comp_logits[:, :target_len],
                            future_targets, reduction="sum",
                        )
                    )
                    mtp_composition_tokens += int(
                        future_targets.numel()
                    )
            else:
                comp_targets = sidecar.event_ids
                composition_sum += float(
                    F.cross_entropy(
                        comp_logits.reshape(-1, comp_logits.size(-1)),
                        comp_targets.reshape(-1).long(),
                        reduction="sum",
                    )
                )
                composition_tokens += int(comp_targets.numel())

        vocab = logits.size(-1)
        main_logits = logits[:, :-1]
        targets = xb[:, 1:]
        valid = targets != IGNORE_INDEX
        if sidecar is not None and valid.any():
            # Exact evaluation denominators: align to the same causal target
            # positions as the CE loss.
            eval_bytes = sidecar.byte_len[:, 1:]
            eval_chars = sidecar.char_len[:, 1:]
            target_bytes += int(eval_bytes[valid].sum())
            target_chars += int(eval_chars[valid].sum())
        if valid.any():
            row_ce = F.cross_entropy(
                main_logits.reshape(-1, vocab),
                targets.reshape(-1),
                reduction="none",
                ignore_index=IGNORE_INDEX,
            ).view(xb.size(0), -1)
            main_ce_sum += float(row_ce.sum())
            main_tokens += int(valid.sum())
            predictions = main_logits.argmax(dim=-1)
            hit = predictions == targets
            correct += int((hit & valid).sum())
            k = min(5, vocab)
            top = main_logits.topk(k, dim=-1).indices
            top_hits = (top == targets.unsqueeze(-1)).any(-1)
            top5 += int((top_hits & valid).sum())
            if sidecar is not None and aux.get("byte_probe_logits") is not None:
                byte_logits = aux["byte_probe_logits"][:, :-1]
                byte_targets = sidecar.first_byte[:, 1:].long()
                byte_ce = F.cross_entropy(
                    byte_logits.reshape(-1, byte_logits.size(-1)),
                    byte_targets.reshape(-1),
                    reduction="none",
                )
                byte_probe_ce += float(byte_ce.sum())
                byte_probe_tokens += int(byte_targets.numel())
                byte_probe_correct += int(
                    (byte_logits.argmax(-1) == byte_targets).sum()
                )
                byte_top = byte_logits.topk(
                    min(5, byte_logits.size(-1)), dim=-1
                ).indices
                byte_probe_top5 += int(
                    (byte_top == byte_targets.unsqueeze(-1)).any(-1).sum()
                )
                sequence_logits = aux.get("byte_sequence_logits")
                if sequence_logits is not None and sidecar.token_bytes is not None:
                    sequence_logits = sequence_logits[:, :-1]
                    sequence_targets = sidecar.token_bytes[:, 1:].long()
                    byte_mask = sequence_targets >= 0
                    if byte_mask.any():
                        seq_ce = F.cross_entropy(
                            sequence_logits.reshape(-1, 256),
                            sequence_targets.reshape(-1),
                            reduction="none",
                            ignore_index=-100,
                        ).view(sequence_targets.shape)
                        byte_sequence_ce += float(seq_ce[byte_mask].sum())
                        byte_sequence_tokens += int(byte_mask.sum())
                        byte_sequence_correct += int(
                            (
                                sequence_logits.argmax(-1)
                                == sequence_targets
                            )[byte_mask].sum()
                        )
                        seq_top = sequence_logits.topk(
                            min(5, sequence_logits.size(-1)), dim=-1
                        ).indices
                        byte_sequence_top5 += int(
                            (
                                (seq_top == sequence_targets.unsqueeze(-1))
                                .any(-1)
                            )[byte_mask].sum()
                        )
            if sidecar is not None and aux.get("word_probe_logits") is not None:
                word_logits = aux["word_probe_logits"][:, :-1]
                word_targets = sidecar.word_vocab_ids[:, 1:].long()
                word_mask = word_targets > 0
                if word_mask.any():
                    word_ce = F.cross_entropy(
                        word_logits.reshape(-1, word_logits.size(-1)),
                        word_targets.reshape(-1),
                        reduction="none",
                        ignore_index=0,
                    ).view(word_targets.shape)
                    word_probe_ce += float(word_ce[word_mask].sum())
                    word_probe_tokens += int(word_mask.sum())
                    word_probe_correct += int(
                        (word_logits.argmax(-1) == word_targets)[
                            word_mask
                        ].sum()
                    )
                    word_top = word_logits.topk(
                        min(5, word_logits.size(-1)), dim=-1
                    ).indices
                    word_probe_top5 += int(
                        (
                            (word_top == word_targets.unsqueeze(-1))
                            .any(-1)
                        )[word_mask].sum()
                    )
            if task_ids is not None:
                task_values = task_ids.detach().cpu().tolist()
                for task in sorted(set(int(value) for value in task_values)):
                    task_mask = valid & (
                        task_ids.unsqueeze(1) == task
                    )
                    tokens = int(task_mask.sum())
                    if tokens <= 0:
                        continue
                    entry = task_stats.setdefault(
                        int(task),
                        {"ce": 0.0, "tokens": 0.0,
                         "correct": 0.0, "top5": 0.0},
                    )
                    entry["ce"] += float(row_ce[task_mask].sum())
                    entry["tokens"] += tokens
                    entry["correct"] += int(
                        (hit & task_mask).sum()
                    )
                    entry["top5"] += int(
                        (top_hits & task_mask).sum()
                    )

        for head_index, head_logits in enumerate(mtp_logits):
            lag = 2 + head_index
            target_len = xb.size(1) - lag
            if target_len <= 0:
                continue
            head_targets = xb[:, lag:]
            head_mask = head_targets != IGNORE_INDEX
            if not head_mask.any():
                continue
            ce = F.cross_entropy(
                head_logits[:, :target_len].reshape(-1, vocab),
                head_targets.reshape(-1),
                reduction="none",
                ignore_index=IGNORE_INDEX,
            ).view(xb.size(0), target_len)
            mtp_ce_sum += float(ce.sum())
            mtp_tokens += int(head_mask.sum())
            if head_index == 0:
                t1_pred = main_logits[:, :target_len].argmax(dim=-1)
                t1_correct = t1_pred == xb[:, 1:1 + target_len]
                mtp_correct_sum += float(ce[t1_correct & head_mask].sum())
                mtp_wrong_sum += float(ce[~t1_correct & head_mask].sum())
                mtp_correct_tokens += int((t1_correct & head_mask).sum())
                mtp_wrong_tokens += int((~t1_correct & head_mask).sum())
                draft = head_logits[:, :target_len].argmax(dim=-1)
                main_next = main_logits[:, 1:1 + target_len].argmax(dim=-1)
                accepted = (draft == main_next) & head_mask
                mtp_accept += int(accepted.sum())
                mtp_accept_tokens += int(head_mask.sum())
                diag = aux.get("mtp_diagnostics") if aux else None
                if diag:
                    item = diag[0]
                    branch_diag = item["branch"][:, :target_len]
                    base_diag = item["base"][:, :target_len]
                    gate_diag = item["gate"][:, :target_len].squeeze(-1)
                    for start in range(0, target_len, 64):
                        end = min(target_len, start + 64)
                        tgt = head_targets[:, start:end]
                        b = branch_diag[:, start:end].reshape(-1, vocab)
                        z = base_diag[:, start:end].reshape(-1, vocab)
                        t = tgt.reshape(-1)
                        bce = F.cross_entropy(
                            b, t, reduction="none", ignore_index=IGNORE_INDEX
                        ).view(xb.size(0), end - start)
                        zce = F.cross_entropy(
                            z, t, reduction="none", ignore_index=IGNORE_INDEX
                        ).view(xb.size(0), end - start)
                        gm = gate_diag[:, start:end]
                        ok = t1_correct[:, start:end]
                        valid_m = (tgt != IGNORE_INDEX)
                        correct_m = ok & valid_m
                        wrong_m = (~ok) & valid_m
                        if correct_m.any():
                            diag_branch_correct += float(
                                bce[correct_m].sum()
                            )
                            diag_base_correct += float(zce[correct_m].sum())
                            diag_gate_correct += float(gm[correct_m].sum())
                            diag_correct_tokens += int(correct_m.sum())
                        if wrong_m.any():
                            diag_branch_wrong += float(bce[wrong_m].sum())
                            diag_base_wrong += float(zce[wrong_m].sum())
                            diag_gate_wrong += float(gm[wrong_m].sum())
                            diag_wrong_tokens += int(wrong_m.sum())

        core = _model_core(model)
        if (
            aux.get("object_triple_pointer_logits") is not None
            and getattr(core, "concept_loop", None) is not None
            and getattr(core.concept_loop, "last_triple_vertices", None)
            is not None
        ):
            triple_target = core.concept_loop.last_triple_vertices
            if triple_target.shape[2] > 0:
                target3 = triple_target[:, :, 0, :]
                ptr3 = aux["object_triple_pointer_logits"]
                triple_pointer_sum += float(
                    sum(
                        F.cross_entropy(
                            ptr3[:, :, slot].reshape(-1, ptr3.size(-1)),
                            target3[:, :, slot].long().reshape(-1),
                            reduction="sum",
                        )
                        for slot in range(3)
                    )
                )
                triple_pointer_tokens += int(
                    3 * triple_target.shape[0] * triple_target.shape[1]
                )
        if (
            aux.get("mtp_object_pointer_logits")
            and getattr(core, "concept_loop", None) is not None
            and getattr(core.concept_loop, "last_object_vertices", None)
            is not None
        ):
            target_vertices = core.concept_loop.last_object_vertices
            if target_vertices.shape[2] > 0:
                for lag, ptr in zip(
                    aux.get("mtp_object_pointer_lags") or [],
                    aux["mtp_object_pointer_logits"],
                ):
                    target_len = xb.size(1) - int(lag)
                    if target_len <= 0:
                        continue
                    future = target_vertices[:, int(lag):, 0, :]
                    mtp_pointer_sum += float(
                        F.cross_entropy(
                            ptr[:, :target_len, 0].reshape(
                                -1, ptr.size(-1)
                            ),
                            future[:, :, 0].long().reshape(-1),
                            reduction="sum",
                        )
                        + F.cross_entropy(
                            ptr[:, :target_len, 1].reshape(
                                -1, ptr.size(-1)
                            ),
                            future[:, :, 1].long().reshape(-1),
                            reduction="sum",
                        )
                    )
                    mtp_pointer_tokens += int(
                        2 * xb.size(0) * target_len
                    )
        if (
            aux.get("mtp_object_triple_pointer_logits")
            and getattr(core, "concept_loop", None) is not None
            and getattr(core.concept_loop, "last_triple_vertices", None)
            is not None
        ):
            triple_target = core.concept_loop.last_triple_vertices
            if triple_target.shape[2] > 0:
                for lag, ptr3 in zip(
                    aux.get("mtp_object_triple_pointer_lags") or [],
                    aux["mtp_object_triple_pointer_logits"],
                ):
                    target_len = xb.size(1) - int(lag)
                    if target_len <= 0:
                        continue
                    future = triple_target[:, int(lag):, 0, :]
                    mtp_triple_pointer_sum += float(
                        sum(
                            F.cross_entropy(
                                ptr3[:, :target_len, slot].reshape(
                                    -1, ptr3.size(-1)
                                ),
                                future[:, :, slot].long().reshape(-1),
                                reduction="sum",
                            )
                            for slot in range(3)
                        )
                    )
                    mtp_triple_pointer_tokens += int(
                        3 * xb.size(0) * target_len
                    )

        if aux:
            if aux.get("load_balance") is not None:
                load_balance_sum += float(_aux_scalar(aux["load_balance"]).detach())
            if aux.get("router_z") is not None:
                router_z_sum += float(_aux_scalar(aux["router_z"]).detach())
            if aux.get("object_memory"):
                for key, value in aux["object_memory"].items():
                    object_sums[key] = object_sums.get(key, 0.0) + float(
                        _aux_scalar(value).detach()
                    )
            if aux.get("feedback"):
                for key, value in aux["feedback"].items():
                    feedback_sums[key] = feedback_sums.get(key, 0.0) + float(
                        _aux_scalar(value).detach()
                    )
            aux_batches += 1

    main_loss = main_ce_sum / max(1, main_tokens)
    mtp_loss = mtp_ce_sum / mtp_tokens if mtp_tokens else 0.0
    total_loss = main_loss + mtp_weight * mtp_loss
    metrics = {
        "loss": main_loss,
        "total_loss": total_loss,
        "mtp_loss": mtp_loss,
        "mtp_correct_loss": (
            mtp_correct_sum / mtp_correct_tokens if mtp_correct_tokens else None
        ),
        "mtp_wrong_loss": (
            mtp_wrong_sum / mtp_wrong_tokens if mtp_wrong_tokens else None
        ),
        "mtp_correct_tokens": int(mtp_correct_tokens),
        "mtp_wrong_tokens": int(mtp_wrong_tokens),
        "mtp_accept": (
            mtp_accept / mtp_accept_tokens if mtp_accept_tokens else None
        ),
        "mtp_branch_correct_loss": (
            diag_branch_correct / diag_correct_tokens
            if diag_correct_tokens else None
        ),
        "mtp_base_correct_loss": (
            diag_base_correct / diag_correct_tokens
            if diag_correct_tokens else None
        ),
        "mtp_branch_wrong_loss": (
            diag_branch_wrong / diag_wrong_tokens
            if diag_wrong_tokens else None
        ),
        "mtp_base_wrong_loss": (
            diag_base_wrong / diag_wrong_tokens
            if diag_wrong_tokens else None
        ),
        "mtp_gate_correct": (
            diag_gate_correct / diag_correct_tokens
            if diag_correct_tokens else None
        ),
        "mtp_gate_wrong": (
            diag_gate_wrong / diag_wrong_tokens
            if diag_wrong_tokens else None
        ),
        "composition_loss": (
            composition_sum / composition_tokens
            if composition_tokens else None
        ),
        "mtp_composition_loss": (
            mtp_composition_sum / mtp_composition_tokens
            if mtp_composition_tokens else None
        ),
        "mtp_object_pointer_loss": (
            mtp_pointer_sum / mtp_pointer_tokens
            if mtp_pointer_tokens else None
        ),
        "object_triple_pointer_loss": (
            triple_pointer_sum / triple_pointer_tokens
            if triple_pointer_tokens else None
        ),
        "mtp_object_triple_pointer_loss": (
            mtp_triple_pointer_sum / mtp_triple_pointer_tokens
            if mtp_triple_pointer_tokens else None
        ),
        "bits_per_token": main_loss / math.log(2),
        "bits_per_byte": (
            main_ce_sum / (math.log(2) * target_bytes)
            if target_bytes > 0
            else (
                main_loss / (math.log(2) * bytes_per_token)
                if bytes_per_token and bytes_per_token > 0 else None
            )
        ),
        "bits_per_char": (
            main_ce_sum / (math.log(2) * target_chars)
            if target_chars > 0 else None
        ),
        "bytes_per_token": (
            target_bytes / main_tokens
            if target_bytes > 0 and main_tokens > 0 else bytes_per_token
        ),
        "chars_per_token": (
            target_chars / main_tokens
            if target_chars > 0 and main_tokens > 0 else None
        ),
        "perplexity": math.exp(main_loss) if main_loss < 50 else None,
        "accuracy": correct / max(1, main_tokens),
        "top5": top5 / max(1, main_tokens),
        "token_top1": correct / max(1, main_tokens),
        "token_top5": top5 / max(1, main_tokens),
        "subword_top1": correct / max(1, main_tokens),
        "subword_top5": top5 / max(1, main_tokens),
        "byte_probe_loss": (
            byte_probe_ce / byte_probe_tokens
            if byte_probe_tokens else None
        ),
        "byte_probe_accuracy": (
            byte_probe_correct / byte_probe_tokens
            if byte_probe_tokens else None
        ),
        "byte_probe_top5": (
            byte_probe_top5 / byte_probe_tokens
            if byte_probe_tokens else None
        ),
        "byte_sequence_loss": (
            byte_sequence_ce / byte_sequence_tokens
            if byte_sequence_tokens else None
        ),
        "byte_sequence_accuracy": (
            byte_sequence_correct / byte_sequence_tokens
            if byte_sequence_tokens else None
        ),
        "byte_sequence_top5": (
            byte_sequence_top5 / byte_sequence_tokens
            if byte_sequence_tokens else None
        ),
        "word_probe_loss": (
            word_probe_ce / word_probe_tokens
            if word_probe_tokens else None
        ),
        "word_probe_accuracy": (
            word_probe_correct / word_probe_tokens
            if word_probe_tokens else None
        ),
        "word_probe_top5": (
            word_probe_top5 / word_probe_tokens
            if word_probe_tokens else None
        ),
        "per_task": {
            str(task): {
                "tokens": int(entry["tokens"]),
                "loss": entry["ce"] / max(1.0, entry["tokens"]),
                "bits_per_token": (
                    entry["ce"] / max(1.0, entry["tokens"])
                ) / math.log(2),
                "bits_per_byte": (
                    (entry["ce"] / max(1.0, entry["tokens"]))
                    / (math.log(2) * bytes_per_token)
                    if bytes_per_token and bytes_per_token > 0 else None
                ),
                "accuracy": entry["correct"] / max(1.0, entry["tokens"]),
                "top5": entry["top5"] / max(1.0, entry["tokens"]),
            }
            for task, entry in sorted(task_stats.items())
        },
        "tokens": int(main_tokens),
        "batches": len(batches),
        "load_balance": (
            load_balance_sum / aux_batches if aux_batches else None
        ),
        "router_z": router_z_sum / aux_batches if aux_batches else None,
        "object_memory": (
            {key: value / aux_batches for key, value in object_sums.items()}
            if aux_batches and object_sums else None
        ),
        "feedback": (
            {key: value / aux_batches for key, value in feedback_sums.items()}
            if aux_batches and feedback_sums else None
        ),
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
        "--tokenizer",
        choices=("auto", "v4", "layered", "bpe", "bytes", "bin"),
        default="auto",
        help="tokenizer; auto=layered text or bin for *.bin",
    )
    data.add_argument("--bpe-model", default="", help="tokenizer.json path for --tokenizer bpe")
    data.add_argument(
        "--layered-model",
        default=str(LAYERED_TOKENIZER_PATH),
        help="tuned layered tokenizer artifact",
    )
    data.add_argument("--token-dtype", choices=("uint8", "uint16", "uint32", "int32"),
                      default="uint16", help="dtype for --tokenizer bin")
    data.add_argument("--val-corpus", default="",
                      help="separate validation token file for --tokenizer bin")
    data.add_argument(
        "--data-manifest",
        default="",
        help="manifest.json from build_domain_bins.py; verifies bin/tokenizer hashes",
    )
    data.add_argument("--sidecar", action=argparse.BooleanOptionalAction,
                      default=True,
                      help="build aligned word/sentence/math sidecars from token IDs")
    data.add_argument(
        "--document-framing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="encode blank-line-separated documents with <doc>/<eos> frames",
    )
    data.add_argument("--max-bytes", type=int, default=0, help="cap corpus bytes (0 = all)")
    data.add_argument(
        "--domain-mixture", action=argparse.BooleanOptionalAction,
        default=False,
        help="treat each --corpus text file as a task/domain segment",
    )
    data.add_argument(
        "--corpus-task-ids", default="",
        help="comma-separated task ids matching --corpus files",
    )
    data.add_argument("--train-fraction", type=float, default=0.95,
                      help="head fraction used for training; the tail is fixed validation")
    data.add_argument("--ctx", type=int, default=1024,
                      help="context length; validated presets are 512, 1024, and 8192")

    model = parser.add_argument_group("model")
    model.add_argument("--vocab-size", type=int, default=V4_VOCAB_SIZE,
                       help="vocabulary for bin/random modes (text uses tokenizer vocab)")
    model.add_argument("--d-model", type=int, default=384)
    model.add_argument("--n-layers", type=int, default=12)
    model.add_argument("--n-heads", type=int, default=12)
    model.add_argument("--n-kv-heads", type=int, default=4,
                       help="KV heads for GQA; 0 = full MHA")
    model.add_argument("--factorized-embedding", action=argparse.BooleanOptionalAction,
                       default=True,
                       help="low-rank token embedding for large vocabularies (default: on)")
    model.add_argument("--embedding-rank", type=int, default=128,
                       help="rank for --factorized-embedding")
    model.add_argument("--mtp-tie-heads", action=argparse.BooleanOptionalAction,
                       default=True,
                       help="tie MTP output to the main token geometry (default: on)")
    model.add_argument("--attn-gate", action=argparse.BooleanOptionalAction,
                       default=True,
                       help="sigmoid-gate the attention output (default: on)")
    model.add_argument("--force-manual-attention", action=argparse.BooleanOptionalAction,
                       default=False,
                       help="use the explicit attention fallback (debugging / CPU determinism)")
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
    model.add_argument("--moe-dense-dispatch", action=argparse.BooleanOptionalAction,
                       default=False,
                       help="fixed-shape dense dispatch (XLA); default grouped chunked")
    model.add_argument("--moe-hierarchical", action=argparse.BooleanOptionalAction,
                       default=True,
                       help="use context-conditioned domain->branch->expert routing")
    model.add_argument("--moe-branches", type=int, default=8,
                       help="branch count for hierarchical MoE (0 = auto)")
    model.add_argument("--moe-context-router", action=argparse.BooleanOptionalAction,
                       default=True,
                       help="condition routers on attention/routing context")
    model.add_argument("--moe-route-feedback", action=argparse.BooleanOptionalAction,
                       default=True,
                       help="carry attention/routing context through all blocks")
    model.add_argument("--moe-route-to-attention", action=argparse.BooleanOptionalAction,
                       default=True,
                       help="feed routing context back into each attention block")
    model.add_argument("--gate-route-attention", action=argparse.BooleanOptionalAction,
                       default=False,
                       help="gate the route->attention residual with the feedback curriculum")
    model.add_argument("--moe-hyperdimensional", action=argparse.BooleanOptionalAction,
                       default=True,
                       help="fuse HD expert/branch codebook similarities into routing")
    model.add_argument("--moe-hd-dim", type=int, default=256)
    model.add_argument("--moe-hd-scale", type=float, default=0.5)
    model.add_argument("--moe-hd-diversity-coef", type=float, default=1e-3,
                       help="orthogonality pressure on the HD codebook")
    model.add_argument("--moe-associative-memory", action=argparse.BooleanOptionalAction,
                       default=True,
                       help="add learned key/value hypervectors inside each expert")
    model.add_argument("--moe-memory-keys", type=int, default=16)
    model.add_argument("--moe-memory-dim", type=int, default=0,
                       help="hypervector width for expert memory (0 = D/H)")
    model.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction,
                       default=False,
                       help="activation checkpointing for long-context training")
    model.add_argument("--task-vocab-size", type=int, default=16,
                       help="domain/task embedding rows (0 disables)")
    model.add_argument("--concept-loop", action=argparse.BooleanOptionalAction,
                       default=True,
                       help="multi-resolution concept/object feedback through all blocks")
    model.add_argument("--compute-concept",
                       action=argparse.BooleanOptionalAction, default=True,
                       help="run the concept loop; off keeps its parameters but skips compute")
    model.add_argument("--concept-loop-scale", type=float, default=1.0,
                       help="scale of concept/object residual injection")
    model.add_argument("--concept-loop-attention",
                       action=argparse.BooleanOptionalAction, default=False,
                       help="inject concept state into attention as well as routers")
    model.add_argument("--concept-router-tokens",
                       action=argparse.BooleanOptionalAction, default=True,
                       help="give the MoE router token-aligned concept/object state")
    model.add_argument("--concept-router-global",
                       action=argparse.BooleanOptionalAction, default=True,
                       help="also add the pooled concept state to MoE routers")
    model.add_argument("--feedback-token-residual",
                       action=argparse.BooleanOptionalAction, default=True,
                       help="gate token concept/object state into the residual stream")
    model.add_argument("--feedback-route-concept",
                       action=argparse.BooleanOptionalAction, default=True,
                       help="gate concept/object/hidden summaries into the next route")
    model.add_argument("--feedback-object-carry",
                       action=argparse.BooleanOptionalAction, default=True,
                       help="carry object state across blocks")
    model.add_argument("--feedback-memory-mtp",
                       action=argparse.BooleanOptionalAction, default=True,
                       help="feed the post-trunk memory summary into the MTP context")
    model.add_argument("--feedback-mtp-carry",
                       action=argparse.BooleanOptionalAction, default=True,
                       help="feed each MTP lag's branch summary into the next lag")
    model.add_argument("--learned-router-mix",
                       action=argparse.BooleanOptionalAction, default=False,
                       help="learn positive gates for route/pooled/token router context")
    model.add_argument("--concept-vertices", type=int, default=256)
    model.add_argument("--concept-dim", type=int, default=128)
    model.add_argument("--concept-top-vertices", type=int, default=8)
    model.add_argument("--concept-top-pairs", type=int, default=4)
    model.add_argument("--concept-top-triples", type=int, default=2)
    model.add_argument("--object-gate-init", type=float, default=0.017986,
                       help="initial pair/triple object gate probability")
    model.add_argument("--subword-embedding",
                       action=argparse.BooleanOptionalAction, default=False,
                       help="feed token-aligned subword IDs into the concept encoder")
    model.add_argument("--subword-vocab-size", type=int, default=64060,
                       help="subword embedding size when --subword-embedding is on")
    model.add_argument("--object-id-embedding",
                       action=argparse.BooleanOptionalAction, default=False,
                       help="learned hard C(256,2) object-ID readout")
    model.add_argument("--object-id-dim", type=int, default=64,
                       help="embedding width for the hard pair-ID vocabulary")
    model.add_argument("--composition-head", action=argparse.BooleanOptionalAction,
                       default=True)
    model.add_argument("--byte-probe", action=argparse.BooleanOptionalAction,
                       default=False,
                       help="auxiliary 256-way next-token first-byte probe")
    model.add_argument("--byte-probe-full",
                       action=argparse.BooleanOptionalAction, default=False,
                       help="also predict every byte of the next token")
    model.add_argument("--word-probe-size", type=int, default=0,
                       help="auxiliary word-vocabulary probe rows (0 disables)")
    model.add_argument("--probe-detach", action=argparse.BooleanOptionalAction,
                       default=True,
                       help="train probes on detached hidden state")
    model.add_argument("--combinatorial-memory", action=argparse.BooleanOptionalAction,
                       default=False,
                       help="legacy post-hoc C(V,2)/C(V,3) memory")
    model.add_argument("--comb-vertices", type=int, default=64)
    model.add_argument("--comb-dim", type=int, default=128)
    model.add_argument("--comb-top-vertices", type=int, default=16)
    model.add_argument("--comb-top-pairs", type=int, default=8)
    model.add_argument("--comb-top-triples", type=int, default=8)
    model.add_argument("--mtp", type=int, default=3,
                       help="MTP horizon; V3LM creates mtp-1 heads predicting t+2..t+mtp")
    model.add_argument("--compute-mtp",
                       action=argparse.BooleanOptionalAction, default=True,
                       help="run the MTP branch/heads; off keeps their parameters but skips compute")
    model.add_argument("--mtp-adapter-rank", type=int, default=32,
                       help="low-rank MTP adapter width; 0 disables")
    model.add_argument("--mtp-output-rank", type=int, default=0,
                       help="low-rank MTP output width; 0 reuses embedding_rank")
    model.add_argument("--mtp-block", action=argparse.BooleanOptionalAction,
                       default=True,
                       help="use a per-lag t+1-conditioned MTP branch (default: on)")
    model.add_argument("--mtp-block-experts", type=int, default=8)
    model.add_argument("--mtp-block-hidden", type=int, default=1024)
    model.add_argument("--mtp-attention-global", action=argparse.BooleanOptionalAction,
                       default=True,
                       help="use full causal attention in the MTP branch")
    model.add_argument("--mtp-attention-gate", action=argparse.BooleanOptionalAction,
                       default=False,
                       help="sigmoid-gate the MTP branch attention output")
    model.add_argument("--mtp-teacher-prob", type=float, default=0.75,
                       help="probability of using the soft ground-truth t+1 teacher")
    model.add_argument("--mtp-softmax-temperature", type=float, default=1.0,
                       help="temperature for the predicted t+1 conditioning distribution")
    model.add_argument("--mtp-aux-layer", type=int, default=-1,
                       help="legacy extra intermediate layer for deep-supervision MTP loss (-1 off)")
    model.add_argument("--mtp-aux-layers", default="3,7",
                       help="comma-separated intermediate layers for deep supervision")
    model.add_argument("--mtp-aux-layer-weights", default="",
                       help="optional comma-separated deep-supervision weights")
    model.add_argument("--mtp-aux-weight", type=float, default=0.2)
    model.add_argument("--mtp-position-embed", action=argparse.BooleanOptionalAction,
                       default=True,
                       help="add a learned per-position anchor to each MTP lag")
    model.add_argument("--mtp-hd-time", action=argparse.BooleanOptionalAction,
                       default=True,
                       help="use fixed HD <now> + permutation future rails")
    model.add_argument("--mtp-branch-norm", action=argparse.BooleanOptionalAction,
                       default=True,
                       help="RMS-normalize the MTP branch before its output")
    model.add_argument("--mtp-gate-prior", type=float, default=0.7,
                       help="initial confidence-gate reliability prior")
    model.add_argument("--mtp-gate", action=argparse.BooleanOptionalAction,
                       default=True,
                       help="blend MTP branch logits with the fallback logits")
    model.add_argument("--mtp-gate-hidden", type=int, default=256,
                       help="hidden width of the MTP confidence gate")
    model.add_argument("--mtp-gate-use-hidden", action=argparse.BooleanOptionalAction,
                       default=True,
                       help="feed the trunk hidden state into the MTP gate")
    model.add_argument("--dropout", type=float, default=0.1)
    model.add_argument("--rope-base", type=float, default=10000.0)
    model.add_argument("--max-ctx", type=int, default=8192,
                       help="maximum RoPE position cache/context supported")
    model.add_argument("--long-context",
                       action=argparse.BooleanOptionalAction, default=False,
                       help="force 8k RoPE headroom; checkpoint ctx>=1024")
    model.add_argument("--rope-scaling", choices=("auto", "none", "ntk", "yarn"),
                       default="auto",
                       help="auto uses YaRN when --ctx exceeds --rope-original-ctx")
    model.add_argument("--rope-factor", type=float, default=8.0)
    model.add_argument("--rope-original-ctx", type=int, default=1024)
    model.add_argument("--rope-beta-fast", type=float, default=32.0)
    model.add_argument("--rope-beta-slow", type=float, default=1.0)
    model.add_argument("--rope-attention-factor", type=float, default=None)

    mem = parser.add_argument_group("memory")
    mem.add_argument("--use-memory", action=argparse.BooleanOptionalAction,
                     default=True,
                     help="enable learned memory cross-attention (default: on)")
    mem.add_argument("--ngram-memory", action=argparse.BooleanOptionalAction,
                     default=False,
                     help="enable the sparse hashed n-gram logit bias (default: off)")
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
    train.add_argument("--mtp-lr-scale", type=float, default=0.5,
                       help="MTP branch learning-rate multiplier")
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
    train.add_argument("--composition-weight", type=float, default=0.05,
                       help="auxiliary composition-event CE weight")
    train.add_argument("--byte-probe-weight", type=float, default=0.02,
                       help="auxiliary next-token first-byte probe weight")
    train.add_argument("--byte-sequence-weight", type=float, default=0.02,
                       help="auxiliary full next-token byte-sequence probe weight")
    train.add_argument("--cross-resolution-weight", type=float, default=0.0,
                       help="token-to-byte cross-resolution distillation weight")
    train.add_argument("--cross-resolution-temperature", type=float, default=1.0,
                       help="temperature for token/byte distribution consistency")
    train.add_argument("--word-probe-weight", type=float, default=0.02,
                       help="auxiliary next-token word-identity probe weight")
    train.add_argument("--mtp-composition-weight", type=float, default=0.02,
                       help="future MTP structural-event BCE weight")
    train.add_argument("--object-pointer-weight", type=float, default=0.02,
                       help="auxiliary C(256,2) object-pointer CE weight")
    train.add_argument("--object-negative-weight", type=float, default=0.0,
                       help="hardest-negative margin weight for object pointers")
    train.add_argument("--object-balance-coef", type=float, default=0.0,
                       help="Switch-style balance over the 256 object vertices")
    train.add_argument("--object-negative-margin", type=float, default=1.0,
                       help="margin for the hardest-negative object pointer loss")
    train.add_argument("--object-triple-pointer-weight", type=float, default=0.01,
                       help="auxiliary C(256,3) object-triple-pointer CE weight")
    train.add_argument("--mtp-object-pointer-weight", type=float, default=0.01,
                       help="future MTP object-pointer CE weight")
    train.add_argument("--mtp-object-triple-pointer-weight", type=float, default=0.005,
                       help="future MTP object-triple-pointer CE weight")
    train.add_argument("--task-id", type=int, default=-1,
                       help="domain/task id broadcast to every batch (-1 off)")
    train.add_argument("--mtp-weight", type=float, default=0.3,
                       help="total MTP weight; each of k heads gets weight/k")
    train.add_argument("--mtp-gt-mix-start", type=float, default=0.85,
                       help="starting ground-truth mix for soft MTP conditioning")
    train.add_argument("--mtp-gt-mix-end", type=float, default=0.25,
                       help="final ground-truth mix; predicted distribution always remains")
    train.add_argument("--mtp-gt-mix-steps", type=int, default=0,
                       help="steps over which to decay the ground-truth mix")
    train.add_argument(
        "--feedback-gate-warmup-steps",
        type=int,
        default=0,
        help="linearly open the feedback bus over this many steps (0 = full)",
    )
    train.add_argument("--mtp-gate-coef", type=float, default=0.1,
                       help="supervision coefficient for the MTP confidence gate")
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
    run.add_argument("--overfit-single-batch", action="store_true",
                     help="reuse one training batch for diagnostics")
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
                        help="'auto', 'cpu', 'cuda', 'cuda:1', 'xla', or 'tpu'")
    system.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                        help="bf16 autocast on XLA/TPU; fp16/bf16 on CUDA; CPU fp32")
    system.add_argument("--amp-dtype", choices=("auto", "fp16", "bf16"), default="auto",
                        help="CUDA autocast dtype; auto uses bf16 on Ampere+ and fp16 on T4")
    system.add_argument("--data-parallel", action="store_true",
                        help="wrap the model in nn.DataParallel when multiple CUDA GPUs exist")
    system.add_argument("--amp-init-scale", type=float, default=1024.0)
    system.add_argument(
        "--preset",
        default="",
        choices=("", *sorted(PRESETS)),
        help="reproducible argument bundle; explicit flags still win",
    )
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
    if args.object_balance_coef < 0:
        raise SystemExit("--object-balance-coef must be non-negative")
    if args.mtp_lr_scale < 0:
        raise SystemExit("--mtp-lr-scale must be non-negative")
    if args.max_bytes < 0 or args.max_seconds < 0:
        raise SystemExit("--max-bytes and --max-seconds must be non-negative")
    if args.resume and args.init_from:
        raise SystemExit("use either --resume or --init-from, not both")
    if args.d_model % args.n_heads != 0:
        raise SystemExit("--d-model must be divisible by --n-heads")
    if args.factorized_embedding and args.embedding_rank < 1:
        raise SystemExit("--embedding-rank must be positive")
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
    if args.moe_hyperdimensional and args.moe_hd_dim < 8:
        raise SystemExit("--moe-hd-dim must be at least 8")
    if args.moe_hd_scale < 0:
        raise SystemExit("--moe-hd-scale must be non-negative")
    if args.moe_associative_memory and args.moe_memory_keys < 1:
        raise SystemExit("--moe-memory-keys must be positive")
    if args.moe_memory_dim < 0:
        raise SystemExit("--moe-memory-dim must be non-negative")
    if args.concept_loop:
        if not 0.0 <= args.concept_loop_scale <= 2.0:
            raise SystemExit("--concept-loop-scale must be in [0, 2]")
        if args.concept_vertices < 2:
            raise SystemExit("--concept-vertices must be at least 2")
        if args.concept_dim < 8:
            raise SystemExit("--concept-dim must be at least 8")
        if not 2 <= args.concept_top_vertices <= args.concept_vertices:
            raise SystemExit("--concept-top-vertices must be in [2, concept-vertices]")
        if args.concept_top_pairs < 1 or args.concept_top_triples < 0:
            raise SystemExit("invalid concept object top-k")
        if args.subword_vocab_size < 1:
            raise SystemExit("--subword-vocab-size must be positive")
    if args.combinatorial_memory:
        if args.comb_vertices < 2:
            raise SystemExit("--comb-vertices must be at least 2")
        if args.comb_dim < 8:
            raise SystemExit("--comb-dim must be at least 8")
        if not 2 <= args.comb_top_vertices <= args.comb_vertices:
            raise SystemExit("--comb-top-vertices must be in [2, comb-vertices]")
        if args.comb_top_pairs < 1:
            raise SystemExit("--comb-top-pairs must be positive")
        if args.comb_top_triples < 0:
            raise SystemExit("--comb-top-triples must be non-negative")
    if args.mtp < 1:
        raise SystemExit("--mtp must be at least 1")
    if args.mtp_adapter_rank < 0:
        raise SystemExit("--mtp-adapter-rank must be non-negative")
    if args.mtp_output_rank < 0:
        raise SystemExit("--mtp-output-rank must be non-negative")
    if not 0.0 < args.object_gate_init < 1.0:
        raise SystemExit("--object-gate-init must be in (0, 1)")
    if args.mtp_block_experts < 1 or args.mtp_block_hidden < 1:
        raise SystemExit("--mtp-block-experts and --mtp-block-hidden must be positive")
    if args.mtp_gate_hidden < 1:
        raise SystemExit("--mtp-gate-hidden must be positive")
    if not 0.0 <= args.mtp_teacher_prob <= 1.0:
        raise SystemExit("--mtp-teacher-prob must be in [0, 1]")
    if args.mtp_aux_layer >= args.n_layers:
        raise SystemExit("--mtp-aux-layer must be < --n-layers")
    if args.mtp_aux_weight < 0:
        raise SystemExit("--mtp-aux-weight must be non-negative")
    if not 0.0 <= args.mtp_gt_mix_end <= args.mtp_gt_mix_start <= 1.0:
        raise SystemExit("MTP gt-mix must satisfy 0 <= end <= start <= 1")
    if args.mtp_gt_mix_steps < 0:
        raise SystemExit("--mtp-gt-mix-steps must be non-negative")
    if args.feedback_gate_warmup_steps < 0:
        raise SystemExit("--feedback-gate-warmup-steps must be non-negative")
    if args.mtp_gate_coef < 0:
        raise SystemExit("--mtp-gate-coef must be non-negative")
    if args.global_every < 1 or args.window < 1:
        raise SystemExit("--global-every and --window must be positive")
    if not 0.0 <= args.dropout < 1.0:
        raise SystemExit("--dropout must be in [0, 1)")
    if args.vocab_size < 1:
        raise SystemExit("--vocab-size must be positive")
    if args.task_vocab_size < 0:
        raise SystemExit("--task-vocab-size must be non-negative")
    if (
        args.task_id >= 0
        and (args.task_vocab_size <= 0 or args.task_id >= args.task_vocab_size)
    ):
        raise SystemExit("--task-id must be in [0, --task-vocab-size)")
    if args.long_context:
        args.max_ctx = max(int(args.max_ctx), 8192)
        args.rope_scaling = "auto"
        if args.ctx >= 1024:
            args.grad_checkpoint = True
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
        factorized_embedding=bool(args.factorized_embedding),
        embedding_rank=int(args.embedding_rank),
        mtp_tie_heads=bool(args.mtp_tie_heads),
        attn_gate=bool(args.attn_gate),
        force_manual_attention=bool(args.force_manual_attention),
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
        moe_dense_dispatch=bool(args.moe_dense_dispatch),
        moe_hierarchical=bool(args.moe_hierarchical),
        moe_branches=int(args.moe_branches),
        moe_context_router=bool(args.moe_context_router),
        moe_route_feedback=bool(args.moe_route_feedback),
        moe_route_to_attention=bool(args.moe_route_to_attention),
        gate_route_attention=bool(args.gate_route_attention),
        moe_hyperdimensional=bool(args.moe_hyperdimensional),
        moe_hd_dim=int(args.moe_hd_dim),
        moe_hd_scale=float(args.moe_hd_scale),
        moe_associative_memory=bool(args.moe_associative_memory),
        moe_memory_keys=int(args.moe_memory_keys),
        moe_memory_dim=int(args.moe_memory_dim),
        grad_checkpoint=bool(args.grad_checkpoint),
        use_concept_loop=bool(args.concept_loop),
        compute_concept=bool(args.compute_concept),
        task_vocab_size=int(args.task_vocab_size),
        concept_loop_scale=float(args.concept_loop_scale),
        concept_loop_attention=bool(args.concept_loop_attention),
        concept_router_tokens=bool(args.concept_router_tokens),
        concept_router_global=bool(args.concept_router_global),
        feedback_token_residual=bool(args.feedback_token_residual),
        feedback_route_concept=bool(args.feedback_route_concept),
        feedback_object_carry=bool(args.feedback_object_carry),
        feedback_memory_mtp=bool(args.feedback_memory_mtp),
        feedback_mtp_carry=bool(args.feedback_mtp_carry),
        learned_router_mix=bool(args.learned_router_mix),
        concept_loop_vertices=int(args.concept_vertices),
        concept_loop_dim=int(args.concept_dim),
        concept_top_vertices=int(args.concept_top_vertices),
        concept_top_pairs=int(args.concept_top_pairs),
        concept_top_triples=int(args.concept_top_triples),
        object_gate_init=float(args.object_gate_init),
        subword_embedding=bool(args.subword_embedding),
        subword_vocab_size=int(args.subword_vocab_size),
        object_id_embedding=bool(args.object_id_embedding),
        object_id_dim=int(args.object_id_dim),
        composition_head=bool(args.composition_head),
        byte_probe=bool(args.byte_probe),
        byte_probe_full=bool(args.byte_probe_full),
        word_probe_size=int(args.word_probe_size),
        probe_detach=bool(args.probe_detach),
        use_combinatorial_memory=bool(args.combinatorial_memory),
        comb_vertices=int(args.comb_vertices),
        comb_dim=int(args.comb_dim),
        comb_top_vertices=int(args.comb_top_vertices),
        comb_top_pairs=int(args.comb_top_pairs),
        comb_top_triples=int(args.comb_top_triples),
        mtp=int(args.mtp),
        compute_mtp=(
            bool(args.compute_mtp) and float(args.mtp_weight) > 0.0
        ),
        mtp_adapter_rank=int(args.mtp_adapter_rank),
        mtp_output_rank=int(args.mtp_output_rank),
        mtp_block=bool(args.mtp_block),
        mtp_block_experts=int(args.mtp_block_experts),
        mtp_block_hidden=int(args.mtp_block_hidden),
        mtp_attention_global=bool(args.mtp_attention_global),
        mtp_attention_gate=bool(args.mtp_attention_gate),
        mtp_teacher_prob=float(args.mtp_teacher_prob),
        mtp_softmax_temperature=float(args.mtp_softmax_temperature),
        mtp_aux_layer=int(args.mtp_aux_layer),
        mtp_position_embed=bool(args.mtp_position_embed),
        mtp_hd_time=bool(args.mtp_hd_time),
        mtp_branch_norm=bool(args.mtp_branch_norm),
        mtp_gate_prior=float(args.mtp_gate_prior),
        mtp_gate=bool(args.mtp_gate),
        mtp_gate_hidden=int(args.mtp_gate_hidden),
        mtp_gate_use_hidden=bool(args.mtp_gate_use_hidden),
        mtp_aux_layers=tuple(
            int(x) for x in str(args.mtp_aux_layers).split(",") if x.strip()
        ),
        mtp_aux_layer_weights=(
            tuple(
                float(x)
                for x in str(args.mtp_aux_layer_weights).split(",")
                if x.strip()
            )
            or None
        ),
        mtp_aux_weight=float(args.mtp_aux_weight),
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
        use_ngram_memory=bool(
            getattr(args, "ngram_memory", False)
            and not getattr(args, "no_ngram_memory", False)
        ),
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
        if args.preset:
            print(
                f"[preset] ignoring {args.preset} for --random-smoke",
                flush=True,
            )
            args.preset = ""
        apply_smoke_defaults(args, raw_argv)
    else:
        apply_preset(args, raw_argv)
    validate_args(args)
    data_manifest = (
        None if args.random_smoke else verify_data_manifest(args)
    )

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
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif _tpu_environment():
            try:
                import torch_xla.core.xla_model as xm
            except ImportError as exc:
                raise SystemExit(
                    "TPU environment detected but torch_xla is missing"
                ) from exc
            device = xm.xla_device()
        else:
            device = torch.device("cpu")
    elif args.device in ("xla", "tpu"):
        try:
            import torch_xla.core.xla_model as xm
        except ImportError as exc:
            raise SystemExit(
                "--device xla/tpu requires torch_xla; install the TPU wheel first"
            ) from exc
        device = xm.xla_device()
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    if device.type == "cuda":
        index = device.index if device.index is not None else 0
        major, minor = torch.cuda.get_device_capability(index)
        print(
            f"[cuda] {torch.cuda.get_device_name(index)} "
            f"capability={major}.{minor}",
            flush=True,
        )
        if major >= 8:
            # bf16 autocast covers the matmuls; TF32 keeps any residual fp32
            # work on tensor cores instead of falling back to fp32 SIMT.
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            print("[cuda] tf32 enabled", flush=True)
    if args.amp is None:
        amp = device.type in ("cuda", "xla")
    else:
        amp = bool(args.amp) and device.type in ("cuda", "xla")
        if args.amp and device.type not in ("cuda", "xla"):
            print(
                "[amp] requested on a non-accelerator device; running fp32",
                flush=True,
            )
    if args.random_smoke:
        amp = False
    if args.amp_dtype == "auto":
        cuda_amp_dtype = _cuda_amp_dtype()
    elif args.amp_dtype == "bf16":
        cuda_amp_dtype = torch.bfloat16
    else:
        cuda_amp_dtype = torch.float16
    print(
        f"[amp] cuda_dtype={cuda_amp_dtype} scaler="
        f"{amp and device.type == 'cuda' and cuda_amp_dtype == torch.float16}",
        flush=True,
    )

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
    first_byte_map_cpu: torch.Tensor | None = None
    if not args.random_smoke and args.tokenizer == "auto":
        paths = [Path(p).expanduser() for p in args.corpus]
        args.tokenizer = (
            "bin"
            if len(paths) == 1 and paths[0].suffix.lower() == ".bin"
            else "layered"
        )
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
        if args.domain_mixture:
            raise SystemExit(
                "--domain-mixture is for text corpora; a token bin has "
                "no document task labels"
            )
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
        # Bin training still needs bytes/token for the frontier metric even
        # when the concept sidecar is disabled (e.g. the no-concept ablation).
        from layered_tokenizer import load_artifact
        bin_tokenizer = load_artifact(Path(args.layered_model).expanduser())
        _set_bytes_per_token(source, bin_tokenizer.subword, source.ids)
        first_byte_map_cpu = _first_byte_map(
            bin_tokenizer.subword.token_bytes, vocab_size
        )
        print(
            f"[data] tokenizer=bin vocab={vocab_size} "
            f"train={len(token_ids):,} val={len(val_ids):,} ctx={args.ctx}",
            flush=True,
        )
    else:
        tokenizer = build_tokenizer(args)
        subword = getattr(tokenizer, "subword", None)
        if subword is not None and hasattr(subword, "token_bytes"):
            first_byte_map_cpu = _first_byte_map(
                subword.token_bytes,
                int(getattr(tokenizer, "vocab_size", args.vocab_size)),
            )
        token_task_ids = None
        if args.domain_mixture:
            if len(args.corpus) < 1:
                raise SystemExit("--domain-mixture needs --corpus files")
            if args.corpus_task_ids.strip():
                task_values = [
                    int(value) for value in args.corpus_task_ids.split(",")
                    if value.strip()
                ]
            else:
                task_values = list(range(len(args.corpus)))
            if len(task_values) != len(args.corpus):
                raise SystemExit(
                    "--corpus-task-ids must match --corpus length"
                )
            if any(
                task < 0 or task >= args.task_vocab_size
                for task in task_values
            ):
                raise SystemExit(
                    "corpus task ids must be in [0, --task-vocab-size)"
                )
            token_ids, token_task_ids = load_text_ids_with_tasks(
                args.corpus,
                tokenizer,
                args.max_bytes,
                task_values,
                document_framing=bool(args.document_framing),
            )
            print(
                f"[data] domain mixture tasks={task_values}", flush=True
            )
            if not args.val_corpus:
                print(
                    "[warn] --domain-mixture without --val-corpus uses the "
                    "tail of the concatenated stream as validation; pass a "
                    "document-disjoint --val-corpus for a clean test split",
                    flush=True,
                )
        else:
            token_ids = load_text_ids(
                args.corpus,
                tokenizer,
                args.max_bytes,
                document_framing=bool(args.document_framing),
            )
        vocab_size = tokenizer.vocab_size
        args.vocab_size = int(vocab_size)
        n_tokens = len(token_ids)
        if n_tokens < args.ctx + 2:
            raise SystemExit(
                f"corpus has {n_tokens} tokens, need at least --ctx + 2 "
                f"({args.ctx + 2})"
            )
        val_ids = None
        if args.val_corpus:
            val_ids = load_text_ids(
                [args.val_corpus],
                tokenizer,
                args.max_bytes,
                document_framing=bool(args.document_framing),
            )
            if len(val_ids) < args.ctx + 2:
                raise SystemExit(
                    f"validation corpus has {len(val_ids)} tokens, need at "
                    f"least --ctx + 2 ({args.ctx + 2})"
                )
            print(
                f"[data] separate validation corpus={args.val_corpus} "
                f"tokens={len(val_ids):,}",
                flush=True,
            )
        if val_ids is not None:
            # The explicit validation file is the complete held-out stream;
            # do not also reserve a tail slice from the training corpus.
            split = len(token_ids)
        else:
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
            val_ids=val_ids,
            token_task_ids=token_task_ids,
        )
        source.info["vocab_size"] = vocab_size
        source.info["corpus"] = [str(Path(p).expanduser()) for p in args.corpus]
        source.info["tokenizer"] = args.tokenizer
        source.info["domain_mixture"] = bool(args.domain_mixture)
        source.info["corpus_task_ids"] = (
            task_values if args.domain_mixture else None
        )
        source.info["val_corpus"] = (
            str(Path(args.val_corpus).expanduser())
            if args.val_corpus else None
        )
        source.info["document_framing"] = bool(args.document_framing)
        # Bytes/token is independent of the concept sidecar.  Compute it for
        # every layered text run so the no-concept ablation still reports BPB.
        layered = getattr(tokenizer, "layered", None)
        subword = getattr(layered, "subword", None)
        if subword is not None:
            _set_bytes_per_token(source, subword, source.ids)
        elif args.tokenizer == "bytes":
            source.info["bytes_per_token"] = 1.0
        print(
            f"[data] tokenizer={args.tokenizer} vocab={vocab_size} "
            f"tokens={n_tokens:,} train={split:,} "
            f"val={len(val_ids) if val_ids is not None else n_tokens - split:,} "
            f"ctx={args.ctx}",
            flush=True,
        )

    if data_manifest is not None:
        source.info["data_manifest"] = str(
            Path(args.data_manifest).expanduser().resolve()
        )

    if args.sidecar and args.tokenizer in (
        "bin", "v4", "layered",
    ):
        from concept_loop import SpanSidecar
        from layered_tokenizer import load_artifact

        side_tok = load_artifact(Path(args.layered_model).expanduser())
        source.sidecar_builder = (
            lambda ids: SpanSidecar.from_token_ids(ids, side_tok)
        )
        bytes_per_token = _set_bytes_per_token(
            source, side_tok.subword, source.ids
        )
        print(
            "[sidecar] aligned word/sentence/math feed enabled"
            + (
                f" bytes/token={bytes_per_token:.4f}"
                if bytes_per_token is not None else ""
            ),
            flush=True,
        )

    overfit_batch = source.train_batch() if args.overfit_single_batch else None
    if resume_checkpoint is not None:
        saved_model_config = (
            resume_checkpoint.get("model_config")
            if isinstance(resume_checkpoint, dict) else None
        )
        if isinstance(saved_model_config, dict):
            # Pre-v4 checkpoints have no embedding-layout fields.  Resume
            # them in their original full-rank/separate-head layout unless
            # the user explicitly asked for the v4 layout.
            if (
                "factorized_embedding" not in saved_model_config
                and not _was_provided(raw_argv, "--factorized-embedding")
            ):
                args.factorized_embedding = False
            if (
                "mtp_tie_heads" not in saved_model_config
                and not _was_provided(raw_argv, "--mtp-tie-heads")
            ):
                args.mtp_tie_heads = False
            for key, arg_name in (
                ("moe_hierarchical", "--moe-hierarchical"),
                ("moe_context_router", "--moe-context-router"),
                ("moe_route_feedback", "--moe-route-feedback"),
                ("moe_route_to_attention", "--moe-route-to-attention"),
                ("moe_associative_memory", "--moe-associative-memory"),
                ("moe_hyperdimensional", "--moe-hyperdimensional"),
                ("mtp_hd_time", "--mtp-hd-time"),
                ("mtp_branch_norm", "--mtp-branch-norm"),
                ("use_combinatorial_memory", "--combinatorial-memory"),
                ("use_concept_loop", "--concept-loop"),
                ("composition_head", "--composition-head"),
                ("task_vocab_size", "--task-vocab-size"),
                ("concept_router_tokens", "--concept-router-tokens"),
                ("grad_checkpoint", "--grad-checkpoint"),
            ):
                if (
                    key not in saved_model_config
                    and not _was_provided(raw_argv, arg_name)
                ):
                    setattr(args, key, False)
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
    first_byte_map = (
        first_byte_map_cpu.to(device)
        if first_byte_map_cpu is not None else None
    )
    parameter_count = model.parameter_count()
    print(
        f"[model] params={parameter_count:,} layers={cfg.n_layers} "
        f"d_model={cfg.d_model} heads={cfg.n_heads} ctx={cfg.ctx} "
        f"mtp_heads={_model_core(model).n_mtp_heads} moe_layers={cfg.moe_layers}",
        flush=True,
    )
    expected_params = expected_parameter_count(args.preset)
    if expected_params is not None:
        if parameter_count == expected_params:
            print(
                f"[preset] parameter count matches {args.preset}: "
                f"{parameter_count:,}",
                flush=True,
            )
        else:
            print(
                f"[preset] WARNING expected {expected_params:,} params for "
                f"{args.preset}, got {parameter_count:,}",
                flush=True,
            )
    moe_layout = []
    for layer_index, block in enumerate(_model_core(model).blocks):
        if getattr(block, "is_moe", False):
            inner = block.mlp.inner
            moe_layout.append(
                {
                    "layer": layer_index,
                    "branches": int(inner.branches),
                    "experts_per_branch": int(inner.experts_per_branch),
                    "topk": int(inner.topk),
                    "degenerate": bool(inner.branch_degenerate),
                }
            )
    if moe_layout:
        print(f"[moe] layout={moe_layout}", flush=True)
        if any(item["degenerate"] for item in moe_layout):
            print(
                "[moe] WARNING hierarchical layout is degenerate: top-k "
                "selects every expert in the selected branch",
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
                name.startswith("mtp_")
                or name.startswith("mtp_heads.")
                or name.startswith("mtp_adapters.")
                or name.startswith("mtp_blocks.")
                or name.startswith("mtp_now.")
                or name == "mtp_next_norm.weight"
                or name.startswith("mtp_cond_proj.")
                or name.startswith("mtp_gate.")
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

    mtp_param_ids = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (
            name.startswith("mtp_")
            or name.startswith("mtp.")
            or ".mtp_" in name
        )
    }
    trunk_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in mtp_param_ids
    ]
    mtp_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) in mtp_param_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": trunk_params, "lr": args.lr, "lr_scale": 1.0},
            {
                "params": mtp_params,
                "lr": args.lr * args.mtp_lr_scale,
                "lr_scale": args.mtp_lr_scale,
            },
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=(device.type == "cuda" and not amp),
    )
    scaler = (
        torch.amp.GradScaler("cuda", init_scale=args.amp_init_scale)
        if amp and device.type == "cuda" and cuda_amp_dtype == torch.float16
        else None
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
            _load_resume_state(
                _model_core(model), resume_state, label="resume checkpoint"
            )
        except RuntimeError as exc:
            raise SystemExit(
                f"resume checkpoint is incompatible with this model: {exc}"
            ) from exc
        if ema_model is not None and resume_checkpoint.get("ema") is not None:
            try:
                _load_resume_state(
                    _model_core(ema_model), resume_checkpoint["ema"],
                    label="resume EMA",
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

    lr_total_steps = int(args.lr_total_steps or args.steps)
    if args.warmup_steps > 0:
        warmup_steps = args.warmup_steps
    else:
        warmup_steps = int(round(lr_total_steps * args.warmup_frac))
    warmup_steps = max(0, min(warmup_steps, max(0, lr_total_steps - 1)))
    effective_warmup_frac = warmup_steps / max(1, lr_total_steps)
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
        "mtp_heads": _model_core(model).n_mtp_heads,
        "mtp_max_lag": cfg.mtp,
        "lr_horizon": horizon,
        "warmup_steps": warmup_steps,
        "warmup_frac": effective_warmup_frac,
        "moe_layout": moe_layout,
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
    nonfinite_skips = 0

    def current_lr(step: int, elapsed: float) -> float:
        if horizon == "time" and args.max_seconds > 0:
            progress = elapsed / args.max_seconds
        else:
            progress = step / max(1, lr_total_steps)
        # A wall-clock horizon should use the same stable/decay shape as the
        # WSD step horizon.  Passing "time" directly used to fall through to
        # cosine decay from step zero.
        lr_shape = "wsd" if horizon == "time" else horizon
        scale = lr_scale(
            progress,
            effective_warmup_frac,
            args.min_lr_frac,
            schedule=lr_shape,
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
        # ``best_every`` is a checkpoint cadence, so a non-cadence evaluation
        # must not become ``best`` unless it is the final/early-stop write.
        # Otherwise the best state is tracked but never written: if the final
        # model is worse, no best.pt exists at all.
        if not force and step != args.steps:
            if args.best_every <= 0 or step % args.best_every != 0:
                return
        if best is not None and loss >= best.get("loss", float("inf")):
            return
        best = {
            "step": step,
            "source": source_name,
            "loss": float(loss),
            "metrics": metrics,
        }
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
        f"horizon={horizon} warmup_steps={warmup_steps} "
        f"feedback_warmup={args.feedback_gate_warmup_steps}",
        flush=True,
    )

    def feedback_progress_for(step: int) -> float:
        if args.feedback_gate_warmup_steps <= 0:
            return 1.0
        return min(
            1.0,
            max(0.0, step / float(args.feedback_gate_warmup_steps)),
        )

    core_model = _model_core(model)
    core_model.set_feedback_progress(feedback_progress_for(start_step))

    eval_before = None
    if args.eval_before:
        eval_before = evaluate(
            model, source.eval_batches, device, amp, args.mtp_weight,
            sidecar_builder=getattr(source, "build_sidecar", None),
            task_id=args.task_id,
            bytes_per_token=source.info.get("bytes_per_token"),
            eval_task_ids=getattr(source, "eval_task_ids", None),
            amp_dtype=cuda_amp_dtype,
        )
        print(
            f"[eval-before] model loss={eval_before['loss']:.4f} "
            f"bits={eval_before['bits_per_token']:.4f} "
            f"top1={eval_before['accuracy'] * 100:.2f}%",
            flush=True,
        )
        if device.type == "cuda":
            # The eval graph is gone, but its reserved blocks can otherwise
            # starve the first backward at long context.
            torch.cuda.empty_cache()

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

        lr = current_lr(step, run_elapsed)
        for group in optimizer.param_groups:
            group["lr"] = lr * float(group.get("lr_scale", 1.0))
        feedback_progress = feedback_progress_for(step)
        core_model.set_feedback_progress(feedback_progress)
        if args.mtp_gt_mix_steps > 0:
            mix_frac = min(1.0, step / args.mtp_gt_mix_steps)
        else:
            mix_frac = 1.0
        mtp_gt_mix = (
            args.mtp_gt_mix_start
            + (args.mtp_gt_mix_end - args.mtp_gt_mix_start) * mix_frac
        )

        xb = (
            overfit_batch
            if overfit_batch is not None
            else source.train_batch()
        )
        sidecar = (
            source.build_sidecar(xb)
            if hasattr(source, "build_sidecar") else None
        )
        if sidecar is not None:
            sidecar = sidecar.to(device)
        xb = xb.to(device, non_blocking=True)
        if args.task_id >= 0:
            task_ids = torch.full(
                (xb.size(0),), args.task_id, device=device,
                dtype=torch.long,
            )
        elif getattr(source, "last_task_ids", None) is not None:
            task_ids = source.last_task_ids.to(
                device=device, dtype=torch.long
            )
        else:
            task_ids = None
        optimizer.zero_grad(set_to_none=True)
        with _amp_context(amp, device.type, cuda_amp_dtype):
            logits, mtp_logits, aux = model(
                xb, mtp_teacher=True, mtp_gt_mix=mtp_gt_mix,
                sidecar=sidecar, task_ids=task_ids,
                compute_mtp_aux=args.mtp_aux_weight > 0,
                compute_byte_probe=(
                    args.byte_probe_weight > 0
                    or args.cross_resolution_weight > 0
                ),
                compute_byte_sequence=args.byte_sequence_weight > 0,
            )
            loss, parts = lm_mtp_loss(
                logits,
                mtp_logits,
                xb,
                mtp_weight=args.mtp_weight,
                aux_logits=aux.get("mtp_aux_logits") if aux else None,
                aux_weight=args.mtp_aux_weight,
                aux_weights=aux.get("mtp_aux_layer_weights") if aux else None,
            )
            load_balance = _aux_scalar(aux.get("load_balance")) if aux else None
            router_z = _aux_scalar(aux.get("router_z")) if aux else None
            mtp_load_balance = (
                _aux_scalar(aux.get("mtp_load_balance")) if aux else None
            )
            mtp_router_z = (
                _aux_scalar(aux.get("mtp_router_z")) if aux else None
            )
            hd_diversity = (
                _aux_scalar(aux.get("hd_code_diversity")) if aux else None
            )
            object_balance = (
                _aux_scalar(aux.get("object_balance")) if aux else None
            )
            if load_balance is not None:
                loss = loss + args.load_balance_coef * load_balance
            if router_z is not None:
                loss = loss + args.router_z_coef * router_z
            if object_balance is not None:
                loss = loss + args.object_balance_coef * object_balance
            # Per-lag MTP blocks have their own routers.  They need the same
            # balancing pressure as the trunk or they collapse to a few
            # experts and one or more experts die.
            if mtp_load_balance is not None:
                loss = loss + args.load_balance_coef * mtp_load_balance
            if mtp_router_z is not None:
                loss = loss + args.router_z_coef * mtp_router_z
            if hd_diversity is not None:
                loss = loss + args.moe_hd_diversity_coef * hd_diversity
            if aux.get("mtp_gate_loss") is not None:
                loss = loss + args.mtp_gate_coef * aux["mtp_gate_loss"]
            composition_loss = None
            mtp_composition_loss = None
            if sidecar is not None and aux.get("composition_logits") is not None:
                comp_logits = aux["composition_logits"]
                if getattr(sidecar, "event_multi", None) is not None:
                    composition_loss = F.binary_cross_entropy_with_logits(
                        comp_logits,
                        sidecar.event_multi.to(comp_logits.dtype),
                    )
                    future_logits = aux.get("mtp_composition_logits") or []
                    future_lags = aux.get("mtp_composition_lags") or []
                    future_losses = []
                    for lag, future_comp_logits in zip(
                        future_lags, future_logits
                    ):
                        target_len = xb.size(1) - int(lag)
                        if target_len <= 0:
                            continue
                        future_targets = sidecar.event_multi[
                            :, int(lag):
                        ].to(future_comp_logits.dtype)
                        future_losses.append(
                            F.binary_cross_entropy_with_logits(
                                future_comp_logits[:, :target_len],
                                future_targets,
                            )
                        )
                    if future_losses:
                        mtp_composition_loss = torch.stack(
                            future_losses
                        ).mean()
                        loss = loss + (
                            args.mtp_composition_weight
                            * mtp_composition_loss
                        )
                else:
                    composition_loss = F.cross_entropy(
                        comp_logits.reshape(-1, comp_logits.size(-1)),
                        sidecar.event_ids.reshape(-1).long(),
                        ignore_index=-100,
                    )
                loss = loss + args.composition_weight * composition_loss
            byte_probe_loss = None
            byte_sequence_loss = None
            cross_resolution_loss = None
            word_probe_loss = None
            if sidecar is not None:
                byte_logits = aux.get("byte_probe_logits")
                if byte_logits is not None:
                    byte_targets = sidecar.first_byte[:, 1:].long()
                    byte_probe_loss = F.cross_entropy(
                        byte_logits[:, :-1].reshape(
                            -1, byte_logits.size(-1)
                        ),
                        byte_targets.reshape(-1),
                    )
                    loss = loss + args.byte_probe_weight * byte_probe_loss
                    if (
                        args.cross_resolution_weight > 0
                        and first_byte_map is not None
                    ):
                        temperature = max(
                            1e-3, float(args.cross_resolution_temperature)
                        )
                        main_probs = F.softmax(
                            logits[:, :-1].float() / temperature, dim=-1
                        )
                        # Sparse scatter to the 256 first-byte classes avoids
                        # a 64,060 x 256 dense projection per token.
                        index = first_byte_map.view(1, 1, -1).expand_as(
                            main_probs
                        )
                        main_byte = torch.zeros(
                            *main_probs.shape[:2], 256,
                            device=main_probs.device,
                            dtype=main_probs.dtype,
                        ).scatter_add_(2, index, main_probs)
                        main_byte = main_byte / main_byte.sum(
                            dim=-1, keepdim=True
                        ).clamp_min(1e-8)
                        byte_probs = F.softmax(
                            byte_logits[:, :-1].float() / temperature,
                            dim=-1,
                        )
                        byte_log_probs = F.log_softmax(
                            byte_logits[:, :-1].float() / temperature,
                            dim=-1,
                        )
                        # Bidirectional consistency: the byte head learns the
                        # token distribution's byte marginal, while the token
                        # head learns the byte head's marginal.  Each side sees
                        # the other as a detached teacher.
                        result_shape = byte_log_probs.shape
                        loss_byte = F.kl_div(
                            byte_log_probs.reshape(-1, 256),
                            main_byte.detach().reshape(-1, 256),
                            reduction="batchmean",
                        )
                        main_byte_log_probs = torch.log(
                            main_byte.clamp_min(1e-8)
                        )
                        loss_main = F.kl_div(
                            main_byte_log_probs.reshape(-1, 256),
                            byte_probs.detach().reshape(-1, 256),
                            reduction="batchmean",
                        )
                        if result_shape != byte_log_probs.shape:
                            raise RuntimeError("cross-resolution shape drift")
                        cross_resolution_loss = 0.5 * (
                            loss_byte + loss_main
                        )
                        loss = loss + (
                            args.cross_resolution_weight
                            * cross_resolution_loss
                        )
                sequence_logits = aux.get("byte_sequence_logits")
                if sequence_logits is not None and sidecar.token_bytes is not None:
                    sequence_targets = sidecar.token_bytes[:, 1:].long()
                    byte_sequence_loss = F.cross_entropy(
                        sequence_logits[:, :-1].reshape(-1, 256),
                        sequence_targets.reshape(-1),
                        ignore_index=-100,
                    )
                    loss = loss + (
                        args.byte_sequence_weight * byte_sequence_loss
                    )
                word_logits = aux.get("word_probe_logits")
                if word_logits is not None:
                    word_targets = sidecar.word_vocab_ids[:, 1:].long()
                    if (word_targets > 0).any():
                        word_probe_loss = F.cross_entropy(
                            word_logits[:, :-1].reshape(
                                -1, word_logits.size(-1)
                            ),
                            word_targets.reshape(-1),
                            ignore_index=0,
                        )
                        loss = loss + (
                            args.word_probe_weight * word_probe_loss
                        )
            pointer_loss = None
            triple_pointer_loss = None
            object_negative_loss = None
            core = _model_core(model)
            mtp_pointer_loss = None
            mtp_triple_pointer_loss = None
            if (
                aux.get("object_pointer_logits") is not None
                and getattr(core.concept_loop, "last_object_vertices", None)
                is not None
            ):
                ptr = aux["object_pointer_logits"]
                target_vertices = core.concept_loop.last_object_vertices
                if target_vertices.shape[2] > 0:
                    u_target = target_vertices[:, :, 0, 0].long()
                    v_target = target_vertices[:, :, 0, 1].long()
                    pointer_loss = 0.5 * (
                        F.cross_entropy(
                            ptr[:, :, 0].reshape(-1, ptr.size(-1)),
                            u_target.reshape(-1),
                        )
                        + F.cross_entropy(
                            ptr[:, :, 1].reshape(-1, ptr.size(-1)),
                            v_target.reshape(-1),
                        )
                    )
                    loss = loss + args.object_pointer_weight * pointer_loss
                    if args.object_negative_weight > 0:
                        object_negative_loss = 0.5 * (
                            _hard_negative_margin(
                                ptr[:, :, 0], u_target,
                                args.object_negative_margin,
                            )
                            + _hard_negative_margin(
                                ptr[:, :, 1], v_target,
                                args.object_negative_margin,
                            )
                        )
                        loss = loss + (
                            args.object_negative_weight
                            * object_negative_loss
                        )
            if (
                aux.get("object_triple_pointer_logits") is not None
                and getattr(
                    core.concept_loop, "last_triple_vertices", None
                ) is not None
            ):
                ptr3 = aux["object_triple_pointer_logits"]
                triple_target = core.concept_loop.last_triple_vertices
                if triple_target.shape[2] > 0:
                    target3 = triple_target[:, :, 0, :].long()
                    triple_pointer_loss = sum(
                        F.cross_entropy(
                            ptr3[:, :, slot].reshape(-1, ptr3.size(-1)),
                            target3[:, :, slot].reshape(-1),
                        )
                        for slot in range(3)
                    ) / 3.0
                    loss = loss + (
                        args.object_triple_pointer_weight
                        * triple_pointer_loss
                    )
                    if args.object_negative_weight > 0:
                        triple_negative = sum(
                            _hard_negative_margin(
                                ptr3[:, :, slot],
                                target3[:, :, slot],
                                args.object_negative_margin,
                            )
                            for slot in range(3)
                        ) / 3.0
                        object_negative_loss = (
                            triple_negative
                            if object_negative_loss is None
                            else 0.5 * (
                                object_negative_loss + triple_negative
                            )
                        )
                        loss = loss + (
                            args.object_negative_weight
                            * triple_negative
                        )
            if (
                getattr(core, "concept_loop", None) is not None
                and getattr(core.concept_loop, "last_object_vertices", None)
                is not None
                and aux.get("mtp_object_pointer_logits")
            ):
                target_vertices = core.concept_loop.last_object_vertices
                if target_vertices.shape[2] > 0:
                    future_pointer_losses = []
                    for lag, ptr in zip(
                        aux.get("mtp_object_pointer_lags") or [],
                        aux["mtp_object_pointer_logits"],
                    ):
                        target_len = xb.size(1) - int(lag)
                        if target_len <= 0:
                            continue
                        future_target = target_vertices[:, int(lag):, 0, :]
                        future_pointer_losses.append(
                            0.5 * (
                                F.cross_entropy(
                                    ptr[:, :target_len, 0].reshape(
                                        -1, ptr.size(-1)
                                    ),
                                    future_target[:, :, 0].long().reshape(-1),
                                )
                                + F.cross_entropy(
                                    ptr[:, :target_len, 1].reshape(
                                        -1, ptr.size(-1)
                                    ),
                                    future_target[:, :, 1].long().reshape(-1),
                                )
                            )
                        )
                    if future_pointer_losses:
                        mtp_pointer_loss = torch.stack(
                            future_pointer_losses
                        ).mean()
                        loss = loss + (
                            args.mtp_object_pointer_weight
                            * mtp_pointer_loss
                        )
            if (
                getattr(core, "concept_loop", None) is not None
                and getattr(core.concept_loop, "last_triple_vertices", None)
                is not None
                and aux.get("mtp_object_triple_pointer_logits")
            ):
                triple_target = core.concept_loop.last_triple_vertices
                if triple_target.shape[2] > 0:
                    future_triple_losses = []
                    for lag, ptr3 in zip(
                        aux.get("mtp_object_triple_pointer_lags") or [],
                        aux["mtp_object_triple_pointer_logits"],
                    ):
                        target_len = xb.size(1) - int(lag)
                        if target_len <= 0:
                            continue
                        future = triple_target[:, int(lag):, 0, :].long()
                        future_triple_losses.append(
                            sum(
                                F.cross_entropy(
                                    ptr3[:, :target_len, slot].reshape(
                                        -1, ptr3.size(-1)
                                    ),
                                    future[:, :, slot].reshape(-1),
                                )
                                for slot in range(3)
                            ) / 3.0
                        )
                    if future_triple_losses:
                        mtp_triple_pointer_loss = torch.stack(
                            future_triple_losses
                        ).mean()
                        loss = loss + (
                            args.mtp_object_triple_pointer_weight
                            * mtp_triple_pointer_loss
                        )
            loss_value = float(loss.detach())

        amp_skip = False
        loss_finite = bool(torch.isfinite(loss))
        if not loss_finite:
            # A non-finite MTP diagnostic must not poison the trunk.  Skip the
            # optimizer step and keep training from the last finite weights.
            amp_skip = True
            nonfinite_skips += 1
            optimizer.zero_grad(set_to_none=True)
            grad_norm = torch.tensor(float("nan"), device=device)
            print(
                f"[warn] non-finite loss at step {step}; skipped optimizer step "
                f"(ce={_finite(parts['ce'])} mtp={_finite(parts['mtp'])})",
                flush=True,
            )
        elif scaler is not None:
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            )
            if not bool(torch.isfinite(grad_norm)):
                amp_skip = True
                nonfinite_skips += 1
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                print(
                    f"[warn] non-finite grad at step {step}; skipped optimizer step",
                    flush=True,
                )
            else:
                scaler.step(optimizer)
                scaler.update()
                amp_skip = scaler.get_scale() < scale_before
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            )
            if not bool(torch.isfinite(grad_norm)):
                amp_skip = True
                nonfinite_skips += 1
                optimizer.zero_grad(set_to_none=True)
                print(
                    f"[warn] non-finite grad at step {step}; skipped optimizer step",
                    flush=True,
                )
            elif device.type == "xla":
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
            "mtp_load_balance": (
                _finite(mtp_load_balance) if mtp_load_balance is not None else None
            ),
            "mtp_router_z": (
                _finite(mtp_router_z) if mtp_router_z is not None else None
            ),
            "hd_code_diversity": (
                _finite(hd_diversity) if hd_diversity is not None else None
            ),
            "object_balance": (
                _finite(object_balance) if object_balance is not None else None
            ),
            "feedback": (
                {
                    key: _finite(value)
                    for key, value in aux.get("feedback", {}).items()
                }
                if aux else None
            ),
            "composition": (
                _finite(composition_loss)
                if composition_loss is not None else None
            ),
            "byte_probe": (
                _finite(byte_probe_loss)
                if byte_probe_loss is not None else None
            ),
            "byte_sequence": (
                _finite(byte_sequence_loss)
                if byte_sequence_loss is not None else None
            ),
            "cross_resolution": (
                _finite(cross_resolution_loss)
                if cross_resolution_loss is not None else None
            ),
            "word_probe": (
                _finite(word_probe_loss)
                if word_probe_loss is not None else None
            ),
            "mtp_composition": (
                _finite(mtp_composition_loss)
                if mtp_composition_loss is not None else None
            ),
            "object_pointer": (
                _finite(pointer_loss) if pointer_loss is not None else None
            ),
            "object_triple_pointer": (
                _finite(triple_pointer_loss)
                if triple_pointer_loss is not None else None
            ),
            "object_negative": (
                _finite(object_negative_loss)
                if object_negative_loss is not None else None
            ),
            "mtp_object_pointer": (
                _finite(mtp_pointer_loss)
                if mtp_pointer_loss is not None else None
            ),
            "mtp_object_triple_pointer": (
                _finite(mtp_triple_pointer_loss)
                if mtp_triple_pointer_loss is not None else None
            ),
            "grad_norm": _finite(grad_norm),
            "amp_skip": bool(amp_skip),
            "tokens": int(args.batch_size * args.ctx),
            "feedback_progress": float(feedback_progress),
            "bits_per_token": float(parts["ce"]) / math.log(2),
            "bits_per_byte": (
                float(parts["ce"])
                / (math.log(2) * source.info["bytes_per_token"])
                if source.info.get("bytes_per_token") else None
            ),
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
                model, source.eval_batches, device, amp, args.mtp_weight,
                sidecar_builder=getattr(source, "build_sidecar", None),
                task_id=args.task_id,
                bytes_per_token=source.info.get("bytes_per_token"),
                eval_task_ids=getattr(source, "eval_task_ids", None),
                amp_dtype=cuda_amp_dtype,
            )
            val_ema = (
                evaluate(
                    ema_model, source.eval_batches, device, amp,
                    args.mtp_weight,
                    sidecar_builder=getattr(source, "build_sidecar", None),
                    task_id=args.task_id,
                    bytes_per_token=source.info.get("bytes_per_token"),
                    eval_task_ids=getattr(source, "eval_task_ids", None),
                    amp_dtype=cuda_amp_dtype,
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
            if device.type == "cuda":
                torch.cuda.empty_cache()
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
    core_model.set_feedback_progress(1.0)
    final_metrics = evaluate(
        model, source.eval_batches, device, amp, args.mtp_weight,
        sidecar_builder=getattr(source, "build_sidecar", None),
        task_id=args.task_id,
        bytes_per_token=source.info.get("bytes_per_token"),
        eval_task_ids=getattr(source, "eval_task_ids", None),
        amp_dtype=cuda_amp_dtype,
    )
    final_ema_metrics = (
        evaluate(
            ema_model, source.eval_batches, device, amp, args.mtp_weight,
            sidecar_builder=getattr(source, "build_sidecar", None),
            task_id=args.task_id,
            bytes_per_token=source.info.get("bytes_per_token"),
            eval_task_ids=getattr(source, "eval_task_ids", None),
            amp_dtype=cuda_amp_dtype,
        )
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
        "nonfinite_skips": int(nonfinite_skips),
        "elapsed_seconds": elapsed_total,
        "wall_seconds": time.time() - total_start,
        "device": str(device),
        "amp": amp,
        "data": source.info,
        "objective": {
            "main_ce_weight": 1.0,
            "mtp_weight": args.mtp_weight,
            "mtp_heads": _model_core(model).n_mtp_heads,
            "mtp_lags": [2 + i for i in range(_model_core(model).n_mtp_heads)],
            "load_balance_coef": args.load_balance_coef,
            "router_z_coef": args.router_z_coef,
        },
        "mtp": {
            "enabled": _model_core(model).n_mtp_heads > 0,
            "heads": _model_core(model).n_mtp_heads,
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
        "feedback_gate_warmup_steps": int(
            args.feedback_gate_warmup_steps
        ),
        "feedback_progress_final": 1.0,
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
