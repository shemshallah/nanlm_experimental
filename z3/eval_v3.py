#!/usr/bin/env python3
"""Evaluation harness for the z3 V3LM architecture.

The harness evaluates a checkpoint (or a tiny random model) on fixed validation
positions and reports bits/token, top-1, top-5, MTP top-1 per future offset,
and router utilization when the model contains sparse MoE layers.

Checkpoint formats accepted:

  * ``{"model": state_dict, "config": {...}}``
  * ``{"model": state_dict, "config": V3Config(...)}`` (trusted checkpoints)
  * ``{"state_dict": ..., "model_config": ...}`` and similar aliases
  * a bare state_dict plus ``--config model_config.json``
  * a bare model config JSON via ``--config`` (the checkpoint is still needed)

Per-corpus evaluation uses repeated ``--corpus`` flags.  A corpus name may be
provided as ``name=path``; otherwise the file stem is used::

    python z3/eval_v3.py \
        --checkpoint run/final.pt \
        --corpus prose=corpora/prose.txt \
        --corpus code=corpora/code.txt \
        --tokenizer bpe --bpe-model bpe32k.json \
        --ctx 1024 --batch-size 8 --batches 16 \
        --device cuda --out run/eval.json

A dependency-free smoke run needs no checkpoint or corpus::

    python z3/eval_v3.py --random-smoke --out /tmp/eval_smoke.json

The BPE path prefers the HuggingFace ``tokenizers`` package and falls back to a
small byte-level BPE reader for a tokenizer JSON file when the package is not
installed.  The fallback is intended for evaluation convenience, not exact
high-throughput parity with the native tokenizer.
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import math
import random
import re
import sys
import time
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

try:  # imported as ``z3.eval_v3``
    from .v3_model import V3Config, V3LM
except ImportError:  # run as a script or with z3/ on sys.path
    from v3_model import V3Config, V3LM


VERSION = 1
DEFAULT_SEED = 20261112
V4_TOKENIZER_PATH = Path(__file__).resolve().with_name(
    "tokenizers"
) / "v4_64k" / "tokenizer.json"
LAYERED_TOKENIZER_PATH = Path(__file__).resolve().with_name(
    "tokenizers"
) / "layered_v1" / "layered_tokenizer.json"
CONFIG_FIELD_NAMES = {f.name for f in fields(V3Config)}
MODEL_KEYS = (
    "model",
    "ema",
    "ema_model",
    "model_state",
    "state_dict",
    "model_state_dict",
    "net",
)
CONFIG_KEYS = ("model_config", "config", "cfg", "v3_config", "hparams", "args")


# ---------------------------------------------------------------------------
# Config and checkpoint loading
# ---------------------------------------------------------------------------


def _as_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "__dict__"):
        try:
            return vars(value)
        except TypeError:
            return None
    return None


def _validate_config(cfg: V3Config) -> None:
    if cfg.vocab_size < 2:
        raise ValueError("config.vocab_size must be >= 2")
    if cfg.ctx < 1:
        raise ValueError("config.ctx must be >= 1")
    if cfg.d_model < 1 or cfg.n_layers < 1 or cfg.n_heads < 1:
        raise ValueError("config.d_model, n_layers and n_heads must be positive")
    if cfg.d_model % cfg.n_heads:
        raise ValueError(
            f"config.d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})"
        )
    if cfg.swiglu < 1:
        raise ValueError("config.swiglu must be positive")
    if cfg.moe_layers > cfg.n_layers:
        raise ValueError("config.moe_layers cannot exceed n_layers")
    if cfg.moe_layers > 0:
        if cfg.n_experts < 1 or cfg.topk < 1:
            raise ValueError("config.n_experts and config.topk must be positive")
        if cfg.topk > cfg.n_experts:
            raise ValueError("config.topk cannot exceed n_experts")
        if cfg.expert_hidden < 1:
            raise ValueError("config.expert_hidden must be positive")
    if cfg.mtp < 1:
        raise ValueError("config.mtp must be >= 1")
    if not 0.0 <= float(cfg.dropout) <= 1.0:
        raise ValueError("config.dropout must be in [0, 1]")


def parse_config(raw: Any) -> V3Config:
    """Build a V3Config from a dataclass, namespace, or (possibly nested) dict."""

    if isinstance(raw, V3Config):
        _validate_config(raw)
        return raw
    mapping = _as_mapping(raw)
    if mapping is None:
        raise ValueError("model config must be a V3Config, dict, or namespace")

    if not any(key in CONFIG_FIELD_NAMES for key in mapping):
        for key in ("config", "cfg", "model_config", "v3_config", "model", "args"):
            nested = _as_mapping(mapping.get(key)) if key in mapping else None
            if nested and any(k in CONFIG_FIELD_NAMES for k in nested):
                mapping = nested
                break

    kwargs = {
        key: value
        for key, value in mapping.items()
        if key in CONFIG_FIELD_NAMES and value is not None
    }
    if not kwargs:
        raise ValueError(
            "no V3Config fields found in model config "
            f"(expected some of: {', '.join(sorted(CONFIG_FIELD_NAMES))})"
        )
    # JSON has no tuples.  Normalize tuple-typed fields so a config loaded
    # from a checkpoint round-trips equal to the in-memory V3Config.
    for key in ("mtp_aux_layers", "mtp_aux_layer_weights"):
        value = kwargs.get(key)
        if isinstance(value, list):
            kwargs[key] = tuple(value)
    # Checkpoints created before the v4 defaults have no embedding-layout
    # fields.  Keep them on their original full-rank, separate-head layout.
    if "factorized_embedding" not in kwargs:
        kwargs["factorized_embedding"] = False
    if "mtp_tie_heads" not in kwargs:
        kwargs["mtp_tie_heads"] = False
    for key in (
        "moe_hierarchical", "moe_context_router",
        "moe_route_feedback", "moe_route_to_attention",
        "moe_associative_memory", "moe_hyperdimensional", "mtp_hd_time",
        "mtp_branch_norm", "use_combinatorial_memory",
    ):
        if key not in kwargs:
            kwargs[key] = False
    try:
        cfg = V3Config(**kwargs)
    except TypeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"invalid V3Config values: {exc}") from exc
    _validate_config(cfg)
    return cfg


def load_config_json(path: str | Path) -> V3Config:
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"model config not found: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON model config {p}: {exc}") from exc
    return parse_config(raw)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as first_error:  # config objects / older checkpoint payloads
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception as second_error:
            raise RuntimeError(f"could not load checkpoint {path}: {second_error}") from first_error


def _is_state_dict(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(isinstance(k, str) and torch.is_tensor(v) for k, v in value.items())
    )


def _state_from_payload(payload: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any] | None, Any]:
    if isinstance(payload, torch.nn.Module):
        return payload.state_dict(), None, payload
    if _is_state_dict(payload):
        return payload, None, None
    if isinstance(payload, dict):
        for key in MODEL_KEYS:
            value = payload.get(key)
            if value is None:
                continue
            if isinstance(value, torch.nn.Module):
                return value.state_dict(), payload, value
            if _is_state_dict(value):
                return value, payload, None
            if isinstance(value, dict):
                for nested_key in ("state_dict", "model", "model_state_dict"):
                    nested = value.get(nested_key)
                    if _is_state_dict(nested):
                        return nested, payload, None
    raise ValueError(
        "checkpoint is not a model state_dict; expected a dict with a 'model' entry"
    )


def _config_from_payload(payload: Any, model_obj: Any = None) -> V3Config | None:
    if model_obj is not None and getattr(model_obj, "cfg", None) is not None:
        try:
            return parse_config(model_obj.cfg)
        except ValueError:
            pass
    if not isinstance(payload, dict):
        return None
    for key in CONFIG_KEYS:
        if key not in payload:
            continue
        try:
            return parse_config(payload[key])
        except ValueError:
            continue
    for key in ("model", "ema_model"):
        nested = payload.get(key)
        if not isinstance(nested, dict):
            continue
        for config_key in CONFIG_KEYS:
            if config_key in nested:
                try:
                    return parse_config(nested[config_key])
                except ValueError:
                    continue
    try:
        return parse_config(payload)
    except ValueError:
        return None


def _clean_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    prefixes = ("module.", "_orig_mod.", "model.")
    for key, value in state.items():
        name = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if name.startswith(prefix) and len(name) > len(prefix):
                    name = name[len(prefix):]
                    changed = True
        cleaned[name] = value
    return cleaned


def load_checkpoint(
    checkpoint: str | Path,
    config_json: str | Path | None = None,
) -> tuple[V3LM, V3Config, dict[str, Any]]:
    """Load a V3LM checkpoint and return ``(model, config, metadata)``."""

    path = Path(checkpoint).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    payload = _torch_load(path)
    state, container, model_obj = _state_from_payload(payload)

    if config_json:
        cfg = load_config_json(config_json)
        config_source = "config-json"
    else:
        cfg = _config_from_payload(container if container is not None else payload, model_obj)
        config_source = "checkpoint"
    if cfg is None:
        raise ValueError(
            f"{path} does not contain a V3 model config; pass --config model_config.json"
        )

    model = V3LM(cfg)
    cleaned = _clean_state_dict(state)
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={sorted(missing)[:8]}")
        if unexpected:
            details.append(f"unexpected={sorted(unexpected)[:8]}")
        raise RuntimeError(
            f"checkpoint state does not match V3Config ({', '.join(details)})"
        )
    model.eval()

    metadata: dict[str, Any] = {
        "config_source": config_source,
        "checkpoint_entries": len(cleaned),
        "step": None,
    }
    if isinstance(container, dict):
        for key in ("step", "global_step", "iteration", "epoch"):
            value = container.get(key)
            if isinstance(value, (int, float)):
                metadata["step"] = int(value)
                break
    return model, cfg, metadata


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte -> printable unicode mapping."""

    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = list(bs)
    n = 0
    for byte in range(256):
        if byte not in bs:
            bs.append(byte)
            cs.append(256 + n)
            n += 1
    return {byte: chr(codepoint) for byte, codepoint in zip(bs, cs)}


_BYTE_ENCODER = _bytes_to_unicode()

try:  # optional, gives closer parity with HuggingFace's GPT-2 regex
    import regex as _regex

    _PRETOKEN_RE = _regex.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:  # ASCII-oriented approximation is still valid BPE
    _PRETOKEN_RE = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
    )


class BytesTokenizer:
    """UTF-8 byte tokenizer, matching the ``bytes`` mode used by z1/z2."""

    kind = "bytes"

    def __init__(self) -> None:
        self.vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8", errors="replace"))


class _SimpleBPE:
    """Small slow byte-level BPE encoder used when ``tokenizers`` is absent."""

    def __init__(self, path: str | Path) -> None:
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"BPE model not found: {p}")
        raw = json.loads(p.read_text(encoding="utf-8"))
        model = raw.get("model", raw)
        if not isinstance(model, dict):
            raise ValueError(f"unsupported BPE model JSON: {p}")
        model_type = str(model.get("type", "BPE")).upper()
        if model_type not in ("BPE", "BYTELEVEL"):
            raise ValueError(f"unsupported BPE model type {model_type!r} in {p}")
        vocab = model.get("vocab")
        if not isinstance(vocab, dict):
            raise ValueError(f"BPE model {p} has no vocab mapping")

        self.vocab: dict[str, int] = {str(k): int(v) for k, v in vocab.items()}
        self.ranks: dict[tuple[str, str], int] = {}
        for rank, merge in enumerate(model.get("merges", []) or []):
            if isinstance(merge, str):
                pair = tuple(merge.split(" "))
            else:
                pair = tuple(merge)
            if len(pair) == 2:
                self.ranks[(str(pair[0]), str(pair[1]))] = rank

        unk_token = model.get("unk_token") or "<unk>"
        self.unk_id = int(self.vocab.get(str(unk_token), 0))
        self.added: dict[str, int] = {}
        for token in raw.get("added_tokens", []) or []:
            if isinstance(token, dict) and "content" in token and "id" in token:
                self.added[str(token["content"])] = int(token["id"])
        self._added_re = None
        if self.added:
            parts = sorted(self.added, key=len, reverse=True)
            self._added_re = re.compile("(" + "|".join(re.escape(x) for x in parts) + ")")
        self._cache: dict[str, tuple[int, ...]] = {}
        self.vocab_size = max(self.vocab.values(), default=0) + 1

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        ids: list[int] = []
        if self._added_re is None:
            return self._encode_plain(text)
        for part in self._added_re.split(text):
            if not part:
                continue
            if part in self.added:
                ids.append(self.added[part])
            else:
                ids.extend(self._encode_plain(part))
        return ids

    def _encode_plain(self, text: str) -> list[int]:
        mapped = "".join(_BYTE_ENCODER[b] for b in text.encode("utf-8"))
        ids: list[int] = []
        for token in _PRETOKEN_RE.findall(mapped):
            ids.extend(self._encode_token(token))
        return ids

    def _encode_token(self, token: str) -> tuple[int, ...]:
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        chars = tuple(token)
        while len(chars) > 1:
            best_pair = None
            best_rank = None
            for pair in zip(chars, chars[1:]):
                rank = self.ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            merged: list[str] = []
            i = 0
            while i < len(chars):
                if i + 1 < len(chars) and (chars[i], chars[i + 1]) == best_pair:
                    merged.append(chars[i] + chars[i + 1])
                    i += 2
                else:
                    merged.append(chars[i])
                    i += 1
            chars = tuple(merged)
        result = tuple(self.vocab.get(char, self.unk_id) for char in chars)
        self._cache[token] = result
        return result


class BPETokenizer:
    """HuggingFace tokenizer JSON loader with a small pure-Python fallback."""

    kind = "bpe"

    def __init__(self, bpe_model: str | Path) -> None:
        p = Path(bpe_model).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"BPE model not found: {p}")
        self._native = None
        self._fallback = None
        try:
            from tokenizers import Tokenizer  # type: ignore

            self._native = Tokenizer.from_file(str(p))
            self.vocab_size = int(self._native.get_vocab_size())
            self.backend = "tokenizers"
        except ImportError:
            self._fallback = _SimpleBPE(p)
            self.vocab_size = self._fallback.vocab_size
            self.backend = "fallback"

    def encode(self, text: str) -> list[int]:
        if self._native is not None:
            return list(self._native.encode(text).ids)
        assert self._fallback is not None
        return self._fallback.encode(text)


class LayeredTokenizerAdapter:
    """Flat subword stream backed by the tuned layered artifact."""

    kind = "layered"

    def __init__(self, path: str | Path):
        from layered_tokenizer import load_artifact

        self.path = Path(path)
        self._layered = load_artifact(self.path)
        self.vocab_size = self._layered.vocab_size
        self.backend = "layered"

    def encode(self, text: str) -> list[int]:
        encoded = self._layered.encode(text, add_document_frame=False)
        return list(encoded.subword.ids)


def load_tokenizer(
    kind: str | None,
    bpe_model: str | Path | None = None,
    layered_model: str | Path | None = None,
):
    """Resolve ``--tokenizer``/``--bpe-model`` into a tokenizer object."""

    if kind is None or str(kind).lower() == "auto":
        if layered_model:
            kind = "layered"
        else:
            kind = "bpe" if bpe_model else "layered"
    kind = kind.lower()
    if kind == "bytes":
        return BytesTokenizer()
    if kind == "v4":
        return BPETokenizer(V4_TOKENIZER_PATH)
    if kind == "layered":
        return LayeredTokenizerAdapter(
            layered_model or LAYERED_TOKENIZER_PATH
        )
    if kind == "bpe":
        if not bpe_model:
            raise ValueError("--tokenizer bpe requires --bpe-model")
        return BPETokenizer(bpe_model)
    raise ValueError(
        f"unknown tokenizer {kind!r}; expected 'auto', 'v4', 'layered', "
        "'bpe', or 'bytes'"
    )


def encode_file(
    path: str | Path,
    tokenizer: BytesTokenizer | BPETokenizer,
    max_bytes: int = 0,
) -> list[int]:
    """Encode a text file in bounded chunks, preserving UTF-8 boundaries."""

    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"corpus not found: {p}")
    ids: list[int] = []
    decoder = codecs.getincrementaldecoder("utf-8")("ignore")
    total = 0
    with p.open("rb") as handle:
        while True:
            chunk = handle.read(4 << 20)
            if not chunk:
                break
            if max_bytes > 0:
                remaining = max_bytes - total
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk[:remaining]
            total += len(chunk)
            text = decoder.decode(chunk)
            if text:
                ids.extend(tokenizer.encode(text))
            if max_bytes > 0 and total >= max_bytes:
                break
    tail = decoder.decode(b"", final=True)
    if tail:
        ids.extend(tokenizer.encode(tail))
    return ids


# ---------------------------------------------------------------------------
# Fixed validation positions
# ---------------------------------------------------------------------------


def _stable_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def make_positions(
    n_tokens: int,
    ctx: int,
    count: int,
    horizon: int,
    seed: int,
    key: str,
) -> tuple[list[int], dict[str, Any]]:
    """Draw fixed validation positions ``p``.

    A position ``p`` uses tokens ``[p-ctx, p)`` as input and predicts
    ``token[p]``; MTP offset ``k`` predicts ``token[p+k-1]``.  Sampling stops
    before the end so every requested future offset has a valid target.
    """

    if ctx < 1:
        raise ValueError("ctx must be >= 1")
    if count < 1:
        raise ValueError("position count must be >= 1")
    horizon = max(1, int(horizon))
    lo = ctx
    hi = n_tokens - horizon - 1  # inclusive
    if hi < lo:
        raise ValueError(
            f"corpus {key!r} has {n_tokens} tokens, but ctx={ctx} and "
            f"horizon={horizon} require at least {ctx + horizon + 1}"
        )
    population = hi - lo + 1
    replacement = count > population
    rng = random.Random(_stable_seed(seed, key))
    if replacement:
        positions = [rng.randrange(lo, hi + 1) for _ in range(count)]
    else:
        positions = sorted(rng.sample(range(lo, hi + 1), count))
    digest = hashlib.sha1(",".join(str(p) for p in positions).encode("ascii")).hexdigest()[:16]
    meta = {
        "corpus": key,
        "seed": seed,
        "count": count,
        "min": min(positions),
        "max": max(positions),
        "allowed_min": lo,
        "allowed_max": hi,
        "replacement": replacement,
        "digest": digest,
    }
    return positions, meta


# ---------------------------------------------------------------------------
# Router utilization
# ---------------------------------------------------------------------------


def _layer_number(name: str) -> int:
    match = re.search(r"\.(\d+)(?:\.|$)", name)
    return int(match.group(1)) if match else -1


class RouterTracker:
    """Capture top-k gate assignments from model layers that expose a router."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.entries: list[dict[str, Any]] = []
        self.handles: list[Any] = []
        for name, module in model.named_modules():
            # Current V3LM uses VectorizedSparseMoE (router Linear, n_experts,
            # topk).  Keep support for the earlier wrapper layout (gate, cfg).
            router = getattr(module, "router", None)
            if not isinstance(router, torch.nn.Linear):
                router = getattr(module, "gate", None)
            if not isinstance(router, torch.nn.Linear):
                continue
            cfg = getattr(module, "cfg", None)
            n_experts = getattr(module, "n_experts", None)
            topk = getattr(module, "topk", None)
            if n_experts is None and cfg is not None:
                n_experts = getattr(cfg, "n_experts", None)
            if topk is None and cfg is not None:
                topk = getattr(cfg, "topk", None)
            if n_experts is None or topk is None:
                continue
            n_experts = int(n_experts)
            topk = int(topk)
            if n_experts < 1 or topk < 1:
                continue
            entry = {
                "name": name,
                "layer": _layer_number(name),
                "n_experts": n_experts,
                "counts": torch.zeros(n_experts, dtype=torch.float64),
            }
            self.entries.append(entry)
            self.handles.append(
                router.register_forward_hook(self._hook(entry, min(topk, n_experts)))
            )

    @staticmethod
    def _hook(entry: dict[str, Any], topk: int):
        def hook(_module, _inputs, output):
            with torch.no_grad():
                indices = torch.topk(output.detach(), k=topk, dim=-1).indices.reshape(-1)
                counts = torch.bincount(
                    indices.cpu(), minlength=entry["n_experts"]
                ).to(torch.float64)
                entry["counts"] += counts

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def summary(self) -> dict[str, Any] | None:
        return summarize_router(self.entries)


def summarize_router(entries: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not entries:
        return None
    layers = []
    total_assignments = 0.0
    max_shares = []
    normalized_entropies = []
    dead_total = 0
    for entry in entries:
        counts = entry["counts"].to(torch.float64)
        n_experts = int(entry["n_experts"])
        total = float(counts.sum())
        if total > 0:
            probs = counts / total
            utilization = probs.tolist()
            entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum())
            dead = int((counts <= 0).sum())
        else:
            utilization = [0.0] * n_experts
            entropy = 0.0
            dead = n_experts
        max_share = max(utilization) if utilization else 0.0
        normalized = entropy / math.log(n_experts) if n_experts > 1 else 0.0
        layers.append(
            {
                "name": entry["name"],
                "layer": entry["layer"],
                "n_experts": n_experts,
                "assignments": int(total),
                "utilization": utilization,
                "max_share": max_share,
                "entropy": entropy,
                "normalized_entropy": normalized,
                "dead_experts": dead,
            }
        )
        total_assignments += total
        max_shares.append(max_share)
        normalized_entropies.append(normalized)
        dead_total += dead
    return {
        "layers": layers,
        "assignments": int(total_assignments),
        "mean_max_share": sum(max_shares) / len(max_shares),
        "mean_normalized_entropy": sum(normalized_entropies) / len(normalized_entropies),
        "dead_experts": dead_total,
    }


class RouterAccumulator:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] | None = None

    def add(self, entries: Sequence[dict[str, Any]]) -> None:
        if not entries:
            return
        if self.entries is None:
            self.entries = [
                {
                    "name": entry["name"],
                    "layer": entry["layer"],
                    "n_experts": entry["n_experts"],
                    "counts": entry["counts"].clone(),
                }
                for entry in entries
            ]
            return
        if len(self.entries) != len(entries):
            return
        for destination, source in zip(self.entries, entries):
            destination["counts"] += source["counts"]

    def summary(self) -> dict[str, Any] | None:
        if not self.entries:
            return None
        return summarize_router(self.entries)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _ids_list(ids: Iterable[int] | torch.Tensor) -> list[int]:
    if torch.is_tensor(ids):
        return [int(x) for x in ids.detach().cpu().tolist()]
    return [int(x) for x in ids]


def evaluate_ids(
    model: V3LM,
    token_ids: Iterable[int] | torch.Tensor,
    cfg: V3Config,
    *,
    ctx: int,
    batch_size: int,
    batches: int,
    seed: int = DEFAULT_SEED,
    device: str | torch.device = "cpu",
    name: str = "corpus",
    router_accumulator: RouterAccumulator | None = None,
) -> dict[str, Any]:
    """Evaluate a model on fixed positions drawn from ``token_ids``."""

    ids = _ids_list(token_ids)
    horizon = max(1, int(cfg.mtp))
    positions, position_meta = make_positions(
        len(ids), int(ctx), int(batch_size) * int(batches), horizon, seed, name
    )
    torch_device = torch.device(device)
    was_training = model.training

    loss_sum = 0.0
    correct = 0
    top5_correct = 0
    total = 0
    mtp_stats: dict[int, dict[str, float]] = {}
    aux_lb_sum = 0.0
    aux_lb_tokens = 0
    aux_z_sum = 0.0
    aux_z_tokens = 0

    tracker = RouterTracker(model)
    started = time.time()
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(positions), batch_size):
                batch_positions = positions[start : start + batch_size]
                b = len(batch_positions)
                xb = torch.tensor(
                    [ids[p - ctx : p] for p in batch_positions],
                    dtype=torch.long,
                    device=torch_device,
                )
                yb = torch.tensor(
                    [ids[p] for p in batch_positions],
                    dtype=torch.long,
                    device=torch_device,
                )

                logits, mtp_logits, aux = model(xb)
                logits = logits[:, -1, :]
                loss = F.cross_entropy(logits, yb, reduction="sum")
                loss_sum += float(loss.detach().cpu())
                total += b

                correct += int((logits.argmax(dim=-1) == yb).sum().item())
                topk = min(5, logits.shape[-1])
                top5 = logits.topk(topk, dim=-1).indices
                top5_correct += int(
                    (top5 == yb.unsqueeze(-1)).any(dim=-1).sum().item()
                )

                if isinstance(mtp_logits, torch.Tensor):
                    heads = [mtp_logits]
                else:
                    heads = list(mtp_logits or [])
                for head_index, head in enumerate(heads):
                    offset = head_index + 2
                    if offset > cfg.mtp:
                        break
                    head_logits = head[:, -1, :] if head.dim() == 3 else head
                    head_targets = torch.tensor(
                        [ids[p + 1 + head_index] for p in batch_positions],
                        dtype=torch.long,
                        device=torch_device,
                    )
                    head_loss = F.cross_entropy(head_logits, head_targets, reduction="sum")
                    stat = mtp_stats.setdefault(
                        offset, {"loss_sum": 0.0, "correct": 0.0, "tokens": 0.0}
                    )
                    stat["loss_sum"] += float(head_loss.detach().cpu())
                    stat["correct"] += float(
                        (head_logits.argmax(dim=-1) == head_targets).sum().item()
                    )
                    stat["tokens"] += b

                if isinstance(aux, dict):
                    load_balance = aux.get("load_balance")
                    if load_balance is not None:
                        aux_lb_sum += float(load_balance.detach().cpu()) * b
                        aux_lb_tokens += b
                    router_z = aux.get("router_z")
                    if router_z is not None:
                        aux_z_sum += float(router_z.detach().cpu()) * b
                        aux_z_tokens += b
    finally:
        router_summary = tracker.summary()
        if router_accumulator is not None:
            router_accumulator.add(tracker.entries)
        tracker.close()
        model.train(was_training)

    if total < 1:
        raise RuntimeError("evaluation produced no tokens")

    mean_loss = loss_sum / total
    mtp_items = []
    mtp_top1: dict[str, float] = {}
    for offset in sorted(mtp_stats):
        stat = mtp_stats[offset]
        tokens = int(stat["tokens"])
        item = {
            "offset": offset,
            "tokens": tokens,
            "loss": stat["loss_sum"] / max(1, tokens),
            "bits_per_token": (stat["loss_sum"] / max(1, tokens)) / math.log(2),
            "top1": stat["correct"] / max(1, tokens),
        }
        mtp_items.append(item)
        mtp_top1[f"t+{offset}"] = item["top1"]

    metrics: dict[str, Any] = {
        "name": name,
        "tokens": total,
        "tokens_available": len(ids),
        "loss": mean_loss,
        "bits_per_token": mean_loss / math.log(2),
        "top1": correct / total,
        "top5": top5_correct / total,
        "mtp": mtp_items,
        "mtp_top1": mtp_top1,
        "router_utilization": router_summary,
        "aux_load_balance": aux_lb_sum / aux_lb_tokens if aux_lb_tokens else None,
        "aux_router_z": aux_z_sum / aux_z_tokens if aux_z_tokens else None,
        "val_positions": position_meta,
        "eval_seconds": round(time.time() - started, 4),
    }
    return metrics


def _weighted(entries: Sequence[dict[str, Any]], key: str) -> float | None:
    pairs = [(float(m[key]), int(m["tokens"])) for m in entries if m.get(key) is not None]
    if not pairs:
        return None
    total = sum(weight for _, weight in pairs)
    if total <= 0:
        return None
    return sum(value * weight for value, weight in pairs) / total


def aggregate_metrics(
    corpora: dict[str, dict[str, Any]],
    router_accumulator: RouterAccumulator,
) -> dict[str, Any]:
    entries = list(corpora.values())
    total = sum(int(metrics["tokens"]) for metrics in entries)
    aggregate: dict[str, Any] = {"tokens": total}
    for key in ("loss", "bits_per_token", "top1", "top5", "aux_load_balance", "aux_router_z"):
        aggregate[key] = _weighted(entries, key)

    offsets = sorted({item["offset"] for m in entries for item in m.get("mtp", [])})
    mtp_items = []
    mtp_top1 = {}
    for offset in offsets:
        stats = []
        for metrics in entries:
            for item in metrics.get("mtp", []):
                if item["offset"] == offset:
                    stats.append(item)
                    break
        tokens = sum(int(item["tokens"]) for item in stats)
        if tokens < 1:
            continue
        item = {
            "offset": offset,
            "tokens": tokens,
            "loss": sum(float(s["loss"]) * int(s["tokens"]) for s in stats) / tokens,
            "bits_per_token": sum(
                float(s["bits_per_token"]) * int(s["tokens"]) for s in stats
            )
            / tokens,
            "top1": sum(float(s["top1"]) * int(s["tokens"]) for s in stats) / tokens,
        }
        mtp_items.append(item)
        mtp_top1[f"t+{offset}"] = item["top1"]

    aggregate["mtp"] = mtp_items
    aggregate["mtp_top1"] = mtp_top1
    aggregate["router_utilization"] = router_accumulator.summary()
    return aggregate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _smoke_config(ctx_override: int) -> V3Config:
    ctx = ctx_override if ctx_override > 0 else 32
    ctx = max(8, min(int(ctx), 64))
    return V3Config(
        vocab_size=128,
        ctx=ctx,
        d_model=64,
        n_layers=4,
        n_heads=4,
        swiglu=128,
        window=min(16, ctx),
        global_every=2,
        moe_layers=2,
        n_experts=8,
        topk=2,
        shared_experts=1,
        expert_hidden=128,
        mtp=3,
        dropout=0.0,
        rope_base=10000.0,
    )


def _parse_corpus_specs(specs: Sequence[str] | Sequence[Sequence[str]]) -> list[tuple[str, Path]]:
    flat: list[str] = []
    for entry in specs:
        if isinstance(entry, (list, tuple)):
            flat.extend(str(item) for item in entry)
        else:
            flat.append(str(entry))

    corpora: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for spec in flat:
        path = Path(spec).expanduser()
        if "=" in spec and not path.is_file():
            name, raw_path = spec.split("=", 1)
            path = Path(raw_path).expanduser()
        else:
            name = path.stem
        name = name.strip()
        if not name:
            raise ValueError(f"empty corpus name in {spec!r}")
        if name in seen:
            raise ValueError(f"duplicate corpus name {name!r}")
        seen.add(name)
        corpora.append((name, path))
    return corpora


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a z3 V3LM checkpoint on held-out corpora.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default="", help="v3 checkpoint (.pt)")
    parser.add_argument(
        "--config",
        default="",
        help="model config JSON; used when the checkpoint has no config",
    )
    parser.add_argument(
        "--corpus",
        action="append",
        nargs="+",
        default=[],
        metavar="[NAME=]PATH",
        help="one or more corpus text files; repeat for a per-corpus breakdown",
    )
    parser.add_argument(
        "--tokenizer",
        choices=("auto", "v4", "layered", "bpe", "bytes"),
        default="auto",
        help="v4 is bundled; layered uses the tuned six-level artifact",
    )
    parser.add_argument("--bpe-model", default="", help="BPE tokenizer JSON")
    parser.add_argument(
        "--layered-model",
        default=str(LAYERED_TOKENIZER_PATH),
        help="tuned layered tokenizer artifact",
    )
    parser.add_argument("--ctx", type=int, default=0, help="context length; 0 uses the model config")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--out", default="eval_v3.json", help="output JSON path")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="validation position seed")
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=0,
        help="maximum bytes read from each corpus; 0 reads the whole file",
    )
    parser.add_argument(
        "--random-smoke",
        action="store_true",
        help="instantiate a tiny V3LM and evaluate random batches; no checkpoint needed",
    )
    return parser


def _build_result(
    *,
    mode: str,
    cfg: V3Config,
    model: V3LM,
    device: torch.device,
    ctx: int,
    batch_size: int,
    batches: int,
    seed: int,
    checkpoint: str | None,
    checkpoint_step: int | None,
    config_source: str | None,
    tokenizer: str | None,
    corpora: dict[str, dict[str, Any]],
    router_accumulator: RouterAccumulator,
    wall_seconds: float,
) -> dict[str, Any]:
    aggregate = aggregate_metrics(corpora, router_accumulator)
    try:
        parameter_count = int(model.parameter_count())
    except AttributeError:  # pragma: no cover - generic fallback
        parameter_count = sum(p.numel() for p in model.parameters())
    try:
        dtype = str(next(model.parameters()).dtype)
    except StopIteration:  # pragma: no cover
        dtype = "unknown"

    result: dict[str, Any] = {
        "tool": "z3/eval_v3",
        "version": VERSION,
        "mode": mode,
        "checkpoint": checkpoint,
        "checkpoint_step": checkpoint_step,
        "config_source": config_source,
        "config": asdict(cfg),
        "parameter_count": parameter_count,
        "dtype": dtype,
        "device": str(device),
        "ctx": ctx,
        "model_ctx": cfg.ctx,
        "batch_size": batch_size,
        "batches": batches,
        "seed": seed,
        "tokenizer": tokenizer,
        "wall_seconds": round(wall_seconds, 4),
    }
    result.update(aggregate)
    result["corpora"] = corpora
    return result


def _print_summary(result: dict[str, Any], out_path: Path) -> None:
    def fmt(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.4f}"

    mtp = ", ".join(
        f"t+{item['offset']}={item['top1']:.4f}" for item in result.get("mtp", [])
    )
    print(
        f"eval_v3 [{result['mode']}]: tokens={result['tokens']} "
        f"bits/token={fmt(result['bits_per_token'])} "
        f"top1={fmt(result['top1'])} top5={fmt(result['top5'])}"
        + (f" mtp[{mtp}]" if mtp else ""),
        flush=True,
    )
    for name, metrics in result["corpora"].items():
        print(
            f"  {name}: tokens={metrics['tokens']} "
            f"bits/token={fmt(metrics['bits_per_token'])} "
            f"top1={fmt(metrics['top1'])} top5={fmt(metrics['top5'])}",
            flush=True,
        )
    print(f"wrote {out_path}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args, parser)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))
        raise AssertionError("unreachable")  # pragma: no cover


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.batches < 1:
        raise ValueError("--batches must be >= 1")
    if args.max_bytes < 0:
        raise ValueError("--max-bytes must be >= 0")
    device = _resolve_device(args.device)
    started = time.time()

    if args.random_smoke:
        cfg = _smoke_config(args.ctx)
        torch.manual_seed(args.seed)
        model = V3LM(cfg).to(device)
        model.eval()
        batches = min(args.batches, 3)
        ctx = cfg.ctx
        horizon = max(1, cfg.mtp)
        n_tokens = ctx + args.batch_size * batches + horizon + 8
        rng = random.Random(args.seed)
        ids = [rng.randrange(cfg.vocab_size) for _ in range(n_tokens)]
        router_accumulator = RouterAccumulator()
        metrics = evaluate_ids(
            model,
            ids,
            cfg,
            ctx=ctx,
            batch_size=args.batch_size,
            batches=batches,
            seed=args.seed,
            device=device,
            name="random-smoke",
            router_accumulator=router_accumulator,
        )
        corpora = {"random-smoke": metrics}
        result = _build_result(
            mode="random-smoke",
            cfg=cfg,
            model=model,
            device=device,
            ctx=ctx,
            batch_size=args.batch_size,
            batches=batches,
            seed=args.seed,
            checkpoint=None,
            checkpoint_step=None,
            config_source="random-smoke",
            tokenizer="random",
            corpora=corpora,
            router_accumulator=router_accumulator,
            wall_seconds=time.time() - started,
        )
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required unless --random-smoke is used")
        if not args.corpus:
            parser.error("at least one --corpus [NAME=]PATH is required")
        model, cfg, metadata = load_checkpoint(
            args.checkpoint, config_json=args.config or None
        )
        model = model.to(device)
        ctx = args.ctx if args.ctx > 0 else cfg.ctx
        tokenizer_kind = args.tokenizer or ("bpe" if args.bpe_model else "bytes")
        tokenizer = load_tokenizer(
            tokenizer_kind,
            args.bpe_model or None,
            args.layered_model or None,
        )
        specs = _parse_corpus_specs(args.corpus)

        router_accumulator = RouterAccumulator()
        corpora = {}
        for name, path in specs:
            ids = encode_file(path, tokenizer, args.max_bytes)
            if not ids:
                raise ValueError(f"corpus {name!r} ({path}) produced no tokens")
            largest = max(ids)
            if largest >= cfg.vocab_size:
                raise ValueError(
                    f"corpus {name!r} produced token id {largest}, but the model "
                    f"vocab_size is {cfg.vocab_size}; tokenizer/model mismatch"
                )
            metrics = evaluate_ids(
                model,
                ids,
                cfg,
                ctx=ctx,
                batch_size=args.batch_size,
                batches=args.batches,
                seed=args.seed,
                device=device,
                name=name,
                router_accumulator=router_accumulator,
            )
            metrics["path"] = str(path)
            metrics["tokenizer"] = tokenizer.kind
            metrics["tokenizer_backend"] = getattr(tokenizer, "backend", "builtin")
            metrics["tokenizer_vocab_size"] = tokenizer.vocab_size
            raw_bytes = path.stat().st_size
            if args.max_bytes > 0:
                raw_bytes = min(raw_bytes, args.max_bytes)
            bytes_per_token = raw_bytes / max(1, len(ids))
            metrics["raw_bytes"] = raw_bytes
            metrics["encoded_tokens"] = len(ids)
            metrics["bytes_per_token"] = bytes_per_token
            metrics["bits_per_byte"] = (
                metrics["bits_per_token"] / bytes_per_token
            )
            corpora[name] = metrics

        result = _build_result(
            mode="checkpoint",
            cfg=cfg,
            model=model,
            device=device,
            ctx=ctx,
            batch_size=args.batch_size,
            batches=args.batches,
            seed=args.seed,
            checkpoint=str(Path(args.checkpoint).expanduser()),
            checkpoint_step=metadata.get("step"),
            config_source=metadata.get("config_source"),
            tokenizer=tokenizer_kind,
            corpora=corpora,
            router_accumulator=router_accumulator,
            wall_seconds=time.time() - started,
        )

    out_path = Path(args.out).expanduser()
    if out_path.parent != Path(""):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    _print_summary(result, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
