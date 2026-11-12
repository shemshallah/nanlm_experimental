#!/usr/bin/env python3
"""v3 memory path: hashed n-gram retrieval + learned memory cross-attention.

Two complementary memory mechanisms:

  * ``HashedNgramMemory`` -- a top-1 4-gram store with 64-bit collision tags,
    built directly from token IDs.  It never materialises a (size, vocab)
    table: one ``next_token`` and one count per slot is enough for a sparse
    next-token bias.
  * ``MemoryCrossAttention`` -- a small learned global memory tensor
    ``(n_slots, d_model)`` that every position cross-attends to with its own
    query.  Because the keys/values are position-independent and each query
    comes from a single hidden position, the block is causal by construction:
    future tokens cannot influence past outputs.

``MemoryBias`` bridges the two by turning n-gram counts into a sparse logit
bias with a learnable scale.  ``V3MemoryPath`` wires the pieces together for
the v3 model without touching ``v3_model.py``.

This is an architecture prototype, not a trained frontier model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

_MASK64 = (1 << 64) - 1
_FNV_OFFSET = 1469598103934665603
_FNV_PRIME = 1099511628211
_GOLDEN = 0x9E3779B97F4A7C15


def _as_1d_int64(tokens, name: str = "tokens") -> np.ndarray:
    """Accept a list/ndarray/torch tensor of token IDs and return 1-D int64."""
    if isinstance(tokens, torch.Tensor):
        arr = tokens.detach().cpu().numpy()
    else:
        arr = np.asarray(tokens)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        return np.zeros(0, dtype=np.int64)
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"{name} must contain integer token IDs, got {arr.dtype}")
    arr = arr.astype(np.int64, copy=False)
    if int(arr.min()) < 0:
        raise ValueError(f"{name} must contain non-negative token IDs")
    return arr


@dataclass(frozen=True)
class MemoryHit:
    """Raw result of a single top-1 n-gram lookup."""

    hit: bool
    next_token: int = -1
    count: int = 0
    key: int = 0
    bucket: int = -1


class HashedNgramMemory:
    """Top-1 n-gram memory with 64-bit collision tags.

    Each slot stores one full 64-bit context hash, its most common observed
    successor, and how often that successor followed the context.  Colliding
    contexts are rejected through the full tag; no dense ``(size, vocab)``
    table is allocated.

    Args:
        size: number of slots (default 1M).
        order: context length in tokens (default 4-gram).
        min_count: minimum pair count accepted by :meth:`bias` consumers.
        chunk_windows: upper bound on windows processed at once while building.
    """

    def __init__(self, size: int = 1 << 20, order: int = 4, min_count: int = 5,
                 chunk_windows: int = 1 << 18):
        if int(size) <= 0:
            raise ValueError("size must be positive")
        if int(order) <= 0:
            raise ValueError("order must be positive")
        if int(min_count) < 0:
            raise ValueError("min_count must be non-negative")
        self.size = int(size)
        self.order = int(order)
        self.min_count = int(min_count)
        self.chunk_windows = int(chunk_windows)

        self.keys = np.zeros(self.size, dtype=np.uint64)
        self.next_tok = np.zeros(self.size, dtype=np.int64)
        self.counts = np.zeros(self.size, dtype=np.int64)
        self.occupied = np.zeros(self.size, dtype=bool)

        self.collisions = 0
        self.n_windows = 0

    # ------------------------------------------------------------------
    # Hashing (64-bit FNV-1a over the context tokens)
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_context_int(context) -> int:
        h = _FNV_OFFSET
        for value in context:
            h ^= (int(value) + _GOLDEN) & _MASK64
            h = (h * _FNV_PRIME) & _MASK64
        return h

    @classmethod
    def _hash_rows(cls, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        out = np.full(rows.shape[0], np.uint64(_FNV_OFFSET), dtype=np.uint64)
        for j in range(rows.shape[1]):
            out ^= rows[:, j].astype(np.uint64) + np.uint64(_GOLDEN)
            out *= np.uint64(_FNV_PRIME)
        return out

    # ------------------------------------------------------------------
    # Build / query
    # ------------------------------------------------------------------

    def build(self, tokens, max_tokens: int | None = None) -> "HashedNgramMemory":
        """Populate the store from a token-ID sequence.

        Only the first successor seen for a context is kept (top-1).  Later
        occurrences of that same successor increase its count; other
        successors for the same context are ignored.  A context that hashes
        to an occupied slot with a different tag is a collision and never
        produces a false hit.
        """
        tokens = _as_1d_int64(tokens)
        windows = len(tokens) - self.order
        if windows <= 0:
            return self
        if max_tokens is not None:
            windows = min(windows, max(0, int(max_tokens)))
        if windows <= 0:
            return self
        self.n_windows += int(windows)

        for start in range(0, windows, self.chunk_windows):
            stop = min(windows, start + self.chunk_windows)
            n = stop - start
            segment = tokens[start:start + n + self.order - 1]
            hashes = self._hash_rows(
                np.lib.stride_tricks.sliding_window_view(segment, self.order)
            )
            targets = tokens[start + self.order:stop + self.order]
            buckets = (hashes % np.uint64(self.size)).astype(np.int64)

            # First writer wins within an empty bucket.
            empty = ~self.occupied[buckets]
            if empty.any():
                empty_idx = np.flatnonzero(empty)
                unique_buckets, first = np.unique(buckets[empty_idx], return_index=True)
                chosen = empty_idx[first]
                self.keys[unique_buckets] = hashes[chosen]
                self.next_tok[unique_buckets] = targets[chosen]
                self.occupied[unique_buckets] = True

            occupied = self.occupied[buckets]
            key_match = occupied & (self.keys[buckets] == hashes)
            target_match = key_match & (targets == self.next_tok[buckets])
            if target_match.any():
                increments = np.bincount(
                    buckets[target_match], minlength=self.size
                ).astype(np.int64, copy=False)
                self.counts += increments

            self.collisions += int(np.count_nonzero(occupied & ~key_match))

        return self

    def lookup(self, context) -> MemoryHit:
        """Look up one context; the last ``order`` IDs are used if longer."""
        ctx = _as_1d_int64(context, name="context")
        if ctx.size < self.order:
            return MemoryHit(hit=False)
        ctx = ctx[-self.order:]
        key = self._hash_context_int(ctx)
        bucket = key % self.size
        if self.occupied[bucket] and int(self.keys[bucket]) == key:
            return MemoryHit(True, int(self.next_tok[bucket]),
                             int(self.counts[bucket]), key, bucket)
        return MemoryHit(False, key=key, bucket=bucket)

    def lookup_many(self, contexts) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised lookup for an ``(N, order)`` array of contexts.

        Returns ``(hit, next_token, count)`` arrays of length ``N``.  The
        caller is responsible for applying the ``min_count`` threshold.
        """
        ctx = np.asarray(contexts, dtype=np.int64)
        if ctx.ndim != 2 or ctx.shape[1] != self.order:
            raise ValueError(f"contexts must have shape (N, {self.order})")
        if ctx.shape[0] == 0:
            return (np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.int64))
        if int(ctx.min()) < 0:
            raise ValueError("contexts must contain non-negative token IDs")
        hashes = self._hash_rows(ctx)
        buckets = (hashes % np.uint64(self.size)).astype(np.int64)
        hit = self.occupied[buckets] & (self.keys[buckets] == hashes)
        next_tok = np.where(hit, self.next_tok[buckets], -1)
        count = np.where(hit, self.counts[buckets], 0)
        return hit, next_tok, count

    def bias(self, context, vocab_size: int, alpha: float = 1.0) -> np.ndarray:
        """Dense ``(vocab_size,)`` float32 bias for one context.

        This is a convenience for inference/probes.  ``MemoryBias`` returns
        the sparse form and never allocates the dense vocab vector.
        """
        out = np.zeros(int(vocab_size), dtype=np.float32)
        hit = self.lookup(context)
        if hit.hit and hit.count >= self.min_count and hit.next_token < vocab_size:
            out[hit.next_token] = np.float32(alpha * math.log(max(1, hit.count)))
        return out

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return int(np.count_nonzero(self.occupied))

    @property
    def fill_ratio(self) -> float:
        return len(self) / self.size

    def nbytes(self) -> int:
        """Bytes held by the store (no vocab-sized table)."""
        return sum(a.nbytes for a in (self.keys, self.next_tok,
                                      self.counts, self.occupied))

    def stats(self) -> dict:
        return {
            "size": self.size,
            "order": self.order,
            "min_count": self.min_count,
            "entries": len(self),
            "fill_ratio": self.fill_ratio,
            "windows": self.n_windows,
            "collisions": self.collisions,
            "nbytes": self.nbytes(),
        }


@dataclass
class SparseBias:
    """Sparse logit bias: ``index`` is (N, 3) = (batch, time, token)."""

    index: torch.LongTensor
    value: torch.Tensor
    shape: tuple[int, int, int]

    def __len__(self) -> int:
        return int(self.index.shape[0])

    def dense(self, dtype=None, device=None) -> torch.Tensor:
        """Materialise a ``(B, T, V)`` bias; only call this when needed."""
        dtype = self.value.dtype if dtype is None else dtype
        device = self.value.device if device is None else device
        out = torch.zeros(self.shape, dtype=dtype, device=device)
        if len(self):
            _, time, _ = self.shape
            index = self.index.to(device)
            flat = index[:, 0] * time + index[:, 1]
            out.view(-1, self.shape[-1]).index_put_(
                (flat, index[:, 2]),
                self.value.to(device=device, dtype=dtype),
                accumulate=True,
            )
        return out

    def add_(self, logits: torch.Tensor) -> torch.Tensor:
        """In-place ``logits += bias`` with an explicit shape check."""
        if tuple(logits.shape) != tuple(self.shape):
            raise ValueError(f"logits shape {tuple(logits.shape)} != {self.shape}")
        if len(self) == 0:
            return logits
        _, time, _ = self.shape
        index = self.index.to(logits.device)
        flat = index[:, 0] * time + index[:, 1]
        logits.view(-1, self.shape[-1]).index_put_(
            (flat, index[:, 2]), self.value.to(device=logits.device, dtype=logits.dtype),
            accumulate=True,
        )
        return logits


class MemoryBias(nn.Module):
    """Turns n-gram counts into a sparse logit bias with learnable ``alpha``.

    At position ``t`` the context is ``tokens[..., t-order+1 : t+1]``; the
    bias targets the token these IDs predict, i.e. position ``t``'s logits
    (the next-token distribution).  Future tokens are never read.
    """

    def __init__(self, memory: HashedNgramMemory, vocab_size: int,
                 alpha_init: float = 0.1, min_count: int | None = None):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.memory = memory
        self.vocab_size = int(vocab_size)
        self.min_count = memory.min_count if min_count is None else int(min_count)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, tokens: torch.Tensor) -> SparseBias:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.dim() != 2:
            raise ValueError("tokens must have shape (B, T)")
        batch, time = tokens.shape
        device = self.alpha.device
        empty = SparseBias(
            index=torch.zeros(0, 3, dtype=torch.long, device=device),
            value=self.alpha * torch.zeros(0, device=device),
            shape=(batch, time, self.vocab_size),
        )
        if time == 0 or batch == 0:
            return empty

        order = self.memory.order
        ids = tokens.detach().to("cpu", torch.long).numpy()
        # (T, order) context indices; negative entries are left padding.
        offsets = np.arange(time)[:, None] - (order - 1) + np.arange(order)[None, :]
        valid_t = (offsets >= 0).all(axis=1)
        safe = np.clip(offsets, 0, time - 1)
        contexts = ids[:, safe]                      # (B, T, order)
        flat_contexts = contexts.reshape(batch * time, order)
        hit, next_tok, count = self.memory.lookup_many(flat_contexts)

        valid = np.broadcast_to(valid_t, (batch, time)).reshape(-1)
        accepted = valid & hit & (count >= self.min_count) & (next_tok < self.vocab_size)
        selected = np.flatnonzero(accepted)
        if selected.size == 0:
            return empty

        rows = selected // time
        cols = selected % time
        index = torch.from_numpy(
            np.stack([rows, cols, next_tok[selected]], axis=1)
        ).to(dtype=torch.long, device=device)
        log_counts = torch.from_numpy(
            np.log(np.maximum(count[selected], 1)).astype(np.float32)
        ).to(device=device)
        value = self.alpha * log_counts
        return SparseBias(index=index, value=value,
                          shape=(batch, time, self.vocab_size))


class RMSNorm(nn.Module):
    """RMSNorm matching the v3 trunk implementation."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def _make_norm(kind: str, dim: int) -> nn.Module:
    kind = kind.lower()
    if kind in {"rmsnorm", "rms"}:
        return RMSNorm(dim)
    if kind in {"layernorm", "layer", "ln"}:
        return nn.LayerNorm(dim)
    raise ValueError(f"unknown norm kind: {kind!r}")


class MemoryCrossAttention(nn.Module):
    """Causal-safe cross-attention from hidden states to learned memory slots.

    The global memory is a learned ``(n_slots, d_model)`` tensor.  Each
    position forms its own query from the pre-normed hidden state, so there is
    no mixing across time: output at position ``t`` depends only on ``x[:, t]``
    and the (position-independent) memory.  Outputs are ``x + residual`` and
    aux carries the mean attention entropy over slots.
    """

    def __init__(self, d_model: int, n_heads: int, n_slots: int = 256,
                 dropout: float = 0.0, norm: str = "rmsnorm",
                 bias: bool = False, residual_scale_init: float = 1.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if n_slots <= 0:
            raise ValueError("n_slots must be positive")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.n_slots = int(n_slots)
        self.scale = self.head_dim ** -0.5

        self.norm = _make_norm(norm, self.d_model)
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.memory = nn.Parameter(torch.randn(self.n_slots, self.d_model) * 0.02)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        if x.dim() != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"x must have shape (B, T, {self.d_model})")
        batch, time, _ = x.shape
        heads, head_dim = self.n_heads, self.head_dim

        h = self.norm(x)
        q = self.q_proj(h).view(batch, time, heads, head_dim).transpose(1, 2)
        # Memory keys/values are shared across batch and time.
        k = self.k_proj(self.memory).view(self.n_slots, heads, head_dim)
        v = self.v_proj(self.memory).view(self.n_slots, heads, head_dim)
        k = k.permute(1, 0, 2).unsqueeze(0)          # (1, H, S, hd)
        v = v.permute(1, 0, 2).unsqueeze(0)          # (1, H, S, hd)

        # This (B, H, T, S) score tensor *is* the cross-attention; no T x T
        # mask or score matrix is built anywhere.
        logits = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(logits, dim=-1)
        entropy = -(attn * torch.log(attn.clamp_min(1e-9))).sum(-1)  # (B, H, T)

        y = attn @ v                                  # (B, H, T, hd)
        y = y.transpose(1, 2).reshape(batch, time, self.d_model)
        y = self.dropout(self.out_proj(y))
        out = x + self.residual_scale * y

        aux = {
            "memory_entropy": entropy.mean(),
            "memory_entropy_per_head": entropy.mean(dim=(0, 2)),
        }
        if return_attn:
            aux["memory_attn"] = attn
        return out, aux


class V3MemoryPath(nn.Module):
    """Combined v3 memory path: dense learned slots + sparse n-gram bias.

    ``build_memory`` fills the n-gram store; ``forward`` applies learned
    cross-attention and optionally returns the sparse logit bias for logits
    of shape ``(B, T, vocab_size)``.
    """

    def __init__(self, d_model: int, n_heads: int, vocab_size: int,
                 n_slots: int = 256, memory_size: int = 1 << 20, order: int = 4,
                 min_count: int = 5, alpha_init: float = 0.1,
                 dropout: float = 0.0, norm: str = "rmsnorm"):
        super().__init__()
        self.cross = MemoryCrossAttention(
            d_model, n_heads, n_slots=n_slots, dropout=dropout, norm=norm
        )
        self.ngram = HashedNgramMemory(
            size=memory_size, order=order, min_count=min_count
        )
        self.logit_bias = MemoryBias(
            self.ngram, vocab_size, alpha_init=alpha_init
        )

    def build_memory(self, tokens, max_tokens: int | None = None) -> "V3MemoryPath":
        self.ngram.build(tokens, max_tokens=max_tokens)
        return self

    def forward(self, x: torch.Tensor, tokens: torch.Tensor | None = None,
                return_attn: bool = False):
        out, aux = self.cross(x, return_attn=return_attn)
        if tokens is not None:
            aux["logit_bias"] = self.logit_bias(tokens)
        return out, aux
