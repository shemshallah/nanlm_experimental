#!/usr/bin/env python3
"""Factorized C(V,2) / C(V,3) hyperdimensional object codebook.

The diamond engine addresses all pairs/triples over 256 vertices.  Storing a
full FFN per object is impossible (2.8M objects).  This module stores one HD
hypervector per vertex and *binds* vertices to address pair/triple objects on
demand.  Growing V adds only V new hypervectors while the object address space
grows quadratically/cubically.

    pair(i,j)   = normalize(v_i * v_j)
    triple(i,j,k) = normalize(v_i * v_j * v_k)

`combinadic_rank` gives every sorted object a stable integer address, matching
the exact C(V,2)/C(V,3) numbering used by the diamond work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "CombinatorialCodebook",
    "binomial",
    "combinadic_rank",
    "combinadic_unrank",
    "object_counts",
]


def binomial(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def combinadic_rank(indices: tuple[int, ...], vertices: int) -> int:
    """Lexicographic rank of a sorted combination."""
    values = tuple(sorted(int(x) for x in indices))
    if len(set(values)) != len(values):
        raise ValueError("indices must be distinct")
    if values and (values[0] < 0 or values[-1] >= vertices):
        raise ValueError("index outside vertex range")
    return sum(binomial(value, rank + 1) for rank, value in enumerate(values))


def combinadic_unrank(rank: int, k: int, vertices: int) -> tuple[int, ...]:
    """Inverse of :func:`combinadic_rank`."""
    if rank < 0 or rank >= binomial(vertices, k):
        raise ValueError("rank outside combination space")
    result: list[int] = []
    remaining = int(rank)
    for i in range(k, 0, -1):
        # Find the largest c with C(c, i) <= remaining.
        lo, hi = i - 1, vertices - 1
        chosen = lo
        while lo <= hi:
            mid = (lo + hi) // 2
            if binomial(mid, i) <= remaining:
                chosen = mid
                lo = mid + 1
            else:
                hi = mid - 1
        result.append(chosen)
        remaining -= binomial(chosen, i)
    return tuple(reversed(result))


def object_counts(vertices: int) -> dict[str, int]:
    return {
        "vertices": int(vertices),
        "pairs": binomial(vertices, 2),
        "triples": binomial(vertices, 3),
        "pairs_and_triples": binomial(vertices, 2) + binomial(vertices, 3),
    }


def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise binding for real/bipolar hypervectors."""
    return a * b


def bundle(*vectors: torch.Tensor) -> torch.Tensor:
    """Superposition followed by normalization."""
    return F.normalize(torch.stack(vectors, dim=0).sum(dim=0), dim=-1)


def permute(vector: torch.Tensor, permutation: torch.Tensor, steps: int = 1) -> torch.Tensor:
    out = vector
    for _ in range(max(1, int(steps))):
        out = out[..., permutation]
    return out


class CombinatorialMemory(nn.Module):
    """Sparse pair-object memory over a growable vertex set.

    For each token it scores all vertices, keeps the top vertices, forms the
    candidate pair objects among them, and retrieves a bound value code.
    Object addresses are still exact ``C(V,2)``; only the top candidates are
    materialized.  Growth adds vertex key/value hypervectors, not dense
    per-object FFNs.
    """

    def __init__(
        self,
        d_model: int,
        vertices: int = 64,
        dim: int = 128,
        *,
        top_vertices: int = 16,
        top_pairs: int = 8,
        seed: int = 20261113,
    ) -> None:
        super().__init__()
        if vertices < 2 or top_vertices < 2:
            raise ValueError("vertices/top_vertices must be >= 2")
        if top_vertices > vertices:
            raise ValueError("top_vertices cannot exceed vertices")
        if top_pairs < 1:
            raise ValueError("top_pairs must be positive")
        self.d_model = int(d_model)
        self.dim = int(dim)
        self.top_vertices = int(top_vertices)
        self.top_pairs = int(top_pairs)
        gen = torch.Generator().manual_seed(seed)
        self.vertex_keys = nn.Parameter(
            F.normalize(torch.randn(vertices, dim, generator=gen), dim=-1)
        )
        self.vertex_values = nn.Parameter(
            F.normalize(torch.randn(vertices, dim, generator=gen), dim=-1)
        )
        self.query_proj = nn.Linear(d_model, dim, bias=False)
        self.out_proj = nn.Linear(dim, d_model, bias=False)
        self.gate_logit = nn.Parameter(torch.tensor(-4.0))

    @property
    def vertices(self) -> int:
        return int(self.vertex_keys.shape[0])

    def counts(self) -> dict[str, int]:
        return object_counts(self.vertices)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        b, t, d = hidden.shape
        if d != self.d_model:
            raise ValueError("hidden width does not match d_model")
        q = F.normalize(self.query_proj(hidden), dim=-1)  # (B,T,H)
        vertex_scores = torch.einsum("bth,vh->btv", q, self.vertex_keys)
        top_scores, top_ids = vertex_scores.topk(self.top_vertices, dim=-1)
        keys = F.normalize(self.vertex_keys[top_ids], dim=-1)
        values = F.normalize(self.vertex_values[top_ids], dim=-1)
        # Candidate pair codes among the selected vertices.
        pair_keys = F.normalize(
            keys.unsqueeze(3) * keys.unsqueeze(2), dim=-1
        )  # (B,T,Vc,Vc,H)
        pair_values = F.normalize(
            values.unsqueeze(3) * values.unsqueeze(2), dim=-1
        )
        pair_scores = torch.einsum("bth,btuvh->btuv", q, pair_keys)
        mask = torch.eye(
            self.top_vertices, device=hidden.device, dtype=torch.bool
        )
        pair_scores = pair_scores.masked_fill(mask, float("-inf"))
        flat_scores = pair_scores.reshape(b, t, -1)
        k = min(self.top_pairs, flat_scores.shape[-1])
        pair_weights, pair_idx = flat_scores.topk(k, dim=-1)
        pair_weights = F.softmax(pair_weights, dim=-1)
        flat_values = pair_values.reshape(b, t, -1, self.dim)
        selected = torch.gather(
            flat_values,
            2,
            pair_idx.unsqueeze(-1).expand(-1, -1, -1, self.dim),
        )
        memory = (selected * pair_weights.unsqueeze(-1)).sum(dim=2)
        out = self.out_proj(memory)
        return torch.sigmoid(self.gate_logit) * out

    @torch.no_grad()
    def grow(self, new_vertices: int, *, seed: int | None = None) -> "CombinatorialMemory":
        if new_vertices <= 0:
            return self
        old_k = self.vertex_keys.detach()
        old_v = self.vertex_values.detach()
        gen = torch.Generator().manual_seed(
            seed if seed is not None else 20261113 + self.vertices
        )
        extra_k = F.normalize(
            torch.randn(new_vertices, self.dim, generator=gen), dim=-1
        ).to(old_k.device, old_k.dtype)
        extra_v = F.normalize(
            torch.randn(new_vertices, self.dim, generator=gen), dim=-1
        ).to(old_v.device, old_v.dtype)
        self.vertex_keys = nn.Parameter(torch.cat([old_k, extra_k], dim=0))
        self.vertex_values = nn.Parameter(torch.cat([old_v, extra_v], dim=0))
        return self

    def manifest(self) -> dict[str, int | str]:
        return {
            "vertices": self.vertices,
            "dim": self.dim,
            "top_vertices": self.top_vertices,
            "top_pairs": self.top_pairs,
            "pairs": binomial(self.vertices, 2),
            "triples": binomial(self.vertices, 3),
        }


class CombinatorialCodebook(nn.Module):
    """Growable HD object address space with exact pair/triple IDs."""

    def __init__(
        self,
        vertices: int = 256,
        dim: int = 256,
        *,
        seed: int = 20261113,
    ) -> None:
        super().__init__()
        if vertices < 2:
            raise ValueError("vertices must be at least 2")
        if dim < 8:
            raise ValueError("dim must be at least 8")
        self.dim = int(dim)
        self.initial_vertices = int(vertices)
        generator = torch.Generator().manual_seed(seed)
        codes = F.normalize(
            torch.randn(vertices, dim, generator=generator), dim=-1
        )
        self.vertex_codes = nn.Parameter(codes)
        self.register_buffer(
            "vertex_permutation", torch.randperm(dim, generator=generator)
        )

    @property
    def vertices(self) -> int:
        return int(self.vertex_codes.shape[0])

    def counts(self) -> dict[str, int]:
        return object_counts(self.vertices)

    def pair(self, i: int, j: int) -> torch.Tensor:
        if i == j:
            raise ValueError("pair vertices must be distinct")
        return F.normalize(bind(self.vertex_codes[i], self.vertex_codes[j]), dim=-1)

    def triple(self, i: int, j: int, k: int) -> torch.Tensor:
        if len({i, j, k}) != 3:
            raise ValueError("triple vertices must be distinct")
        return F.normalize(
            bind(bind(self.vertex_codes[i], self.vertex_codes[j]),
                 self.vertex_codes[k]),
            dim=-1,
        )

    def pair_id(self, i: int, j: int) -> int:
        return combinadic_rank((i, j), self.vertices)

    def triple_id(self, i: int, j: int, k: int) -> int:
        return combinadic_rank((i, j, k), self.vertices)

    def pair_from_id(self, object_id: int) -> tuple[int, int]:
        return combinadic_unrank(object_id, 2, self.vertices)

    def triple_from_id(self, object_id: int) -> tuple[int, int, int]:
        return combinadic_unrank(object_id, 3, self.vertices)

    def future(self, vector: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """Permutation-based future rail for MTP time binding."""
        return permute(vector, self.vertex_permutation, steps)

    def score_pairs(self, query: torch.Tensor) -> torch.Tensor:
        """Scores for all C(V,2) pairs, shape (V*(V-1)/2,)."""
        q = F.normalize(query, dim=-1)
        codes = F.normalize(
            self.vertex_codes[:, None, :] * self.vertex_codes[None, :, :],
            dim=-1,
        )
        iu = torch.triu_indices(self.vertices, self.vertices, offset=1)
        return (codes[iu[0], iu[1]] @ q) * math.sqrt(self.dim)

    def score_triples_from_pairs(
        self,
        query: torch.Tensor,
        pair_ids: torch.Tensor,
        topk: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score triples extending a small candidate set of pair addresses.

        This avoids materializing C(V,3) when V is large.
        """
        q = F.normalize(query, dim=-1)
        pair_ids = pair_ids.reshape(-1)
        if pair_ids.numel() == 0:
            return torch.empty(0), torch.empty(0, dtype=torch.long)
        pairs = torch.stack(
            [torch.tensor(self.pair_from_id(int(x)), device=pair_ids.device)
             for x in pair_ids],
            dim=0,
        )
        triple_ids: list[int] = []
        scores: list[torch.Tensor] = []
        for i, j in pairs.tolist():
            for k in range(self.vertices):
                if k == i or k == j:
                    continue
                triple_ids.append(self.triple_id(i, j, k))
                scores.append(self.triple(i, j, k) @ q * math.sqrt(self.dim))
        if not scores:
            return torch.empty(0), torch.empty(0, dtype=torch.long)
        score_tensor = torch.stack(scores)
        k = min(int(topk), score_tensor.numel())
        values, indices = score_tensor.topk(k)
        ids = torch.tensor(triple_ids, device=pair_ids.device)[indices]
        return values, ids

    @torch.no_grad()
    def grow(self, new_vertices: int, *, init_scale: float = 0.02) -> "CombinatorialCodebook":
        """Append vertices while preserving every existing address/code."""
        if new_vertices <= 0:
            return self
        old = self.vertex_codes.detach()
        generator = torch.Generator().manual_seed(
            20261113 + int(old.shape[0])
        )
        extra = F.normalize(
            torch.randn(new_vertices, self.dim, generator=generator)
            * max(1e-6, float(init_scale)),
            dim=-1,
        ).to(device=old.device, dtype=old.dtype)
        self.vertex_codes = nn.Parameter(torch.cat([old, extra], dim=0))
        return self

    def manifest(self) -> dict[str, int | str]:
        return {
            "vertices": self.vertices,
            "dim": self.dim,
            "pairs": binomial(self.vertices, 2),
            "triples": binomial(self.vertices, 3),
            "object_storage": "vertex_codes + on-demand binding",
        }
