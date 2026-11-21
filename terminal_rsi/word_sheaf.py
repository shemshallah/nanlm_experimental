#!/usr/bin/env python3
"""Source-aware word-expert fusion with cellular sheaves and Ising/DPP selection.

This module is deliberately dependency-light: it only needs NumPy.  It is a
selector/fusion layer for the cached word expert scores used by the terminal
RSI word experiments.  The cache produced by ``word_model_rerank.py`` has::

    head, bridge, prior : (n_records, 4096) float16 logits
    ppm, cache, copy    : (n_records, 100) int32 rankings (-1 padded)
    target, book, prev  : (n_records,), (n_records,), (n_records, 3)

The main pieces are:

``CellularSheafFusion``
    Vertices are experts.  Each vertex carries a 4096-dimensional stalk of
    log-probabilities/logits.  Restrictions to an edge stalk are affine
    linear maps ``F(x) = s * x + U @ (V.T @ x)`` (scale plus low rank).  The
    sheaf Laplacian is ``L = B.T W B`` and the fused section is the harmonic
    extension solving ``(M + lambda * L) p = M p0``.  The returned H1
    disagreement is the weighted RMS coboundary on the graph's cycle edges
    (a 3-cycle has a non-trivial cycle space; a tree returns zero).

``IsingDPPSelect``
    Reranks a shortlist using an Ising-style field (base logits + PMI and
    context fields) and greedily selects a top-k set with either a DPP or an
    MMR diversity penalty over word embeddings.

``fit_sheaf_weights``
    Fits per-expert scalar temperatures and non-negative edge weights on a
    train split with a NumPy-only projected/Adam optimiser.  No sklearn.

CLI::

    python -m terminal_rsi.word_sheaf
    python -m terminal_rsi.word_sheaf --cache /marimo/word_dev_cache.npz

The CLI fits on books < 8 and reports true/conditional top-5 plus the mean H1
disagreement on books >= 8.  If the server cache is unavailable it prints a
clear message instead of inventing results.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

DEFAULT_EXPERTS: tuple[str, ...] = (
    "head", "bridge", "prior", "ppm", "cache", "copy", "hrr", "lzp", "stdp",
)
DEFAULT_CACHE = Path("/marimo/word_dev_cache.npz")
EPS = 1e-12

__all__ = [
    "CellularSheafFusion",
    "IsingDPPSelect",
    "AffineRestriction",
    "fit_sheaf_weights",
    "rankings_to_log_probs",
    "main",
]


def _as_numpy(x: Any, dtype: Any = None) -> np.ndarray:
    """Convert NumPy, CPU torch tensors, or array-like objects to NumPy."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=dtype)


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def _log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable log-softmax along ``axis``."""
    x = np.asarray(x, dtype=np.float64)
    z = x - np.max(x, axis=axis, keepdims=True)
    return z - np.log(np.sum(np.exp(z), axis=axis, keepdims=True) + EPS)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=axis, keepdims=True) + EPS)


def rankings_to_log_probs(
    rankings: np.ndarray,
    vocab_size: int,
    *,
    alpha: float = 1.0,
    floor: float = 1e-8,
) -> np.ndarray:
    """Turn padded integer rankings into dense log-probability rows.

    Parameters
    ----------
    rankings:
        ``(..., k)`` integer array.  Negative entries are padding.
    vocab_size:
        Size of the output vocabulary dimension.
    alpha:
        Power-law decay of the rank distribution: ``p_r ~ 1 / (r + 1) **
        alpha``.  ``alpha=1`` puts roughly 20% of the mass on rank 1 for a
        typical 100-word list, which is a sensible PPM/cache prior.
    floor:
        Probability floor for words absent from the ranking list.
    """
    rankings = _as_numpy(rankings)
    squeeze = rankings.ndim == 1
    if squeeze:
        rankings = rankings[None, :]
    if rankings.ndim != 2:
        raise ValueError("rankings must be 1-D or 2-D")
    n, k = rankings.shape
    v = int(vocab_size)
    probs = np.full((n, v), float(floor) / max(v, 1), dtype=np.float64)
    if k:
        ranks = np.arange(k, dtype=np.float64)
        weights = 1.0 / np.power(ranks + 1.0, float(alpha))
        for row in range(n):
            ids = rankings[row]
            valid = (ids >= 0) & (ids < v)
            if not np.any(valid):
                continue
            ids = ids[valid].astype(np.int64)
            w = weights[valid]
            # Keep the first (best) occurrence of a word.
            _, first = np.unique(ids, return_index=True)
            ids, w = ids[first], w[first]
            probs[row, ids] = w / np.sum(w)
    probs /= np.sum(probs, axis=1, keepdims=True) + EPS
    out = np.log(probs + EPS)
    return out[0] if squeeze else out


# ---------------------------------------------------------------------------
# Cellular sheaf fusion
# ---------------------------------------------------------------------------

@dataclass
class AffineRestriction:
    """A scaled low-rank restriction map ``F(x) = s*x + U (V.T x) + b``.

    The rank-``r`` term is the "affine" part used to make the sheaf non-trivial
    while keeping the restriction map cheap.  With ``U=None`` (or ``r=0``) the
    map is purely diagonal.
    """

    scale: float | np.ndarray
    U: np.ndarray | None = None
    V: np.ndarray | None = None
    bias: np.ndarray | None = None

    def apply(self, x: np.ndarray) -> np.ndarray:
        y = self.scale * x
        if self.U is not None and self.V is not None:
            y = y + (x @ self.V) @ self.U.T
        if self.bias is not None:
            y = y + self.bias
        return y

    def apply_transpose(self, y: np.ndarray) -> np.ndarray:
        x = self.scale * y
        if self.U is not None and self.V is not None:
            x = x + (y @ self.U) @ self.V.T
        return x

    def diagonal(self) -> np.ndarray:
        """Diagonal of ``F.T F`` (used as a Jacobi preconditioner)."""
        scale = np.asarray(self.scale, dtype=np.float64)
        if self.U is None or self.V is None:
            if scale.ndim == 0:
                return np.full(1, float(scale) ** 2)
            return scale ** 2
        u = np.asarray(self.U, dtype=np.float64)
        v = np.asarray(self.V, dtype=np.float64)
        d = u.shape[0]
        s = scale if scale.ndim == 0 else scale
        diag = np.full(d, float(s) ** 2 if scale.ndim == 0 else 0.0)
        if scale.ndim:
            diag = s ** 2
        diag = diag + 2.0 * s * np.einsum("jr,jr->j", u, v)
        gram = u.T @ u
        diag = diag + np.einsum("jr,rs,js->j", v, gram, v)
        return diag


def _resolve_endpoint(value: Any, index: Mapping[str, int], n: int) -> int:
    if isinstance(value, str):
        if value not in index:
            raise KeyError(f"unknown expert {value!r}; known: {list(index)}")
        return index[value]
    out = int(value)
    if out < 0 or out >= n:
        raise ValueError(f"expert index {out} out of range for {n} experts")
    return out


def _resolve_edges(
    edges: Any, experts: Sequence[str]
) -> tuple[list[tuple[int, int]], list[float]]:
    """Normalise an edge specification to index pairs and initial weights."""
    n = len(experts)
    index = {name: i for i, name in enumerate(experts)}
    if edges is None:
        edges = "cycle"
    if isinstance(edges, str):
        mode = edges.lower().strip()
        if mode in {"triangle", "tri", "3-cycle", "3cycle"}:
            if n < 3:
                raise ValueError("a 3-cycle needs at least three experts")
            raw = [(0, 1), (1, 2), (2, 0)]
        elif mode in {"cycle", "ring", "loop"}:
            if n < 2:
                raw = []
            else:
                raw = [(i, (i + 1) % n) for i in range(n if n > 2 else 1)]
        elif mode in {"complete", "full", "clique"}:
            raw = [(i, j) for i in range(n) for j in range(i + 1, n)]
        elif mode in {"path", "chain", "tree"}:
            raw = [(i, i + 1) for i in range(n - 1)]
        else:
            raise ValueError(f"unknown edge mode {edges!r}")
    else:
        if (isinstance(edges, tuple) and len(edges) == 2
                and not isinstance(edges[0], (list, tuple, np.ndarray))):
            raw = [edges]
        else:
            raw = list(edges)

    out: list[tuple[int, int]] = []
    weights: list[float] = []
    seen: set[tuple[int, int]] = set()
    for item in raw:
        if len(item) == 2:
            a, b = item
            w = 1.0
        elif len(item) == 3:
            a, b, w = item
        else:
            raise ValueError(f"edge must be (u, v) or (u, v, weight): {item!r}")
        u = _resolve_endpoint(a, index, n)
        v = _resolve_endpoint(b, index, n)
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        out.append((u, v))
        weights.append(float(w))
    return out, weights


def _cycle_edges(n: int, edges: Sequence[tuple[int, int]]) -> list[int]:
    """Return indices of edges that lie on at least one cycle.

    An edge is on a cycle iff its endpoints remain connected after removing
    it.  Since the expert graphs here are tiny this brute-force check is fine.
    """
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for ei, (u, v) in enumerate(edges):
        adjacency[u].append((v, ei))
        adjacency[v].append((u, ei))

    def reachable(src: int, dst: int, skip: int) -> bool:
        stack = [src]
        seen = {src}
        while stack:
            cur = stack.pop()
            if cur == dst:
                return True
            for nxt, ei in adjacency[cur]:
                if ei == skip or nxt in seen:
                    continue
                seen.add(nxt)
                stack.append(nxt)
        return False

    return [ei for ei, (u, v) in enumerate(edges) if reachable(u, v, ei)]


class CellularSheafFusion:
    """Harmonic-extension fusion over word experts.

    Parameters
    ----------
    experts:
        Vertex names.  Defaults to the nine word-expert names.
    stalk_dim:
        Stalk dimension (4096 for the word cache).
    edges:
        ``None``/``"cycle"``/``"triangle"``/``"complete"``/``"path"`` or a
        list of ``(u, v)`` / ``(u, v, weight)`` pairs.  Names or indices are
        accepted.  A 3-cycle is the recommended graph for H1 diagnostics.
    low_rank:
        Rank of the low-rank part of every restriction map.  ``0`` gives a
        scaled graph Laplacian with diagonal restrictions.
    scale:
        Restriction scale.  Scalar, ``(stalk_dim,)``, ``(n_edges,)`` or
        ``(n_edges, stalk_dim)``.
    low_rank_scale:
        Magnitude of the low-rank term relative to the identity term.
    mass:
        ``"uniform"``, ``"confidence"``, ``"entropy"``, an array of shape
        ``(n_experts,)``, or a per-sample ``(n, n_experts)`` array.
    expert_weights:
        Optional non-negative global aggregation weights (for example the
        ``expert_weights`` returned by :func:`fit_sheaf_weights`).  When set,
        the output is a mass-times-expert-weight mixture of the harmonic
        vertex signals.
    normalize:
        If True, each expert input is converted to log-probabilities with a
        log-softmax before fusion.
    """

    def __init__(
        self,
        experts: Sequence[str] | None = None,
        *,
        stalk_dim: int = 4096,
        edges: Any = None,
        low_rank: int = 4,
        scale: Any = 1.0,
        low_rank_scale: float = 0.02,
        edge_weight: Any = 1.0,
        laplacian_weight: float = 1.0,
        mass: Any = "uniform",
        mass_power: float = 1.0,
        mass_floor: float = 1e-6,
        expert_weights: Any = None,
        normalize: bool = True,
        directional: bool = False,
        seed: int = 0,
        max_iter: int = 60,
        tol: float = 1e-6,
        dtype: Any = np.float64,
    ) -> None:
        self.experts = list(DEFAULT_EXPERTS if experts is None else experts)
        if not self.experts:
            raise ValueError("at least one expert is required")
        if len(set(self.experts)) != len(self.experts):
            raise ValueError("expert names must be unique")
        self.stalk_dim = int(stalk_dim)
        if self.stalk_dim <= 0:
            raise ValueError("stalk_dim must be positive")
        self.index = {name: i for i, name in enumerate(self.experts)}
        self.edges, initial_weights = _resolve_edges(edges, self.experts)
        self.n_experts = len(self.experts)
        self.low_rank = max(0, int(low_rank))
        self.low_rank_scale = float(low_rank_scale)
        self.laplacian_weight = max(0.0, float(laplacian_weight))
        self.mass_mode = mass
        self.mass_power = float(mass_power)
        self.mass_floor = max(float(mass_floor), 1e-12)
        self.normalize = bool(normalize)
        self.directional = bool(directional)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.dtype = np.dtype(dtype)
        self.temperatures = np.ones(self.n_experts, dtype=np.float64)
        self.expert_weights: np.ndarray | None = None
        if expert_weights is not None:
            self.set_expert_weights(expert_weights)
        self.cycle_edges = _cycle_edges(self.n_experts, self.edges)

        # Edge weights: explicit sequence wins over the per-edge defaults.
        if np.ndim(edge_weight) == 0:
            weights = np.full(len(self.edges), float(edge_weight))
        else:
            weights = np.asarray(edge_weight, dtype=np.float64).reshape(-1)
            if weights.size != len(self.edges):
                raise ValueError(
                    f"edge_weight must have {len(self.edges)} entries")
        if weights.size == len(self.edges) and np.all(weights == 1.0):
            weights = np.asarray(initial_weights, dtype=np.float64)
        if np.any(weights < 0):
            raise ValueError("edge weights must be non-negative")
        self.edge_weights = weights

        self._rng = np.random.default_rng(seed)
        self.maps: dict[tuple[int, int], AffineRestriction] = {}
        self._build_restrictions(scale)
        self._diag_l = self._compute_diag_l()

    # -- construction -----------------------------------------------------

    def _scale_for(self, scale: Any, ei: int) -> Any:
        arr = np.asarray(scale, dtype=np.float64) if not np.isscalar(scale) \
            else None
        if arr is None:
            return float(scale)
        if arr.ndim == 0:
            return float(arr)
        if arr.shape == (self.stalk_dim,):
            return arr
        if arr.shape == (len(self.edges),):
            return float(arr[ei])
        if arr.shape == (len(self.edges), self.stalk_dim):
            return arr[ei]
        raise ValueError(
            "scale must be scalar, (stalk_dim,), (n_edges,) or "
            "(n_edges, stalk_dim)")

    def _make_map(self, scale: Any) -> AffineRestriction:
        if self.low_rank > 0:
            d = self.stalk_dim
            r = self.low_rank
            u = self._rng.normal(
                0.0, self.low_rank_scale / math.sqrt(max(d, 1)), (d, r))
            v = self._rng.normal(0.0, 1.0 / math.sqrt(r), (d, r))
            return AffineRestriction(scale=scale, U=u, V=v)
        return AffineRestriction(scale=scale)

    def _build_restrictions(self, scale: Any) -> None:
        for ei, (u, v) in enumerate(self.edges):
            s = self._scale_for(scale, ei)
            if self.directional:
                fu = self._make_map(s)
                fv = self._make_map(s)
            else:
                shared = self._make_map(s)
                fu = fv = shared
            self.maps[(u, ei)] = fu
            self.maps[(v, ei)] = fv

    def _compute_diag_l(self) -> np.ndarray:
        diag = np.zeros((self.n_experts, self.stalk_dim),
                        dtype=np.float64)
        for ei, (u, v) in enumerate(self.edges):
            w = float(self.edge_weights[ei])
            if w == 0.0:
                continue
            du = self.maps[(u, ei)].diagonal()
            dv = self.maps[(v, ei)].diagonal()
            if du.size == 1:
                du = np.full(self.stalk_dim, float(du[0]))
            if dv.size == 1:
                dv = np.full(self.stalk_dim, float(dv[0]))
            diag[u] += w * du
            diag[v] += w * dv
        return diag

    # -- public configuration ---------------------------------------------

    @property
    def restriction_maps(self) -> dict[tuple[int, int], AffineRestriction]:
        """Mapping ``(vertex, edge) -> AffineRestriction``."""
        return self.maps

    def set_edge_weights(self, weights: Sequence[float]) -> "CellularSheafFusion":
        weights = _as_numpy(weights, dtype=np.float64).reshape(-1)
        if weights.size != len(self.edges):
            raise ValueError(f"expected {len(self.edges)} weights")
        if np.any(weights < 0):
            raise ValueError("edge weights must be non-negative")
        self.edge_weights = weights
        self._diag_l = self._compute_diag_l()
        return self

    def set_temperatures(self, temperatures: Any) -> "CellularSheafFusion":
        temps = _as_numpy(temperatures, dtype=np.float64).reshape(-1)
        if temps.size == 1:
            temps = np.full(self.n_experts, float(temps[0]))
        if temps.size != self.n_experts:
            raise ValueError(f"expected {self.n_experts} temperatures")
        if np.any(temps <= 0):
            raise ValueError("temperatures must be positive")
        self.temperatures = temps
        return self

    def set_expert_weights(self, weights: Any) -> "CellularSheafFusion":
        """Set optional global expert aggregation weights."""
        w = _as_numpy(weights, dtype=np.float64).reshape(-1)
        if w.size == 1:
            w = np.full(self.n_experts, float(w[0]))
        if w.size != self.n_experts:
            raise ValueError(f"expected {self.n_experts} expert weights")
        if np.any(w < 0):
            raise ValueError("expert weights must be non-negative")
        total = float(np.sum(w))
        if total <= 0:
            w = np.full(self.n_experts, 1.0 / self.n_experts)
        else:
            w = w / total
        self.expert_weights = w
        return self

    def set_fit(self, fit: Mapping[str, Any]) -> "CellularSheafFusion":
        """Load the arrays returned by :func:`fit_sheaf_weights`."""
        fit_experts = list(fit.get("experts", self.experts))
        fit_temps = np.asarray(fit["temperatures"], dtype=np.float64)
        temp_by_name = {name: float(fit_temps[i])
                        for i, name in enumerate(fit_experts)
                        if i < fit_temps.size}
        self.set_temperatures([temp_by_name.get(name, 1.0)
                               for name in self.experts])

        fit_expert = fit.get("expert_weights")
        if fit_expert is not None:
            fit_expert = _as_numpy(fit_expert, dtype=np.float64).reshape(-1)
            by_name = {name: float(fit_expert[i])
                       for i, name in enumerate(fit_experts)
                       if i < fit_expert.size}
            if by_name:
                self.set_expert_weights([by_name.get(name, 1.0)
                                         for name in self.experts])

        fit_edges = fit.get("edges", [])
        fit_weights = np.asarray(fit.get("edge_weights", []),
                                 dtype=np.float64).reshape(-1)
        by_edge: dict[tuple[str, str], float] = {}
        for i, edge in enumerate(fit_edges):
            if i >= fit_weights.size:
                break
            if len(edge) < 2:
                continue
            by_edge[(str(edge[0]), str(edge[1]))] = float(fit_weights[i])
            by_edge[(str(edge[1]), str(edge[0]))] = float(fit_weights[i])
        if by_edge:
            weights = []
            for u, v in self.edges:
                key = (self.experts[u], self.experts[v])
                weights.append(by_edge.get(key, 1.0))
            self.set_edge_weights(weights)
        return self

    @classmethod
    def from_fit(cls, fit: Mapping[str, Any], **kwargs: Any
                 ) -> "CellularSheafFusion":
        kwargs.setdefault("experts", fit.get("experts", DEFAULT_EXPERTS))
        kwargs.setdefault("stalk_dim", fit.get("vocab_size", 4096))
        if "edges" in fit:
            kwargs.setdefault("edges", fit["edges"])
        if "mass" in fit:
            kwargs.setdefault("mass", fit["mass"])
        if "edge_mass_power" in fit:
            kwargs.setdefault("mass_power", fit["edge_mass_power"])
        obj = cls(**kwargs)
        obj.set_fit(fit)
        return obj

    def fit(self, expert_logits: Any, targets: Any, **kwargs: Any) -> dict:
        """Convenience wrapper around :func:`fit_sheaf_weights`."""
        kwargs.setdefault("experts", self.experts)
        kwargs.setdefault(
            "edges", [(self.experts[u], self.experts[v])
                      for u, v in self.edges])
        kwargs.setdefault("laplacian_weight", self.laplacian_weight)
        if isinstance(self.mass_mode, str):
            kwargs.setdefault("edge_mass", self.mass_mode)
        kwargs.setdefault("edge_mass_power", self.mass_power)
        result = fit_sheaf_weights(expert_logits, targets, **kwargs)
        self.set_fit(result)
        return result

    # -- prior / mass preparation -----------------------------------------

    def _stack_prior(self, expert_logits: Any) -> np.ndarray:
        e = self.n_experts
        d = self.stalk_dim
        if isinstance(expert_logits, Mapping):
            missing = [name for name in self.experts
                       if name not in expert_logits]
            if missing:
                raise KeyError(f"missing expert logits: {missing}")
            rows = []
            for name in self.experts:
                arr = _as_numpy(expert_logits[name], dtype=np.float64)
                if arr.ndim == 1:
                    arr = arr[None, :]
                if arr.ndim != 2 or arr.shape[1] != d:
                    raise ValueError(
                        f"expert {name!r} must have shape (n, {d}) or ({d},); "
                        f"got {arr.shape}")
                rows.append(arr)
            lengths = {row.shape[0] for row in rows}
            if len(lengths) != 1:
                raise ValueError("all experts must have the same batch size")
            p = np.stack(rows, axis=1)
        elif isinstance(expert_logits, (list, tuple)):
            rows = [_as_numpy(x, dtype=np.float64) for x in expert_logits]
            if len(rows) != e:
                raise ValueError(
                    f"expected {e} expert arrays, got {len(rows)}")
            if all(r.ndim == 1 for r in rows):
                if any(r.shape[0] != d for r in rows):
                    raise ValueError(
                        f"each expert array must have length {d}")
                p = np.stack(rows, axis=0)[None, :, :]
            elif all(r.ndim == 2 and r.shape[1] == d for r in rows):
                lengths = {r.shape[0] for r in rows}
                if len(lengths) != 1:
                    raise ValueError(
                        "all experts must have the same batch size")
                p = np.stack(rows, axis=1)
            else:
                raise ValueError(
                    f"each expert array must be ({d},) or (n, {d})")
        else:
            arr = _as_numpy(expert_logits, dtype=np.float64)
            if arr.ndim == 1:
                if e != 1:
                    raise ValueError(
                        "1-D input is only valid with a single expert")
                arr = arr[None, None, :]
            elif arr.ndim == 2:
                if arr.shape[0] == e and arr.shape[1] == d:
                    arr = arr[None, :, :]
                elif e == 1 and arr.shape[1] == d:
                    arr = arr[:, None, :]
                else:
                    raise ValueError(
                        f"expected ({e}, {d}) or (n, {e}, {d}); got {arr.shape}")
            elif arr.ndim != 3:
                raise ValueError("expert logits must be 1-D, 2-D or 3-D")
            if arr.shape[1] != e or arr.shape[2] != d:
                raise ValueError(
                    f"expected (n, {e}, {d}); got {arr.shape}")
            p = arr
        if not np.all(np.isfinite(p)):
            p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        return p

    def _prepare_prior(self, expert_logits: Any,
                       temperatures: Any = None) -> np.ndarray:
        p = self._stack_prior(expert_logits)
        temps = self.temperatures if temperatures is None else _as_numpy(
            temperatures, dtype=np.float64).reshape(-1)
        if temps.size == 1:
            temps = np.full(self.n_experts, float(temps[0]))
        if temps.size != self.n_experts:
            raise ValueError(f"expected {self.n_experts} temperatures")
        if np.any(temps <= 0):
            raise ValueError("temperatures must be positive")
        p = p / temps[None, :, None]
        if self.normalize:
            p = _log_softmax(p, axis=-1)
        return p.astype(self.dtype, copy=False)

    def _resolve_masses(self, p: np.ndarray, masses: Any = None
                        ) -> np.ndarray:
        b = p.shape[0]
        e = self.n_experts
        mode = self.mass_mode if masses is None else masses
        if isinstance(mode, str):
            key = mode.lower().strip()
            if key in {"uniform", "mean", "none"}:
                m = np.ones((b, e), dtype=np.float64)
            elif key in {"confidence", "conf", "maxprob"}:
                # p is either log-probs or logits; softmax is safe either way.
                m = np.max(_softmax(p, axis=-1), axis=-1) ** self.mass_power
            elif key in {"entropy", "neg_entropy"}:
                probs = _softmax(p, axis=-1)
                ent = -np.sum(probs * np.log(probs + EPS), axis=-1)
                m = np.exp(-self.mass_power * ent / math.log(self.stalk_dim))
            else:
                raise ValueError(f"unknown mass mode {mode!r}")
        elif isinstance(mode, Mapping):
            missing = [name for name in self.experts if name not in mode]
            if missing:
                raise KeyError(f"missing masses for experts: {missing}")
            cols = []
            for name in self.experts:
                a = _as_numpy(mode[name], dtype=np.float64)
                cols.append(a)
            m = np.stack(cols, axis=-1)
            if m.ndim == 1:
                m = np.tile(m[None, :], (b, 1))
        else:
            m = _as_numpy(mode, dtype=np.float64)
            if m.ndim == 0:
                m = np.full((b, e), float(m))
            elif m.ndim == 1:
                if m.size != e:
                    raise ValueError(f"expected {e} masses")
                m = np.tile(m[None, :], (b, 1))
            elif m.ndim != 2 or m.shape != (b, e):
                raise ValueError(f"masses must have shape ({e},) or ({b}, {e})")
        m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
        m = np.maximum(m, self.mass_floor)
        # Scale-free relative masses: the largest expert has mass one.
        m = m / (np.max(m, axis=1, keepdims=True) + EPS)
        return m

    # -- sheaf operators ---------------------------------------------------

    def _laplacian_linear(self, x: np.ndarray) -> np.ndarray:
        """Apply ``B.T W B`` to ``x`` of shape ``(batch, experts, dim)``."""
        out = np.zeros_like(x)
        for ei, (u, v) in enumerate(self.edges):
            w = float(self.edge_weights[ei])
            if w == 0.0:
                continue
            fu = self.maps[(u, ei)]
            fv = self.maps[(v, ei)]
            diff = fu.apply(x[:, u, :]) - fv.apply(x[:, v, :])
            out[:, u, :] += w * fu.apply_transpose(diff)
            out[:, v, :] -= w * fv.apply_transpose(diff)
        return out

    def laplacian_apply(self, vertex_signals: Any) -> np.ndarray:
        """Matrix-free sheaf Laplacian application (public diagnostic)."""
        x = np.asarray(vertex_signals, dtype=self.dtype)
        squeeze = x.ndim == 2
        if squeeze:
            x = x[None, :, :]
        if x.ndim != 3:
            raise ValueError("vertex_signals must be 2-D or 3-D")
        if x.shape[1] != self.n_experts or x.shape[2] != self.stalk_dim:
            raise ValueError(
                f"expected (n, {self.n_experts}, {self.stalk_dim}); "
                f"got {x.shape}")
        out = self._laplacian_linear(x)
        return out[0] if squeeze else out

    laplacian = laplacian_apply

    def _laplacian_constant(self) -> np.ndarray:
        """Constant part of the affine coboundary, shape ``(experts, dim)``."""
        out = np.zeros((self.n_experts, self.stalk_dim), dtype=np.float64)
        for ei, (u, v) in enumerate(self.edges):
            w = float(self.edge_weights[ei])
            if w == 0.0:
                continue
            fu = self.maps[(u, ei)]
            fv = self.maps[(v, ei)]
            if fu.bias is None and fv.bias is None:
                continue
            bu = np.zeros(self.stalk_dim) if fu.bias is None else fu.bias
            bv = np.zeros(self.stalk_dim) if fv.bias is None else fv.bias
            const = bu - bv
            out[u] += w * fu.apply_transpose(const)
            out[v] -= w * fv.apply_transpose(const)
        return out

    def _system(self, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        return (m[..., None] * x
                + self.laplacian_weight * self._laplacian_linear(x))

    def _cg(self, b: np.ndarray, x0: np.ndarray, m: np.ndarray,
            max_iter: int | None = None, tol: float | None = None
            ) -> tuple[np.ndarray, int, np.ndarray]:
        """Batched preconditioned conjugate gradients.

        Each batch item is an independent system ``(M + lambda L) x = b``.
        """
        max_iter = self.max_iter if max_iter is None else int(max_iter)
        tol = self.tol if tol is None else float(tol)
        dtype = self.dtype
        x = np.array(x0, dtype=dtype, copy=True)
        b = np.asarray(b, dtype=dtype)
        r = (b - self._system(x, m)).astype(dtype, copy=False)
        prec = (m[..., None]
                + self.laplacian_weight * self._diag_l[None, :, :]
                + self.mass_floor).astype(dtype, copy=False)
        z = r / prec
        p = z.copy()
        rz = np.sum(r * z, axis=(1, 2))
        b_norm = np.sqrt(np.sum(b * b, axis=(1, 2))) + 1e-30
        iterations = 0
        residual = np.sqrt(np.sum(r * r, axis=(1, 2))) / b_norm
        for it in range(1, max_iter + 1):
            ap = self._system(p, m).astype(dtype, copy=False)
            denom = np.sum(p * ap, axis=(1, 2))
            denom = np.where(np.abs(denom) < EPS, EPS, denom)
            alpha = rz / denom
            x += (alpha[:, None, None] * p).astype(dtype, copy=False)
            r -= (alpha[:, None, None] * ap).astype(dtype, copy=False)
            residual = np.sqrt(np.sum(r * r, axis=(1, 2))) / b_norm
            iterations = it
            if np.max(residual) <= tol:
                break
            z = r / prec
            rz_new = np.sum(r * z, axis=(1, 2))
            beta = np.where(np.abs(rz) < EPS, 0.0, rz_new / (rz + EPS))
            p = z + (beta[:, None, None] * p).astype(dtype, copy=False)
            p = p.astype(dtype, copy=False)
            rz = rz_new
        return x, iterations, residual

    def harmonic_extension(
        self,
        expert_logits: Any,
        masses: Any = None,
        *,
        temperatures: Any = None,
        max_iter: int | None = None,
        tol: float | None = None,
    ) -> np.ndarray:
        """Solve ``(M + lambda L) p = M p0``.

        Returns vertex signals of shape ``(n, experts, stalk_dim)``.
        """
        p0 = self._prepare_prior(expert_logits, temperatures)
        m = self._resolve_masses(p0, masses)
        const = self._laplacian_constant()[None, :, :]
        rhs = m[..., None] * p0 - self.laplacian_weight * const
        # Starting from the prior is a good initial guess for confidence
        # masses; for uniform masses the mean is even closer to the solution.
        if masses is None and isinstance(self.mass_mode, str) and \
                self.mass_mode.lower().strip() in {"uniform", "mean", "none"}:
            x0 = np.mean(p0, axis=1, keepdims=True).repeat(
                self.n_experts, axis=1)
        else:
            x0 = p0
        x, _it, _res = self._cg(rhs, x0, m, max_iter=max_iter, tol=tol)
        return x

    def _output_weights(self, m: np.ndarray) -> np.ndarray:
        if self.expert_weights is not None:
            w = m * self.expert_weights[None, :]
        else:
            w = m
        return w / (np.sum(w, axis=1, keepdims=True) + EPS)

    def _h1_from_vertices(self, x: np.ndarray) -> np.ndarray:
        b = x.shape[0]
        if not self.cycle_edges:
            return np.zeros(b, dtype=np.float64)
        numerator = np.zeros(b, dtype=np.float64)
        denominator = 0.0
        for ei in self.cycle_edges:
            w = float(self.edge_weights[ei])
            if w == 0.0:
                continue
            u, v = self.edges[ei]
            fu = self.maps[(u, ei)]
            fv = self.maps[(v, ei)]
            diff = fu.apply(x[:, u, :]) - fv.apply(x[:, v, :])
            if fu.bias is not None or fv.bias is not None:
                bu = np.zeros(self.stalk_dim) if fu.bias is None else fu.bias
                bv = np.zeros(self.stalk_dim) if fv.bias is None else fv.bias
                diff = diff - (bu - bv)
            energy = np.mean(diff * diff, axis=-1)
            numerator += w * energy
            denominator += w
        if denominator == 0.0:
            return np.zeros(b, dtype=np.float64)
        return np.sqrt(numerator / denominator)

    # -- fusion ------------------------------------------------------------

    def fuse(
        self,
        expert_logits: Any,
        masses: Any = None,
        *,
        temperatures: Any = None,
        return_info: bool = False,
        max_iter: int | None = None,
        tol: float | None = None,
    ) -> Any:
        """Fuse experts and return ``(fused_logits, h1_disagreement)``.

        For a single input the result is ``(stalk_dim,)`` and a float; for a
        batch it is ``(n, stalk_dim)`` and ``(n,)``.  With
        ``return_info=True`` the second element is a dict containing ``h1``
        plus the vertex signals and diagnostics.
        """
        p0 = self._prepare_prior(expert_logits, temperatures)
        m = self._resolve_masses(p0, masses)
        const = self._laplacian_constant()[None, :, :]
        rhs = m[..., None] * p0 - self.laplacian_weight * const
        if masses is None and isinstance(self.mass_mode, str) and \
                self.mass_mode.lower().strip() in {"uniform", "mean", "none"}:
            x0 = np.mean(p0, axis=1, keepdims=True).repeat(
                self.n_experts, axis=1)
        else:
            x0 = p0
        vertex, iterations, residual = self._cg(
            rhs, x0, m, max_iter=max_iter, tol=tol)
        out_w = self._output_weights(m)
        fused = np.einsum("be,bed->bd", out_w, vertex)
        # H1 is a diagnostic of expert disagreement, so measure the input
        # section (before harmonic smoothing).  The post-smoothing residual
        # is retained in the info dict as ``h1_fused``.
        h1 = self._h1_from_vertices(p0)
        h1_fused = self._h1_from_vertices(vertex)
        if fused.shape[0] == 1:
            fused_out = fused[0]
            h1_out: Any = float(h1[0])
        else:
            fused_out = fused
            h1_out = h1
        if return_info:
            info = {
                "h1": h1_out,
                "h1_fused": (float(h1_fused[0]) if h1_fused.size == 1
                             else h1_fused),
                "vertex_logits": vertex,
                "masses": m,
                "output_weights": out_w,
                "iterations": int(iterations),
                "residual": residual,
                "experts": list(self.experts),
                "edge_weights": self.edge_weights.copy(),
                "temperatures": self.temperatures.copy(),
            }
            return fused_out, info
        return fused_out, h1_out

    def __call__(self, expert_logits: Any, masses: Any = None, **kwargs: Any
                 ) -> Any:
        return self.fuse(expert_logits, masses=masses, **kwargs)

    def h1_disagreement(
        self,
        expert_logits: Any = None,
        *,
        fused: Any = None,
        vertex_signals: Any = None,
        smoothed: bool = False,
    ) -> Any:
        """Weighted RMS coboundary on cycle edges.

        The graph must contain a cycle for this to be non-zero in the sheaf
        sense; use a 3-cycle (or a larger ring).  By default this measures the
        input expert section, which is the useful disagreement gate.  Set
        ``smoothed=True`` to measure the post-harmonic-extension section.
        Pass either raw expert logits, already-computed vertex signals, or a
        fused section.  A fused section of shape ``(experts, stalk_dim)`` is
        treated as vertex signals; a ``(stalk_dim,)`` vector is broadcast to
        all vertices.
        """
        if vertex_signals is not None:
            x = np.asarray(vertex_signals, dtype=self.dtype)
            if x.ndim == 2:
                x = x[None, :, :]
            elif x.ndim != 3:
                raise ValueError("vertex_signals must be 2-D or 3-D")
        elif expert_logits is not None:
            arr = None
            if (self.n_experts > 1
                    and not isinstance(expert_logits,
                                       (Mapping, list, tuple))):
                arr = np.asarray(expert_logits)
                # A bare fused vector or an (experts, dim) section is a
                # useful diagnostic input, not a set of expert priors.
                if arr.ndim == 1 and arr.size == self.stalk_dim:
                    x = np.tile(arr[None, None, :],
                                (1, self.n_experts, 1))
                elif arr.ndim == 2 and arr.shape == (
                        self.n_experts, self.stalk_dim):
                    x = arr[None, :, :]
                else:
                    arr = None
            if arr is None:
                p0 = self._prepare_prior(expert_logits)
                x = self.harmonic_extension(expert_logits) if smoothed else p0
        elif fused is not None:
            x = np.asarray(fused, dtype=self.dtype)
            if x.ndim == 1:
                x = np.tile(x[None, None, :], (1, self.n_experts, 1))
            elif x.ndim == 2:
                x = x[None, :, :]
            elif x.ndim != 3:
                raise ValueError("fused must be 1-D, 2-D or 3-D")
        else:
            raise ValueError(
                "provide expert_logits, fused, or vertex_signals")
        h1 = self._h1_from_vertices(x)
        return float(h1[0]) if h1.size == 1 else h1

    # Alias used by older experiments / notebooks.
    h1 = h1_disagreement
    disagreement = h1_disagreement

    def effective_vertex_weights(
        self, masses: Any = None, edge_weights: Any = None
    ) -> np.ndarray:
        """Approximate global mixture weights for the current sheaf.

        This is exact for diagonal restrictions (``low_rank=0``) and a useful
        linearisation for fitting edge weights otherwise.  It answers: if all
        experts emitted a constant signal, how much of each expert's signal
        survives in the aggregated harmonic extension?
        """
        e = self.n_experts
        if masses is None:
            m = np.ones(e, dtype=np.float64)
        else:
            m = np.asarray(masses, dtype=np.float64).reshape(-1)
            if m.size != e:
                raise ValueError(f"expected {e} masses")
        m = np.maximum(m, self.mass_floor)
        weights = self.edge_weights if edge_weights is None else np.asarray(
            edge_weights, dtype=np.float64).reshape(-1)
        if weights.size != len(self.edges):
            raise ValueError(f"expected {len(self.edges)} edge weights")

        # Block Laplacian for scalar restrictions.  For a vector scale we use
        # its mean as the isotropic approximation.
        lap = np.zeros((e, e), dtype=np.float64)
        for ei, (u, v) in enumerate(self.edges):
            w = float(weights[ei])
            if w == 0.0:
                continue
            su = np.mean(self.maps[(u, ei)].scale)
            sv = np.mean(self.maps[(v, ei)].scale)
            lap[u, u] += w * float(su) ** 2
            lap[v, v] += w * float(sv) ** 2
            lap[u, v] -= w * float(su) * float(sv)
            lap[v, u] -= w * float(su) * float(sv)
        mat = np.diag(m) + self.laplacian_weight * lap
        mat = mat + self.mass_floor * np.eye(e)
        try:
            a = np.linalg.solve(mat, np.diag(m))
        except np.linalg.LinAlgError:
            a = np.linalg.pinv(mat) @ np.diag(m)
        c = m if self.expert_weights is None else m * self.expert_weights
        c = c / (np.sum(c) + EPS)
        alpha = a.T @ c
        alpha = np.maximum(alpha, 0.0)
        return alpha / (np.sum(alpha) + EPS)


# ---------------------------------------------------------------------------
# Ising / DPP shortlist reranking
# ---------------------------------------------------------------------------

class IsingDPPSelect:
    """Rerank a candidate shortlist with Ising fields and DPP/MMR diversity.

    The field is

    ``h_i = base_i / temperature + pmi_weight * PMI(i, context)
             + context_weight * cos(emb_i, context_emb) + field_i``

    and the greedy selection maximises either a quality-weighted DPP
    log-determinant (``mode="dpp"``) or the MMR objective
    ``h_i - diversity * max_j cos(emb_i, emb_j)`` (``mode="mmr"``).  The
    diversity term is the anti-ferromagnetic coupling of the Ising picture;
    ``diversity=0`` reduces to greedy fields and ``diversity=1`` maximises
    spread.
    """

    def __init__(
        self,
        vocab_size: int = 4096,
        *,
        embedding_dim: int = 32,
        embeddings: Any = None,
        temperature: float = 1.0,
        pmi_weight: float = 0.5,
        pmi_temperature: float = 1.0,
        context_weight: float = 0.25,
        diversity: float = 0.2,
        mode: str = "dpp",
        quality_temperature: float = 1.0,
        seed: int = 0,
        jitter: float = 1e-6,
        pmi_matrix: Any = None,
    ) -> None:
        self.temperature = float(temperature)
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        self.pmi_weight = float(pmi_weight)
        self.pmi_temperature = max(float(pmi_temperature), EPS)
        self.context_weight = float(context_weight)
        self.diversity = float(diversity)
        if not 0.0 <= self.diversity <= 1.0:
            raise ValueError("diversity must be in [0, 1]")
        self.mode = mode.lower().strip()
        if self.mode not in {"dpp", "mmr"}:
            raise ValueError("mode must be 'dpp' or 'mmr'")
        self.quality_temperature = max(float(quality_temperature), EPS)
        self.jitter = float(jitter)
        self.pmi_matrix = pmi_matrix
        self._provided_embeddings = embeddings is not None
        self._rng = np.random.default_rng(seed)

        if embeddings is None:
            vocab_size = max(int(vocab_size), 1)
            emb = self._rng.normal(0.0, 1.0, (vocab_size, int(embedding_dim)))
        else:
            emb = _as_numpy(embeddings, dtype=np.float64)
            if emb.ndim != 2:
                raise ValueError("embeddings must be a 2-D matrix")
        self.embeddings = self._normalise_rows(emb)
        self.vocab_size = self.embeddings.shape[0]

    # -- embeddings --------------------------------------------------------

    @staticmethod
    def _normalise_rows(emb: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / np.maximum(norms, EPS)

    def ensure_vocab(self, size: int) -> None:
        size = int(size)
        if size <= self.vocab_size:
            return
        if self._provided_embeddings:
            raise ValueError(
                f"candidate id {size - 1} exceeds provided embedding matrix "
                f"with {self.vocab_size} rows")
        extra = self._rng.normal(
            0.0, 1.0, (size - self.vocab_size, self.embeddings.shape[1]))
        self.embeddings = self._normalise_rows(
            np.concatenate([self.embeddings, extra], axis=0))
        self.vocab_size = size

    # -- fields ------------------------------------------------------------

    @staticmethod
    def _as_dense_pmi(pmi: Any) -> Any:
        if pmi is None:
            return None
        if hasattr(pmi, "toarray"):
            return np.asarray(pmi.toarray(), dtype=np.float64)
        return pmi

    def pmi_field(
        self,
        candidates: np.ndarray,
        context_ids: Sequence[int] | None = None,
        pmi_matrix: Any = None,
    ) -> np.ndarray:
        """Mean PMI between each candidate and the context word ids."""
        mat = self._as_dense_pmi(
            self.pmi_matrix if pmi_matrix is None else pmi_matrix)
        out = np.zeros(len(candidates), dtype=np.float64)
        if mat is None or context_ids is None:
            return out
        ctx = np.asarray([int(c) for c in context_ids if int(c) >= 0],
                         dtype=np.int64)
        if ctx.size == 0:
            return out
        if isinstance(mat, np.ndarray):
            if mat.ndim != 2:
                raise ValueError("pmi_matrix must be 2-D")
            ctx = ctx[ctx < mat.shape[1]]
            if ctx.size == 0:
                return out
            valid = (candidates >= 0) & (candidates < mat.shape[0])
            vals = np.zeros((len(candidates), ctx.size), dtype=np.float64)
            if np.any(valid):
                vals[valid] = mat[np.ix_(candidates[valid], ctx)]
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            return np.mean(vals, axis=1)
        # Dict-of-pairs fallback: {(a, b): pmi}.
        vals = []
        for c in candidates:
            row = []
            for ctx_id in ctx:
                row.append(float(mat.get((int(c), int(ctx_id)),
                                        mat.get((int(ctx_id), int(c)), 0.0))))
            vals.append(np.mean(row) if row else 0.0)
        return np.asarray(vals, dtype=np.float64)

    def _context_similarity(
        self,
        candidates: np.ndarray,
        context_embedding: Any = None,
        context_ids: Sequence[int] | None = None,
    ) -> np.ndarray:
        if context_embedding is None and context_ids is not None:
            ids = np.asarray([int(c) for c in context_ids
                              if 0 <= int(c) < self.vocab_size],
                             dtype=np.int64)
            if ids.size:
                context_embedding = np.mean(
                    self.embeddings[ids], axis=0, keepdims=True)
        if context_embedding is None:
            return np.zeros(len(candidates), dtype=np.float64)
        ctx = _as_numpy(context_embedding, dtype=np.float64)
        if ctx.ndim == 2:
            ctx = np.mean(ctx, axis=0)
        norm = np.linalg.norm(ctx)
        if norm < EPS:
            return np.zeros(len(candidates), dtype=np.float64)
        sim = self.embeddings[candidates] @ (ctx / norm)
        return np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)

    # -- selection ---------------------------------------------------------

    def select(
        self,
        candidates: Sequence[int],
        scores: Any = None,
        *,
        logits: Any = None,
        context_ids: Sequence[int] | None = None,
        context_embedding: Any = None,
        pmi_matrix: Any = None,
        pmi_fields: Any = None,
        fields: Any = None,
        top_k: int = 5,
        return_scores: bool = False,
        temperature: float | None = None,
        diversity: float | None = None,
        mode: str | None = None,
    ) -> Any:
        """Return the top-``top_k`` candidate ids.

        ``scores`` is the fused expert score for each candidate (logits or
        log-probs).  Alternatively pass a dense ``logits`` vector and the
        candidate scores are gathered from it.  PMI/context fields are added
        before the diversity-aware greedy selection.
        """
        raw_cand = _as_numpy(candidates, dtype=np.int64).reshape(-1)
        # Preserve the caller's shortlist order while removing duplicate and
        # invalid ids.  Keep the masks so per-candidate side inputs stay
        # aligned.  This makes ties deterministic.
        keep = np.zeros(raw_cand.size, dtype=bool)
        seen: set[int] = set()
        for pos, c in enumerate(raw_cand.tolist()):
            if c < 0 or c in seen:
                continue
            seen.add(c)
            keep[pos] = True
        cand = raw_cand[keep]
        if cand.size == 0:
            return (np.empty(0, dtype=np.int64), np.empty(0)) \
                if return_scores else []
        max_id = int(np.max(cand))
        self.ensure_vocab(max_id + 1)

        def _aligned(value: Any, name: str) -> np.ndarray:
            arr = _as_numpy(value, dtype=np.float64).reshape(-1)
            if arr.size != raw_cand.size:
                raise ValueError(
                    f"expected {raw_cand.size} {name}, got {arr.size}")
            return arr[keep]

        if scores is None and logits is not None:
            dense = _as_numpy(logits, dtype=np.float64).reshape(-1)
            if max_id >= dense.size:
                raise ValueError("candidate id exceeds logits length")
            base = dense[cand]
        elif scores is not None:
            base = _aligned(scores, "scores")
        else:
            base = np.zeros(cand.size, dtype=np.float64)
        base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
        temp = self.temperature if temperature is None else float(temperature)
        if temp <= 0:
            raise ValueError("temperature must be positive")
        field = base / temp

        if pmi_fields is not None:
            pmi = _aligned(pmi_fields, "pmi fields")
            field = field + self.pmi_weight * pmi / self.pmi_temperature
        elif self.pmi_weight != 0.0:
            pmi = self.pmi_field(cand, context_ids, pmi_matrix)
            field = field + self.pmi_weight * pmi / self.pmi_temperature

        if self.context_weight != 0.0:
            field = field + self.context_weight * self._context_similarity(
                cand, context_embedding, context_ids)

        if fields is not None:
            extra = _aligned(fields, "fields")
            field = field + extra
        field = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)

        k = max(0, min(int(top_k), cand.size))
        if k == 0:
            return (np.empty(0, dtype=np.int64), np.empty(0)) \
                if return_scores else []
        if k == cand.size:
            order = np.argsort(-field, kind="stable")
            selected = cand[order]
            if return_scores:
                return selected, field[order]
            return selected.tolist()

        emb = self.embeddings[cand]
        div = self.diversity if diversity is None else float(diversity)
        if not 0.0 <= div <= 1.0:
            raise ValueError("diversity must be in [0, 1]")
        mode_eff = self.mode if mode is None else str(mode).lower().strip()
        if mode_eff not in {"dpp", "mmr"}:
            raise ValueError("mode must be 'dpp' or 'mmr'")
        if mode_eff == "mmr":
            selected = self._mmr_select(field, emb, k, diversity=div)
        else:
            selected = self._dpp_select(field, emb, k, diversity=div)
        chosen = cand[selected]
        if return_scores:
            return chosen, field[selected]
        return chosen.tolist()

    # Convenience aliases used by other selector experiments.
    rerank = select

    def __call__(self, candidates: Sequence[int], scores: Any = None,
                 **kwargs: Any) -> Any:
        return self.select(candidates, scores, **kwargs)

    def _mmr_select(self, field: np.ndarray, emb: np.ndarray, k: int,
                    diversity: float | None = None) -> list[int]:
        n = len(field)
        diversity = self.diversity if diversity is None else float(diversity)
        selected: list[int] = []
        available = np.ones(n, dtype=bool)
        # Scale the diversity penalty to the field's spread so one knob works
        # for both log-prob and raw-logit inputs.
        spread = float(np.std(field)) + 1e-6
        for _ in range(k):
            if not selected:
                penalty = np.zeros(n, dtype=np.float64)
            else:
                sim = emb @ emb[selected].T
                penalty = np.max(sim, axis=1)
            objective = field - diversity * spread * penalty
            objective[~available] = -np.inf
            idx = int(np.argmax(objective))
            selected.append(idx)
            available[idx] = False
        return selected

    def _dpp_select(self, field: np.ndarray, emb: np.ndarray, k: int,
                    diversity: float | None = None) -> list[int]:
        n = len(field)
        diversity = self.diversity if diversity is None else float(diversity)
        # Quality-weighted DPP: q_i = exp((h_i - max h) / quality_temp).
        q = np.exp((field - np.max(field)) / self.quality_temperature)
        q = np.clip(q, 1e-12, 1.0)
        sim = emb @ emb.T
        kernel = (1.0 - diversity) * np.eye(n) + diversity * sim
        # Symmetrise defensively; a convex combination of PSD matrices is PSD.
        kernel = 0.5 * (kernel + kernel.T)
        quality = np.outer(q, q)
        gram = quality * kernel
        gram[np.diag_indices(n)] += self.jitter
        diag = np.diag(gram).copy()
        selected: list[int] = []
        available = np.ones(n, dtype=bool)
        for _ in range(k):
            diag[~available] = -np.inf
            idx = int(np.argmax(diag))
            if not np.isfinite(diag[idx]):
                break
            selected.append(idx)
            available[idx] = False
            if len(selected) == k or not np.any(available):
                break
            # Conditional variance update:
            # d_i <- L_ii - l_Si.T L_SS^{-1} l_Si
            s_idx = np.asarray(selected, dtype=np.int64)
            l_ss = gram[np.ix_(s_idx, s_idx)]
            l_si = gram[np.ix_(s_idx, np.arange(n))]
            try:
                coeff = np.linalg.solve(
                    l_ss + self.jitter * np.eye(len(s_idx)), l_si)
                reduction = np.sum(coeff * coeff, axis=0)
            except np.linalg.LinAlgError:
                reduction = np.zeros(n, dtype=np.float64)
            diag = np.diag(gram) - reduction
            diag = np.maximum(diag, 0.0)
        return selected

    def select_batch(
        self,
        candidates: Sequence[Sequence[int]],
        scores: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> list[list[int]]:
        out = []
        for i, cand in enumerate(candidates):
            s = None if scores is None else scores[i]
            out.append(self.select(cand, s, **kwargs))
        return out


# ---------------------------------------------------------------------------
# Weight fitting
# ---------------------------------------------------------------------------

def _stack_expert_logits(
    expert_logits: Any,
    experts: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    if isinstance(expert_logits, Mapping):
        if experts is None:
            names = list(expert_logits.keys())
        else:
            names = list(experts)
            missing = [n for n in names if n not in expert_logits]
            if missing:
                raise KeyError(f"missing expert logits: {missing}")
        rows = []
        for name in names:
            arr = _as_numpy(expert_logits[name])
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.ndim != 2:
                raise ValueError(
                    f"expert {name!r} must be (n, vocab) or (vocab,)")
            rows.append(arr)
        lengths = {row.shape[0] for row in rows}
        if len(lengths) != 1:
            raise ValueError("all experts must have the same number of rows")
        stack = np.stack(rows, axis=1)
    else:
        if isinstance(expert_logits, (list, tuple)):
            rows = [_as_numpy(x) for x in expert_logits]
            if not rows:
                raise ValueError("expert_logits is empty")
            if all(r.ndim == 1 for r in rows):
                stack = np.stack(rows, axis=0)[None, :, :]
            elif all(r.ndim == 2 for r in rows):
                lengths = {r.shape[0] for r in rows}
                if len(lengths) != 1:
                    raise ValueError(
                        "all experts must have the same number of rows")
                stack = np.stack(rows, axis=1)
            else:
                raise ValueError(
                    "expert_logits list entries must be (vocab,) or "
                    "(n, vocab)")
            if experts is None:
                names = [f"expert{i}" for i in range(stack.shape[1])]
                if stack.shape[1] <= len(DEFAULT_EXPERTS):
                    names = list(DEFAULT_EXPERTS[:stack.shape[1]])
            else:
                names = list(experts)
                if len(names) != stack.shape[1]:
                    raise ValueError(
                        f"expected {stack.shape[1]} expert names, "
                        f"got {len(names)}")
            return np.asarray(stack, dtype=np.float64), names
        arr = _as_numpy(expert_logits)
        if arr.ndim == 2:
            if experts is not None and len(experts) == 1 and arr.shape[1] != 1:
                arr = arr[:, None, :]
            else:
                arr = arr[None, :, :]
        if arr.ndim != 3:
            raise ValueError(
                "expert_logits must be (n, experts, vocab), "
                "(experts, vocab), a list of expert arrays, or a dict of "
                "(n, vocab) arrays")
        stack = arr
        if experts is None:
            names = [f"expert{i}" for i in range(stack.shape[1])]
            if stack.shape[1] <= len(DEFAULT_EXPERTS):
                names = list(DEFAULT_EXPERTS[:stack.shape[1]])
        else:
            names = list(experts)
            if len(names) != stack.shape[1]:
                raise ValueError(
                    f"expected {stack.shape[1]} expert names, got {len(names)}")
    return np.asarray(stack, dtype=np.float64), names


def _nonnegative_lstsq(
    design: np.ndarray,
    target: np.ndarray,
    *,
    ridge: float = 1e-3,
    steps: int = 400,
) -> np.ndarray:
    """Tiny projected-gradient NNLS (no scipy/sklearn)."""
    design = np.asarray(design, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n_var = design.shape[1]
    if n_var == 0:
        return np.empty(0, dtype=np.float64)
    lipschitz = np.linalg.norm(design, 2) ** 2 + ridge + EPS
    step = 1.0 / lipschitz
    w = np.ones(n_var, dtype=np.float64)
    for _ in range(int(steps)):
        grad = design.T @ (design @ w - target) + ridge * (w - 1.0)
        w = np.maximum(0.0, w - step * grad)
    return np.clip(w, 0.0, 10.0)


def _batch_effective_weights(
    masses: np.ndarray,
    edge_weights: np.ndarray,
    edges: Sequence[tuple[int, int]],
    n_experts: int,
    laplacian_weight: float,
) -> np.ndarray:
    """Per-sample effective expert mixture for scalar sheaf restrictions."""
    n = masses.shape[0]
    lap = np.zeros((n_experts, n_experts), dtype=np.float64)
    for ei, (u, v) in enumerate(edges):
        w = float(edge_weights[ei])
        if w == 0.0:
            continue
        lap[u, u] += w
        lap[v, v] += w
        lap[u, v] -= w
        lap[v, u] -= w
    diag_m = np.zeros((n, n_experts, n_experts), dtype=np.float64)
    diag_m[:, np.arange(n_experts), np.arange(n_experts)] = masses
    mat = diag_m + laplacian_weight * lap[None, :, :]
    mat = mat + 1e-8 * np.eye(n_experts)[None, :, :]
    try:
        a = np.linalg.solve(mat, diag_m)
    except np.linalg.LinAlgError:
        a = np.linalg.pinv(mat) @ diag_m
    c = masses / (np.sum(masses, axis=1, keepdims=True) + EPS)
    alpha = np.einsum("nji,nj->ni", a, c)
    alpha = np.maximum(alpha, 0.0)
    return alpha / (np.sum(alpha, axis=1, keepdims=True) + EPS)


def _mass_from_logp(
    logp: np.ndarray,
    mode: str,
    power: float,
    floor: float = 1e-6,
) -> np.ndarray:
    key = str(mode).lower().strip()
    if key in {"uniform", "mean", "none"}:
        m = np.ones(logp.shape[0], dtype=np.float64)
        return m[:, None].repeat(logp.shape[1], axis=1)
    probs = np.exp(logp - np.max(logp, axis=-1, keepdims=True))
    probs /= np.sum(probs, axis=-1, keepdims=True) + EPS
    if key in {"confidence", "conf", "maxprob"}:
        m = np.max(probs, axis=-1) ** float(power)
    elif key in {"entropy", "neg_entropy"}:
        ent = -np.sum(probs * np.log(probs + EPS), axis=-1)
        m = np.exp(-float(power) * ent / math.log(max(logp.shape[-1], 2)))
    else:
        raise ValueError(f"unknown edge mass mode {mode!r}")
    m = np.maximum(m, floor)
    return m / (np.max(m, axis=1, keepdims=True) + EPS)


def _ce_for_alpha(
    alpha: np.ndarray,
    logp: np.ndarray,
    targets: np.ndarray,
) -> float:
    scores = np.einsum("ne,nev->nv", alpha, logp)
    n = scores.shape[0]
    row = np.arange(n)
    return float(np.mean(-scores[row, targets]
                         + np.logaddexp.reduce(scores, axis=1)))


def fit_sheaf_weights(
    expert_logits: Any,
    targets: Any,
    *,
    experts: Sequence[str] | None = None,
    vocab_size: int | None = None,
    normalize: bool = True,
    epochs: int = 80,
    lr: float = 0.05,
    temperature_lr: float | None = None,
    max_positions: int = 1024,
    edges: Any = None,
    laplacian_weight: float = 1.0,
    mass: Any = "uniform",
    edge_mass: Any = "confidence",
    edge_mass_power: float = 0.5,
    edge_search_positions: int = 64,
    edge_search_passes: int = 2,
    ridge: float = 1e-3,
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """Fit scalar temperatures and non-negative edge weights on a train split.

    ``expert_logits`` may be a dict ``{expert: (n, vocab)}`` or an array of
    shape ``(n, experts, vocab)``.  ``targets`` is an integer array of length
    ``n`` (``-100``/negative targets are ignored).

    Temperatures and the mixture weights are fitted by full-batch Adam on the
    train cross-entropy.  Edge weights are then fitted by a short coordinate
    search over the actual per-sample confidence-gated harmonic extension
    (scalar restriction linearisation) on a deterministic subset
    (``edge_search_positions``).  This makes the edge weights identifiable:
    with uniform masses every connected graph would otherwise collapse to the
    same uniform mixture.

    Returns a dict with ``temperatures``, ``edge_weights``,
    ``expert_weights``, ``train_ce``, ``baseline_ce`` and the metadata needed
    by :meth:`CellularSheafFusion.set_fit`.
    """
    stack, names = _stack_expert_logits(expert_logits, experts)
    n, e, v = stack.shape
    targets = _as_numpy(targets, dtype=np.int64).reshape(-1)
    if targets.size != n:
        raise ValueError(f"expected {n} targets, got {targets.size}")
    if vocab_size is not None and int(vocab_size) != v:
        raise ValueError(
            f"vocab_size {vocab_size} does not match logits width {v}")

    valid = (targets >= 0) & (targets < v)
    stack = stack[valid]
    targets = targets[valid]
    n = stack.shape[0]
    if n == 0:
        raise ValueError("no valid training targets")
    if max_positions and n > int(max_positions):
        # Deterministic, evenly spaced subsample preserves book/regime mix.
        idx = np.linspace(0, n - 1, int(max_positions)).astype(np.int64)
        stack = stack[idx]
        targets = targets[idx]
        n = stack.shape[0]

    if normalize:
        logp = _log_softmax(stack, axis=-1)
    else:
        logp = stack
    logp = np.nan_to_num(logp, nan=0.0, posinf=0.0, neginf=0.0)

    # --- mixture / temperature optimisation -----------------------------
    theta = np.zeros(e, dtype=np.float64)
    log_alpha = np.zeros(e, dtype=np.float64)
    m_a = np.zeros(e)
    v_a = np.zeros(e)
    m_t = np.zeros(e)
    v_t = np.zeros(e)
    b1, b2 = 0.9, 0.999
    lr_t = 0.5 * lr if temperature_lr is None else float(temperature_lr)
    row = np.arange(n)
    history: list[float] = []
    best_loss = float("inf")
    best_alpha = np.full(e, 1.0 / e)
    best_theta = theta.copy()

    for epoch in range(1, int(epochs) + 1):
        theta_used = np.clip(theta, -2.3, 2.3)
        temps = np.exp(theta_used)
        alpha = _softmax(log_alpha)
        z = logp / temps[None, :, None]
        scores = np.einsum("e,nev->nv", alpha, z)
        log_z = np.logaddexp.reduce(scores, axis=1)
        nll = -scores[row, targets] + log_z
        loss = float(np.mean(nll))
        history.append(loss)
        if loss < best_loss - 1e-9:
            best_loss = loss
            best_alpha = alpha.copy()
            best_theta = theta_used.copy()
        prob = np.exp(scores - log_z[:, None])
        ds = prob
        ds[row, targets] -= 1.0
        ds /= n
        g_alpha = np.einsum("nv,nev->e", ds, z)
        g_log_alpha = alpha * (g_alpha - np.sum(alpha * g_alpha))
        g_theta = -alpha * np.einsum(
            "nv,nev->e", ds, logp) / temps
        # Keep the fit close to the identity transform unless the data says
        # otherwise; this matters on small book splits.
        g_theta += ridge * theta_used

        # Adam for the unconstrained parameters.
        m_a = b1 * m_a + (1.0 - b1) * g_log_alpha
        v_a = b2 * v_a + (1.0 - b2) * g_log_alpha ** 2
        m_t = b1 * m_t + (1.0 - b1) * g_theta
        v_t = b2 * v_t + (1.0 - b2) * g_theta ** 2
        corr1 = 1.0 - b1 ** epoch
        corr2 = 1.0 - b2 ** epoch
        log_alpha -= lr * (m_a / corr1) / (np.sqrt(v_a / corr2) + 1e-8)
        log_alpha = np.clip(log_alpha, -20.0, 20.0)
        theta -= lr_t * (m_t / corr1) / (np.sqrt(v_t / corr2) + 1e-8)
        theta = np.clip(theta, -2.3, 2.3)
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"[fit_sheaf] epoch {epoch:3d} ce={loss:.5f}",
                  flush=True)

    alpha = best_alpha
    temps = np.exp(best_theta)
    # Recompute the loss of the selected parameters.
    z = logp / temps[None, :, None]
    scores = np.einsum("e,nev->nv", alpha, z)
    train_ce = float(np.mean(-scores[row, targets]
                             + np.logaddexp.reduce(scores, axis=1)))
    base_scores = np.mean(logp, axis=1)
    baseline_ce = float(np.mean(-base_scores[row, targets]
                                + np.logaddexp.reduce(base_scores, axis=1)))

    # --- map mixture weights to non-negative edge weights ----------------
    fusion = CellularSheafFusion(
        experts=names,
        stalk_dim=v,
        edges=edges,
        low_rank=0,
        scale=1.0,
        edge_weight=1.0,
        laplacian_weight=laplacian_weight,
        mass=mass,
        normalize=False,
        max_iter=20,
        tol=1e-8,
    )
    m_uniform = np.ones(e, dtype=np.float64)
    base_alpha = fusion.effective_vertex_weights(
        m_uniform, np.zeros(len(fusion.edges)))
    columns = []
    for j in range(len(fusion.edges)):
        w = np.zeros(len(fusion.edges), dtype=np.float64)
        w[j] = 1.0
        columns.append(fusion.effective_vertex_weights(m_uniform, w)
                       - base_alpha)
    if columns:
        design = np.stack(columns, axis=1)
        edge_weights = _nonnegative_lstsq(
            design, alpha - base_alpha, ridge=ridge)
    else:
        edge_weights = np.empty(0, dtype=np.float64)
    if edge_weights.size == 0:
        edge_weights = np.ones(len(fusion.edges), dtype=np.float64)

    # Refine edge weights on the actual confidence-gated harmonic extension.
    # A short coordinate search on a deterministic subset is enough and keeps
    # the fit dependency-free and fast.
    edge_ce = float("nan")
    edge_ce_uniform = float("nan")
    if len(fusion.edges):
        n_edge = min(int(edge_search_positions), n)
        if n_edge > 0:
            idx = np.linspace(0, n - 1, n_edge).astype(np.int64)
            z_edge = logp[idx] / temps[None, :, None]
            if normalize:
                z_edge = _log_softmax(z_edge, axis=-1)
            t_edge = targets[idx]
            if isinstance(edge_mass, str):
                m_edge = _mass_from_logp(
                    z_edge, edge_mass, edge_mass_power)
            else:
                m_edge = np.asarray(edge_mass, dtype=np.float64)
                if m_edge.ndim == 1:
                    m_edge = np.tile(m_edge[None, :], (n_edge, 1))
                elif m_edge.ndim == 2 and m_edge.shape[0] == n:
                    m_edge = m_edge[idx]
                else:
                    raise ValueError(
                        "edge_mass must be a mode string, (experts,) or "
                        "(n, experts)")
                m_edge = np.maximum(m_edge, 1e-6)
                m_edge = (m_edge /
                          (np.max(m_edge, axis=1, keepdims=True) + EPS))

            def _eval_edge(w: np.ndarray) -> float:
                a = _batch_effective_weights(
                    m_edge, w, fusion.edges, e, laplacian_weight)
                return _ce_for_alpha(a, z_edge, t_edge)

            uniform_w = np.ones(len(fusion.edges), dtype=np.float64)
            edge_ce_uniform = _eval_edge(uniform_w)
            best_w = uniform_w.copy()
            best_ce = edge_ce_uniform
            # Try the NNLS solution as a starting point as well.
            if edge_weights.size == len(fusion.edges):
                ce_nnls = _eval_edge(edge_weights)
                if ce_nnls < best_ce:
                    best_ce = ce_nnls
                    best_w = edge_weights.copy()
            # Global scale first: edge weight magnitude trades coupling
            # against the per-expert confidence mass.
            for factor in (0.25, 0.5, 1.0, 2.0, 4.0):
                cand = best_w * factor
                ce = _eval_edge(cand)
                if ce < best_ce - 1e-9:
                    best_ce = ce
                    best_w = cand
            factors = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
            for _pass in range(max(0, int(edge_search_passes))):
                improved = False
                for j in range(len(best_w)):
                    base_j = float(best_w[j]) if best_w[j] > 1e-9 else 0.25
                    for factor in factors:
                        cand = best_w.copy()
                        cand[j] = min(10.0, max(0.0, base_j * factor))
                        ce = _eval_edge(cand)
                        if ce < best_ce - 1e-9:
                            best_ce = ce
                            best_w = cand
                            improved = True
                if not improved:
                    break
            edge_weights = best_w
            edge_ce = best_ce

    result = {
        "experts": list(names),
        "vocab_size": int(v),
        "temperatures": temps,
        "edge_weights": edge_weights,
        "expert_weights": alpha,
        "edges": [(names[u], names[v]) for u, v in fusion.edges],
        "train_ce": train_ce,
        "baseline_ce": baseline_ce,
        "improvement": baseline_ce - train_ce,
        "edge_ce": edge_ce,
        "edge_ce_uniform": edge_ce_uniform,
        "loss_history": history,
        "n_train": int(n),
        "normalized": bool(normalize),
        "laplacian_weight": float(laplacian_weight),
        "mass": mass,
        "edge_mass": edge_mass if isinstance(edge_mass, str) else "custom",
        "edge_mass_power": float(edge_mass_power),
    }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _topk_accuracy(scores: np.ndarray, targets: np.ndarray, k: int = 5
                   ) -> float:
    if len(targets) == 0:
        return 0.0
    scores = np.asarray(scores)
    k = max(1, min(int(k), scores.shape[1]))
    # Scores are log-probs over the full vocabulary.
    order = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    hits = np.sum(order == targets[:, None])
    return float(hits) / float(len(targets))


def _candidate_union(
    i: int,
    top_ids: Mapping[str, np.ndarray],
    rankings: Mapping[str, np.ndarray],
    k: int,
) -> set[int]:
    out: set[int] = set()
    for name, arr in top_ids.items():
        if i < arr.shape[0]:
            out.update(int(w) for w in arr[i, :k])
    for arr in rankings.values():
        if i < arr.shape[0]:
            out.update(int(w) for w in arr[i, :k] if w >= 0)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sheaf fusion on the cached word expert scores.")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE),
                        help="path to word_dev_cache.npz")
    parser.add_argument("--topk", type=int, default=100,
                        help="candidate cutoff for coverage/conditional top-5")
    parser.add_argument("--low-rank", type=int, default=2,
                        help="rank of the sheaf restriction maps")
    parser.add_argument("--fit-epochs", type=int, default=80)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--max-train", type=int, default=1024)
    parser.add_argument("--limit-test", type=int, default=0,
                        help="evaluate only the first N test records (0=all)")
    parser.add_argument("--mass", default="confidence",
                        choices=("uniform", "confidence", "entropy"))
    parser.add_argument("--mass-power", type=float, default=0.5)
    args = parser.parse_args(argv)

    cache = Path(args.cache)
    if not cache.exists():
        print(f"[word_sheaf] cache not found: {cache}")
        print("[word_sheaf] the word dev cache is produced on the server by "
              "terminal_rsi/word_model_rerank.py;")
        print("[word_sheaf] skipping fusion evaluation (no results invented).")
        return 0

    data = np.load(cache)
    required = ("head", "bridge", "prior", "ppm", "cache", "copy",
                "target", "book")
    missing = [key for key in required if key not in data]
    if missing:
        print(f"[word_sheaf] cache is missing arrays: {missing}")
        return 1
    head = np.asarray(data["head"], dtype=np.float64)
    bridge = np.asarray(data["bridge"], dtype=np.float64)
    prior = np.asarray(data["prior"], dtype=np.float64)
    ppm = np.asarray(data["ppm"])
    cache_rank = np.asarray(data["cache"])
    copy_rank = np.asarray(data["copy"])
    target = np.asarray(data["target"], dtype=np.int64)
    book = np.asarray(data["book"])
    if head.ndim != 2:
        print(f"[word_sheaf] unexpected head shape {head.shape}")
        return 1
    n, vocab = head.shape
    topk = max(1, min(int(args.topk), vocab))
    print(f"[word_sheaf] records={n} vocab={vocab} "
          f"ppm={ppm.shape} cache={cache_rank.shape} copy={copy_rank.shape}",
          flush=True)

    train_idx = np.where(book < 8)[0]
    test_idx = np.where(book >= 8)[0]
    if args.limit_test > 0:
        test_idx = test_idx[:args.limit_test]
    if train_idx.size == 0 or test_idx.size == 0:
        print("[word_sheaf] need both train (book<8) and test (book>=8) "
              "records; nothing to do.")
        return 0

    # Dense versions of the ranking experts.  alpha=1 is a power-law prior;
    # this is deliberately not tuned on the test split.
    rank_logp = {
        "ppm": rankings_to_log_probs(ppm, vocab, alpha=1.0),
        "cache": rankings_to_log_probs(cache_rank, vocab, alpha=1.0),
        "copy": rankings_to_log_probs(copy_rank, vocab, alpha=1.0),
    }
    dense = {
        "head": head,
        "bridge": bridge,
        "prior": prior,
    }
    experts = tuple(dense.keys()) + tuple(rank_logp.keys())
    train = {name: dense[name][train_idx] for name in dense}
    train.update({name: rank_logp[name][train_idx] for name in rank_logp})
    test = {name: dense[name][test_idx] for name in dense}
    test.update({name: rank_logp[name][test_idx] for name in rank_logp})
    test_targets = target[test_idx]

    print(f"[word_sheaf] fitting on {len(train_idx)} train records; "
          f"experts={list(experts)}", flush=True)
    fit = fit_sheaf_weights(
        train, target[train_idx], experts=experts, vocab_size=vocab,
        normalize=True, edges="cycle", mass=args.mass,
        edge_mass=args.mass, edge_mass_power=args.mass_power,
        epochs=args.fit_epochs, max_positions=args.max_train,
        verbose=True,
    )
    fusion = CellularSheafFusion(
        experts=experts, stalk_dim=vocab, edges="cycle",
        low_rank=args.low_rank, mass=args.mass,
        mass_power=args.mass_power,
        normalize=True, max_iter=args.max_iter,
    )
    fusion.set_fit(fit)

    # Evaluate fusion in chunks to keep memory bounded on a phone/server.
    fused_rows = []
    h1_rows = []
    chunk = 32
    for start in range(0, len(test_idx), chunk):
        stop = min(start + chunk, len(test_idx))
        part = {name: test[name][start:stop] for name in experts}
        fused, info = fusion.fuse(part, return_info=True)
        if fused.ndim == 1:
            fused = fused[None, :]
        fused_rows.append(fused)
        h1 = info["h1"]
        h1_rows.append(np.atleast_1d(h1))
    fused_all = np.concatenate(fused_rows, axis=0)
    h1_all = np.concatenate(h1_rows, axis=0)

    # Coverage uses the same top-k source lists as the rest of the word
    # experiments: dense experts are argsorted, ranking experts use their
    # stored lists (padded with -1).
    top_ids = {
        "head": np.argpartition(-head[test_idx], topk - 1, axis=1)[
            :, :topk],
        "bridge": np.argpartition(-bridge[test_idx], topk - 1, axis=1)[
            :, :topk],
        "prior": np.argpartition(-prior[test_idx], topk - 1, axis=1)[
            :, :topk],
    }
    rankings = {
        "ppm": ppm[test_idx],
        "cache": cache_rank[test_idx],
        "copy": copy_rank[test_idx],
    }
    covered = np.zeros(len(test_idx), dtype=bool)
    for i in range(len(test_idx)):
        covered[i] = int(test_targets[i]) in _candidate_union(
            i, top_ids, rankings, topk)

    top1 = _topk_accuracy(fused_all, test_targets, k=1)
    top5 = _topk_accuracy(fused_all, test_targets, k=5)
    cond_top5 = (_topk_accuracy(fused_all[covered], test_targets[covered], 5)
                 if np.any(covered) else 0.0)
    expert_top5 = {}
    for name in experts:
        expert_top5[name] = _topk_accuracy(test[name], test_targets, 5)

    report = {
        "cache": str(cache),
        "records": int(n),
        "vocab": int(vocab),
        "train_records": int(len(train_idx)),
        "test_records": int(len(test_idx)),
        "coverage": float(np.mean(covered)) if len(covered) else 0.0,
        "fusion": {
            "top1_true": top1,
            "top5_true": top5,
            "top5_conditional": cond_top5,
            "h1_mean": float(np.mean(h1_all)) if h1_all.size else 0.0,
            "h1_median": float(np.median(h1_all)) if h1_all.size else 0.0,
        },
        "expert_top5_true": expert_top5,
        "fit": {
            "train_ce": fit["train_ce"],
            "baseline_ce": fit["baseline_ce"],
            "temperatures": fit["temperatures"].tolist(),
            "edge_weights": fit["edge_weights"].tolist(),
            "edge_names": [list(pair) for pair in fit["edges"]],
            "expert_weights": fit["expert_weights"].tolist(),
        },
    }
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
