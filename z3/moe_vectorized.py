#!/usr/bin/env python3
"""Grouped, vectorized top-k sparse MoE dispatch.

This replaces the per-expert Python loop in ``v3_model.SparseMoE`` with a
grouped gather + einsum dispatch:

  * router logits: Linear(D, E), no bias
  * top-k selection per token, softmax over the selected logits
  * expert SwiGLU weights stored as dense Parameters, not a ModuleList:
        gate_w (E, D, H), up_w (E, D, H), down_w (E, H, D)
  * tokens are routed to a group of experts, the group is applied with
    batched einsums, and only the selected expert outputs are gathered
  * shared experts are one batched shared FFN (S, D, H) / (S, H, D)

The original gather materialized ``(tokens * topk, D, H)`` weights per token.
At the T4 profile (4096 tokens, 32 experts, H=1024) that is ~8.6 GB per
projection.  This version bounds memory with ``group_size`` and ``max_rows``
row chunks, so the same profile fits a T4.  The loop is over small expert
groups and row chunks, not over every expert per token.

API
---
    moe = VectorizedSparseMoE(d_model=..., n_experts=..., topk=...,
                              shared_experts=..., expert_hidden=...,
                              dropout=..., group_size=4, max_rows=512)
    out, aux = moe(x)                 # x: (B, T, D)
    # aux["load_balance"]: Switch-style E * sum(f_i * P_i)
    # aux["router_z"]:     mean(logsumexp(logits, -1) ** 2)

    logits, probs, topv, topi, w = moe.route(flat)   # flat: (M, D)

With ``dropout > 0`` and ``training=True``, dropout masks are drawn over the
whole dispatched batch rather than per expert, so the stochastic output is not
bit-identical to a per-expert reference even though the expected computation is
the same. Tests use ``dropout=0``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["VectorizedSparseMoE"]


class VectorizedSparseMoE(nn.Module):
    """Top-k routed SwiGLU MoE with grouped expert dispatch.

    Args:
        d_model: token width D.
        n_experts: number of routed experts E.
        topk: experts selected per token K.
        shared_experts: number of always-on shared experts S.
        expert_hidden: per-expert SwiGLU hidden width H. Defaults to 4 * D.
        dropout: dropout probability on expert outputs.
        group_size: experts processed together in one batched einsum.
        max_rows: token-row chunk size for peak-memory bounding.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        topk: int = 2,
        shared_experts: int = 0,
        expert_hidden: int | None = None,
        dropout: float = 0.0,
        group_size: int = 4,
        max_rows: int = 512,
        dense_dispatch: bool = False,
        hierarchical: bool = False,
        branches: int = 0,
        context_router: bool = False,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if n_experts <= 0:
            raise ValueError(f"n_experts must be positive, got {n_experts}")
        if not 0 < topk <= n_experts:
            raise ValueError(f"topk must be in [1, n_experts], got {topk}")
        if shared_experts < 0:
            raise ValueError(f"shared_experts must be >= 0, got {shared_experts}")
        if expert_hidden is None:
            expert_hidden = 4 * d_model
        if expert_hidden <= 0:
            raise ValueError(f"expert_hidden must be positive, got {expert_hidden}")
        if not 1 <= int(group_size) <= n_experts:
            raise ValueError("group_size must be between 1 and n_experts")
        if int(max_rows) <= 0:
            raise ValueError("max_rows must be positive")

        self.d_model = int(d_model)
        self.n_experts = int(n_experts)
        self.topk = int(topk)
        self.shared_experts = int(shared_experts)
        self.expert_hidden = int(expert_hidden)
        self.dropout = float(dropout)
        self.group_size = int(group_size)
        self.max_rows = int(max_rows)
        self.dense_dispatch = bool(dense_dispatch)
        self.hierarchical = bool(hierarchical)
        self.context_router = bool(context_router)
        self.branches = 0
        self.experts_per_branch = 0
        if self.hierarchical:
            candidates = [
                b
                for b in range(1, self.n_experts + 1)
                if self.n_experts % b == 0
                and self.n_experts // b >= self.topk
            ]
            if not candidates:
                raise ValueError(
                    "hierarchical routing needs at least topk experts per branch"
                )
            requested = int(branches or 0)
            if requested <= 0:
                requested = min(8, max(1, self.n_experts // max(1, self.topk)))
            valid = [b for b in candidates if b <= requested]
            self.branches = max(valid) if valid else min(candidates)
            self.experts_per_branch = self.n_experts // self.branches
        self.branch_router = (
            nn.Linear(self.d_model, self.branches, bias=False)
            if self.hierarchical else None
        )
        # Both context projections start at zero: the hierarchy is active
        # from step zero, but the attention/routing feedback can only grow
        # into the router as the trunk learns to use it.
        self.branch_context = (
            nn.Linear(self.d_model, self.branches, bias=False)
            if self.hierarchical and self.context_router else None
        )
        self.expert_context = (
            nn.Linear(self.d_model, self.n_experts, bias=False)
            if self.context_router else None
        )
        self.branch_embedding = (
            nn.Parameter(torch.empty(self.branches, self.d_model))
            if self.hierarchical else None
        )
        self.expert_embedding = (
            nn.Parameter(torch.empty(self.n_experts, self.d_model))
            if self.hierarchical else None
        )

        self.router = nn.Linear(self.d_model, self.n_experts, bias=False)

        # Routed expert SwiGLU weights, one dense Parameter per projection.
        self.gate_w = nn.Parameter(
            torch.empty(self.n_experts, self.d_model, self.expert_hidden)
        )
        self.up_w = nn.Parameter(
            torch.empty(self.n_experts, self.d_model, self.expert_hidden)
        )
        self.down_w = nn.Parameter(
            torch.empty(self.n_experts, self.expert_hidden, self.d_model)
        )

        # Shared experts as one batched FFN.
        if self.shared_experts > 0:
            self.shared_gate_w = nn.Parameter(
                torch.empty(self.shared_experts, self.d_model, self.expert_hidden)
            )
            self.shared_up_w = nn.Parameter(
                torch.empty(self.shared_experts, self.d_model, self.expert_hidden)
            )
            self.shared_down_w = nn.Parameter(
                torch.empty(self.shared_experts, self.expert_hidden, self.d_model)
            )
        else:
            self.shared_gate_w = None
            self.shared_up_w = None
            self.shared_down_w = None

        self.reset_parameters()

    @classmethod
    def from_config(cls, cfg) -> "VectorizedSparseMoE":
        """Build from a ``V3Config``-like object without editing v3_model."""
        return cls(
            d_model=cfg.d_model,
            n_experts=cfg.n_experts,
            topk=cfg.topk,
            shared_experts=cfg.shared_experts,
            expert_hidden=cfg.expert_hidden,
            dropout=cfg.dropout,
            group_size=getattr(cfg, "moe_group_size", 4),
            max_rows=getattr(cfg, "moe_max_rows", 512),
            dense_dispatch=getattr(cfg, "moe_dense_dispatch", False),
            hierarchical=getattr(cfg, "moe_hierarchical", False),
            branches=getattr(cfg, "moe_branches", 0),
            context_router=getattr(cfg, "moe_context_router", False),
        )

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.router.weight)
        if self.branch_router is not None:
            nn.init.xavier_uniform_(self.branch_router.weight)
        if self.branch_context is not None:
            self.branch_context.weight.zero_()
        if self.expert_context is not None:
            self.expert_context.weight.zero_()
        if self.branch_embedding is not None:
            nn.init.normal_(self.branch_embedding, std=0.02)
        if self.expert_embedding is not None:
            nn.init.normal_(self.expert_embedding, std=0.02)
        # Xavier per expert slice, matching SwiGLU's Linear initialization.
        for e in range(self.n_experts):
            nn.init.xavier_uniform_(self.gate_w[e])
            nn.init.xavier_uniform_(self.up_w[e])
            nn.init.xavier_uniform_(self.down_w[e])
        if self.shared_gate_w is not None:
            for s in range(self.shared_experts):
                nn.init.xavier_uniform_(self.shared_gate_w[s])
                nn.init.xavier_uniform_(self.shared_up_w[s])
                nn.init.xavier_uniform_(self.shared_down_w[s])

    def route(self, flat: torch.Tensor, context: torch.Tensor | None = None):
        """Route flattened tokens, optionally with a context/branch hierarchy.

        Args:
            flat: (M, D) tokens.
            context: optional (M, D) attention/routing feedback summary.

        Returns:
            logits: (M, E) expert router logits (context-conditioned).
            probs: (M, E) full expert distribution.
            topv: (M, K) selected branch-adjusted logits.
            topi: (M, K) selected global expert indices.
            weights: (M, K) softmax over selected logits.
        """
        # The module call keeps forward hooks (used by eval_v3's router
        # utilization tracker) working.  The explicit F.linear branch is only
        # for callers passing a differently-typed activation (e.g. fp16).
        if flat.dtype == self.router.weight.dtype:
            logits = self.router(flat)
        else:
            logits = F.linear(flat, self.router.weight.to(dtype=flat.dtype))
        if self.expert_context is not None and context is not None:
            context = context.to(dtype=flat.dtype)
            if context.ndim == 3:
                context = context.reshape(-1, context.shape[-1])
            if context.shape[0] != flat.shape[0]:
                raise ValueError(
                    "router context must be expanded to (tokens, d_model)"
                )
            logits = logits + self.expert_context(context)
        probs = F.softmax(logits, dim=-1)

        if not self.hierarchical:
            topv, topi = torch.topk(logits, k=self.topk, dim=-1)
            weights = F.softmax(topv, dim=-1)
            self._last_branch_idx = None
            self._last_branch_logits = None
            self._last_branch_probs = None
            return logits, probs, topv, topi, weights

        assert self.branch_router is not None
        branch_logits = self.branch_router(flat)
        if self.branch_context is not None and context is not None:
            branch_logits = branch_logits + self.branch_context(context)
        m = flat.shape[0]
        pair_logits = logits.reshape(
            m, self.branches, self.experts_per_branch
        )
        branch_idx = branch_logits.argmax(dim=-1)
        rows = torch.arange(m, device=flat.device)
        selected_pairs = pair_logits[rows, branch_idx]
        selected_scores = selected_pairs + branch_logits[
            rows, branch_idx
        ].unsqueeze(-1)
        topv, local_idx = selected_scores.topk(self.topk, dim=-1)
        topi = branch_idx.unsqueeze(-1) * self.experts_per_branch + local_idx
        weights = F.softmax(topv, dim=-1)
        self._last_branch_idx = branch_idx
        self._last_branch_logits = branch_logits
        self._last_branch_probs = F.softmax(branch_logits, dim=-1)
        return logits, probs, topv, topi, weights

    def _chunked_group_dispatch(self, flat, topi, weights):
        """Memory-bounded dispatch used on CUDA/CPU (dynamic row selection)."""
        out = torch.zeros_like(flat)
        for g0 in range(0, self.n_experts, self.group_size):
            g1 = min(g0 + self.group_size, self.n_experts)
            group_width = g1 - g0
            in_group = (topi >= g0) & (topi < g1)  # (M, K)
            if not bool(in_group.any()):
                continue
            rows = in_group.any(dim=1).nonzero(as_tuple=False).flatten()
            for r0 in range(0, rows.numel(), self.max_rows):
                sel_rows = rows[r0:r0 + self.max_rows]
                xg = flat.index_select(0, sel_rows)                 # (N, D)
                selected = in_group.index_select(0, sel_rows)       # (N, K)
                local = (topi.index_select(0, sel_rows) - g0).clamp(
                    0, group_width - 1
                )                                                   # (N, K)
                row_ids = torch.arange(
                    sel_rows.numel(), device=flat.device
                ).unsqueeze(1)

                h_gate = torch.einsum(
                    "nd,gdh->gnh", xg,
                    self.gate_w[g0:g1].to(dtype=xg.dtype),
                )
                h_up = torch.einsum(
                    "nd,gdh->gnh", xg,
                    self.up_w[g0:g1].to(dtype=xg.dtype),
                )
                y = torch.einsum(
                    "gnh,ghd->gnd", F.silu(h_gate) * h_up,
                    self.down_w[g0:g1].to(dtype=xg.dtype),
                )                                                   # (G, N, D)
                y = F.dropout(y, p=self.dropout, training=self.training)
                y_sel = y[local, row_ids, :] * selected.unsqueeze(-1)
                w_sel = weights.index_select(0, sel_rows) * selected
                out.index_add_(
                    0, sel_rows,
                    (y_sel * w_sel.unsqueeze(-1)).sum(dim=1).to(
                        dtype=out.dtype
                    ),
                )
        return out

    def _dense_group_dispatch(self, flat, topi, weights):
        """Fixed-shape dispatch for XLA; avoids dynamic ``nonzero`` shapes."""
        out = torch.zeros_like(flat)
        m = flat.shape[0]
        row_ids = torch.arange(m, device=flat.device).unsqueeze(1)
        for g0 in range(0, self.n_experts, self.group_size):
            g1 = min(g0 + self.group_size, self.n_experts)
            group_width = g1 - g0
            selected = (topi >= g0) & (topi < g1)  # (M, K)
            local = (topi - g0).clamp(0, group_width - 1)

            h_gate = torch.einsum(
                "md,gdh->gmh", flat,
                self.gate_w[g0:g1].to(dtype=flat.dtype),
            )
            h_up = torch.einsum(
                "md,gdh->gmh", flat,
                self.up_w[g0:g1].to(dtype=flat.dtype),
            )
            y = torch.einsum(
                "gmh,ghd->gmd", F.silu(h_gate) * h_up,
                self.down_w[g0:g1].to(dtype=flat.dtype),
            )                                                       # (G, M, D)
            y = F.dropout(y, p=self.dropout, training=self.training)
            y_sel = y[local, row_ids, :] * selected.unsqueeze(-1)
            out = out + (y_sel * (weights * selected).unsqueeze(-1)).sum(dim=1)
        return out

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ):
        """Forward pass with optional attention/routing context.

        Args:
            x: (B, T, D) token states.
            context: optional (B, D) or (B, T, D) route context.  When present
                it conditions both the branch and expert routers.

        Returns:
            out: (B, T, D) MoE output.
            aux: scalar balance/z terms plus the route summary consumed by
                the next block's context feedback.
        """
        if x.ndim != 3:
            raise ValueError(f"x must be (B, T, D), got shape {tuple(x.shape)}")
        b, t, d = x.shape
        if d != self.d_model:
            raise ValueError(f"expected last dim {self.d_model}, got {d}")

        flat = x.reshape(-1, d)
        m = flat.shape[0]
        route_context = None
        if context is not None:
            route_context = context.to(dtype=x.dtype)
            if route_context.ndim == 2:
                route_context = route_context[:, None, :].expand(b, t, d)
            route_context = route_context.reshape(-1, d)
            if route_context.shape[0] != m:
                raise ValueError("route context does not match token shape")
        logits, probs, _, topi, weights = self.route(flat, route_context)

        # Grouped dispatch.  A group of experts is applied to every token
        # routed to that group; no (tokens, d_model, expert_hidden) gather is
        # created.  XLA uses a fixed-shape variant to avoid dynamic nonzero
        # shapes; CUDA/CPU uses row chunks to bound peak activation memory.
        if self.dense_dispatch or x.device.type == "xla":
            out = self._dense_group_dispatch(flat, topi, weights)
        else:
            out = self._chunked_group_dispatch(flat, topi, weights)

        # Shared experts: batched SwiGLU over row chunks.
        if self.shared_gate_w is not None:
            for r0 in range(0, m, self.max_rows):
                chunk = flat[r0:r0 + self.max_rows]
                sh_gate = torch.einsum(
                    "nd,sdh->snh", chunk,
                    self.shared_gate_w.to(dtype=chunk.dtype),
                )
                sh_up = torch.einsum(
                    "nd,sdh->snh", chunk,
                    self.shared_up_w.to(dtype=chunk.dtype),
                )
                shared = torch.einsum(
                    "snh,shd->snd", F.silu(sh_gate) * sh_up,
                    self.shared_down_w.to(dtype=chunk.dtype),
                )
                shared = F.dropout(
                    shared, p=self.dropout, training=self.training
                )
                out[r0:r0 + chunk.shape[0]] += shared.sum(dim=0).to(
                    dtype=out.dtype
                )

        # Switch-style load balance and router z-loss.  Hierarchical routers
        # get an additional branch balance term; otherwise the branch router
        # collapses to whichever branch happens to fit the first batches.
        mask = torch.zeros_like(logits).scatter_(1, topi, 1.0)
        f = mask.mean(dim=0)
        p = probs.mean(dim=0)
        expert_load_balance = self.n_experts * (f * p).sum()
        router_z = torch.logsumexp(logits, dim=-1).pow(2).mean()
        aux = {
            "load_balance": expert_load_balance,
            "router_z": router_z,
        }
        if self.hierarchical and self._last_branch_idx is not None:
            branch_idx = self._last_branch_idx
            branch_counts = F.one_hot(
                branch_idx, num_classes=self.branches
            ).to(dtype=logits.dtype).mean(dim=0)
            branch_p = self._last_branch_probs.mean(dim=0)
            branch_lb = self.branches * (branch_counts * branch_p).sum()
            aux["branch_load_balance"] = branch_lb
            aux["branch_router_z"] = torch.logsumexp(
                self._last_branch_logits, dim=-1
            ).pow(2).mean()
            aux["load_balance"] = expert_load_balance + branch_lb
            aux["router_z"] = router_z + aux["branch_router_z"]
            assert self.branch_embedding is not None
            assert self.expert_embedding is not None
            aux["route_summary"] = (
                self.branch_embedding[branch_idx].mean(dim=0)
                + self.expert_embedding[topi].mean(dim=(0, 1))
            )
        return out.view(b, t, d), aux

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_experts={self.n_experts}, "
            f"topk={self.topk}, shared_experts={self.shared_experts}, "
            f"expert_hidden={self.expert_hidden}, dropout={self.dropout}, "
            f"group_size={self.group_size}, max_rows={self.max_rows}"
        )
