#!/usr/bin/env python3
"""Vectorized top-k sparse MoE dispatch.

This replaces the per-expert Python loop in ``v3_model.SparseMoE`` with a
single gather + einsum dispatch:

  * router logits: Linear(D, E), no bias
  * top-k selection per token, softmax over the selected logits
  * expert SwiGLU weights stored as dense Parameters, not a ModuleList:
        gate_w (E, D, H), up_w (E, D, H), down_w (E, H, D)
  * per-token top-k weights are gathered with ``index_select`` and applied
    with einsum, so there is no Python loop over experts in forward
  * shared experts are one batched shared FFN (S, D, H) / (S, H, D)

The gathered weight tensor has shape (M * K, D, H). That is intentional for
correctness and readability; a production kernel would use a grouped GEMM or
permute-and-bmm dispatch to avoid materializing expert weights per token.

API
---
    moe = VectorizedSparseMoE(d_model=..., n_experts=..., topk=...,
                              shared_experts=..., expert_hidden=...,
                              dropout=...)
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
    """Top-k routed SwiGLU MoE with vectorized expert dispatch.

    Args:
        d_model: token width D.
        n_experts: number of routed experts E.
        topk: experts selected per token K.
        shared_experts: number of always-on shared experts S.
        expert_hidden: per-expert SwiGLU hidden width H. Defaults to 4 * D.
        dropout: dropout probability on expert outputs.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        topk: int = 2,
        shared_experts: int = 0,
        expert_hidden: int | None = None,
        dropout: float = 0.0,
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

        self.d_model = d_model
        self.n_experts = n_experts
        self.topk = topk
        self.shared_experts = shared_experts
        self.expert_hidden = expert_hidden
        self.dropout = dropout

        self.router = nn.Linear(d_model, n_experts, bias=False)

        # Routed expert SwiGLU weights, one dense Parameter per projection.
        self.gate_w = nn.Parameter(torch.empty(n_experts, d_model, expert_hidden))
        self.up_w = nn.Parameter(torch.empty(n_experts, d_model, expert_hidden))
        self.down_w = nn.Parameter(torch.empty(n_experts, expert_hidden, d_model))

        # Shared experts as one batched FFN.
        if shared_experts > 0:
            self.shared_gate_w = nn.Parameter(
                torch.empty(shared_experts, d_model, expert_hidden)
            )
            self.shared_up_w = nn.Parameter(
                torch.empty(shared_experts, d_model, expert_hidden)
            )
            self.shared_down_w = nn.Parameter(
                torch.empty(shared_experts, expert_hidden, d_model)
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
        )

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.router.weight)
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

    def route(self, flat: torch.Tensor):
        """Route flattened tokens.

        Args:
            flat: (M, D) tokens.

        Returns:
            logits: (M, E) router logits.
            probs: (M, E) full softmax router distribution.
            topv: (M, K) selected logits.
            topi: (M, K) selected expert indices.
            weights: (M, K) softmax over selected logits (sums to 1).
        """
        logits = self.router(flat)
        probs = F.softmax(logits, dim=-1)
        topv, topi = torch.topk(logits, k=self.topk, dim=-1)
        weights = F.softmax(topv, dim=-1)
        return logits, probs, topv, topi, weights

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: (B, T, D) token states.

        Returns:
            out: (B, T, D) MoE output.
            aux: dict with scalar ``load_balance`` and ``router_z`` terms.
        """
        if x.ndim != 3:
            raise ValueError(f"x must be (B, T, D), got shape {tuple(x.shape)}")
        b, t, d = x.shape
        if d != self.d_model:
            raise ValueError(f"expected last dim {self.d_model}, got {d}")

        flat = x.reshape(-1, d)
        m = flat.shape[0]
        logits, probs, _, topi, weights = self.route(flat)

        # Gather the top-k expert weights for every token: (M*K, ...).
        sel = topi.reshape(-1)
        gate = self.gate_w.index_select(0, sel)
        up = self.up_w.index_select(0, sel)
        down = self.down_w.index_select(0, sel)
        xr = flat.repeat_interleave(self.topk, dim=0)

        # SwiGLU: silu(x @ W_gate) * (x @ W_up) @ W_down, batched over M*K.
        h_gate = torch.einsum("md,mdh->mh", xr, gate)
        h_up = torch.einsum("md,mdh->mh", xr, up)
        y = torch.einsum("mh,mhd->md", F.silu(h_gate) * h_up, down)
        y = F.dropout(y, p=self.dropout, training=self.training)
        y = y.view(m, self.topk, d)
        out = (weights.unsqueeze(-1) * y).sum(dim=1)

        # Shared experts: one batched SwiGLU over all tokens and shared slots.
        if self.shared_gate_w is not None:
            sh_gate = torch.einsum("md,sdh->smh", flat, self.shared_gate_w)
            sh_up = torch.einsum("md,sdh->smh", flat, self.shared_up_w)
            shared = torch.einsum(
                "smh,shd->smd", F.silu(sh_gate) * sh_up, self.shared_down_w
            )
            shared = F.dropout(shared, p=self.dropout, training=self.training)
            out = out + shared.sum(dim=0)

        # Switch-style load balance and router z-loss.
        mask = torch.zeros_like(logits).scatter_(1, topi, 1.0)
        f = mask.mean(dim=0)
        p = probs.mean(dim=0)
        load_balance = self.n_experts * (f * p).sum()
        router_z = torch.logsumexp(logits, dim=-1).pow(2).mean()

        return out.view(b, t, d), {
            "load_balance": load_balance,
            "router_z": router_z,
        }

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_experts={self.n_experts}, "
            f"topk={self.topk}, shared_experts={self.shared_experts}, "
            f"expert_hidden={self.expert_hidden}, dropout={self.dropout}"
        )
