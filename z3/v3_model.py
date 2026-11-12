#!/usr/bin/env python3
"""nanlmv3 v3 — frontier-style sparse hybrid prototype.

Architecture changes over v2:

  * QK-RMSNorm inside attention for stability
  * RoPE with configurable base and optional long-context scaling
  * sliding-window local attention + periodic full-causal global layers
  * sparse MoE with 32 routed experts, top-2, plus 2 shared experts
  * multi-token prediction (MTP) heads for better data efficiency
  * pre-LN / RMSNorm / SwiGLU throughout
  * tied main output embedding; separate small MTP heads

This is an architecture prototype, not a frontier-scale training claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class V3Config:
    vocab_size: int = 32000
    ctx: int = 1024
    d_model: int = 384
    n_layers: int = 12
    n_heads: int = 12
    swiglu: int = 1536
    window: int = 512
    global_every: int = 4
    moe_layers: int = 4
    n_experts: int = 32
    topk: int = 2
    shared_experts: int = 2
    expert_hidden: int = 1536
    mtp: int = 2
    dropout: float = 0.1
    rope_base: float = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class RoPE(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, x):
        t = torch.arange(x.shape[-2], device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        x1, x2 = x.chunk(2, dim=-1)
        return x * cos + torch.cat([-x2, x1], dim=-1) * sin


class Attention(nn.Module):
    def __init__(self, cfg: V3Config, layer: int):
        super().__init__()
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(self.dh)
        self.k_norm = RMSNorm(self.dh)
        self.rope = RoPE(self.dh, cfg.rope_base)
        self.drop = cfg.dropout
        self.is_global = layer % cfg.global_every == 0
        self.window = cfg.window

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q = q.view(b, t, self.h, self.dh).transpose(1, 2)
        k = k.view(b, t, self.h, self.dh).transpose(1, 2)
        v = v.view(b, t, self.h, self.dh).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = self.rope(q)
        k = self.rope(k)
        if self.is_global:
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.drop if self.training else 0.0, is_causal=True
            )
        else:
            ones = torch.ones(t, t, device=x.device, dtype=torch.bool)
            mask = ones.tril() & ones.triu(1 - self.window)
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.drop if self.training else 0.0
            )
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        return self.out(y)


class SwiGLU(nn.Module):
    def __init__(self, d_in: int, hidden: int, d_out: int | None = None, dropout: float = 0.1):
        super().__init__()
        d_out = d_in if d_out is None else d_out
        self.gate = nn.Linear(d_in, hidden, bias=False)
        self.up = nn.Linear(d_in, hidden, bias=False)
        self.down = nn.Linear(hidden, d_out, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class SparseMoE(nn.Module):
    def __init__(self, cfg: V3Config):
        super().__init__()
        self.cfg = cfg
        self.gate = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(cfg.d_model, cfg.expert_hidden, cfg.d_model, cfg.dropout)
             for _ in range(cfg.n_experts)]
        )
        self.shared = nn.ModuleList(
            [SwiGLU(cfg.d_model, cfg.expert_hidden, cfg.d_model, cfg.dropout)
             for _ in range(cfg.shared_experts)]
        )

    def forward(self, x):
        b, t, d = x.shape
        flat = x.reshape(-1, d)
        logits = self.gate(flat)
        probs = F.softmax(logits, dim=-1)
        topv, topi = torch.topk(logits, k=self.cfg.topk, dim=-1)
        w = F.softmax(topv, dim=-1)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            hit = (topi == e).nonzero(as_tuple=False)
            if hit.numel() == 0:
                continue
            rows, slots = hit[:, 0], hit[:, 1]
            out[rows] += w[rows, slots].unsqueeze(-1) * expert(flat[rows])
        for expert in self.shared:
            out += expert(flat)
        # Switch-style load balance + router z-loss.
        mask = torch.zeros_like(logits).scatter_(1, topi, 1.0)
        f = mask.mean(dim=0)
        p = probs.mean(dim=0)
        lb = self.cfg.n_experts * (f * p).sum()
        z = torch.logsumexp(logits, dim=-1).pow(2).mean()
        return out.view(b, t, d), lb, z


class Block(nn.Module):
    def __init__(self, cfg: V3Config, layer: int):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg, layer)
        self.n2 = RMSNorm(cfg.d_model)
        self.is_moe = layer >= cfg.n_layers - cfg.moe_layers
        self.mlp = (
            SparseMoE(cfg) if self.is_moe
            else SwiGLU(cfg.d_model, cfg.swiglu, cfg.d_model, cfg.dropout)
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.n1(x)))
        h = self.n2(x)
        if self.is_moe:
            h, lb, z = self.mlp(h)
            x = x + self.drop(h)
            return x, lb, z
        return x + self.mlp(h), None, None


class V3LM(nn.Module):
    def __init__(self, cfg: V3Config):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        # Tied main head; separate MTP heads.
        self.mtp_heads = nn.ModuleList(
            [nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
             for _ in range(max(0, cfg.mtp - 1))]
        )
        self.apply(self._init)
        nn.init.normal_(self.tok.weight, mean=0.0, std=0.02)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, tokens):
        x = self.tok(tokens)
        lbs, zs = [], []
        for block in self.blocks:
            x, lb, z = block(x)
            if lb is not None:
                lbs.append(lb)
                zs.append(z)
        x = self.norm(x)
        main = x @ self.tok.weight.T
        mtp = [head(x) for head in self.mtp_heads]
        aux = {}
        if lbs:
            aux["load_balance"] = torch.stack(lbs).mean()
            aux["router_z"] = torch.stack(zs).mean()
        return main, mtp, aux

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
