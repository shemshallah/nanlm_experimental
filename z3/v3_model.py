#!/usr/bin/env python3
"""nanlmv3 v3 — frontier-style sparse hybrid prototype.

Architecture changes over v2:

  * QK-RMSNorm inside attention for stability
  * RoPE with YaRN/NTK scaling for an 8k position range
  * sliding-window local attention + periodic full-causal global layers
  * sparse MoE with 32 routed experts, top-2, plus 2 shared experts
  * grouped, vectorized top-k expert dispatch with bounded memory
  * multi-token prediction (MTP) heads for better data efficiency
  * optional learned memory cross-attention + sparse hashed n-gram bias
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

try:  # support both `python z3/train_v3.py` and `import z3.v3_model`
    from .memory_attention import MemoryCrossAttention, V3MemoryPath
    from .moe_vectorized import VectorizedSparseMoE
    from .rope_scaling import build_rope
except ImportError:  # pragma: no cover - script execution path
    from memory_attention import MemoryCrossAttention, V3MemoryPath
    from moe_vectorized import VectorizedSparseMoE
    from rope_scaling import build_rope


@dataclass
class V3Config:
    vocab_size: int = 32000
    ctx: int = 1024
    # Long-context RoPE.  ``max_ctx`` is the architecture limit; with
    # ``rope_scaling="auto"`` the model uses YaRN when ctx exceeds the
    # original 1k training window, and standard RoPE otherwise.
    max_ctx: int = 8192
    rope_scaling: str = "auto"  # auto | none | ntk | yarn
    rope_factor: float = 8.0
    rope_original_ctx: int = 1024
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    rope_attention_factor: float | None = None
    # Optional memory path: learned global slots plus a sparse hashed
    # n-gram logit bias.  Off by default; build the n-gram store with
    # ``V3LM.build_memory(tokens)`` before training/evaluation.
    use_memory: bool = True
    use_ngram_memory: bool = False
    memory_slots: int = 256
    memory_heads: int = 4
    memory_dropout: float = 0.0
    memory_size: int = 1 << 20
    memory_order: int = 4
    memory_min_count: int = 5
    memory_alpha_init: float = 0.1
    d_model: int = 384
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4  # GQA by default; 0 = full MHA
    swiglu: int = 1536
    window: int = 512
    global_every: int = 4
    moe_layers: int = 4
    n_experts: int = 32
    topk: int = 2
    shared_experts: int = 2
    expert_hidden: int = 1536
    moe_group_size: int = 4
    moe_max_rows: int = 512
    moe_dense_dispatch: bool = False
    force_manual_attention: bool = False
    attn_gate: bool = True
    mtp_adapter_rank: int = 0
    mtp_block: bool = True
    mtp_block_experts: int = 8
    mtp_block_hidden: int = 1024
    mtp_teacher_prob: float = 1.0
    mtp_gt_mix: float = 0.0
    mtp_gate: bool = True
    mtp_gate_hidden: int = 32
    mtp_aux_layer: int = -1
    mtp_aux_weight: float = 0.2
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


class Attention(nn.Module):
    def __init__(self, cfg: V3Config, layer: int):
        super().__init__()
        self.h = cfg.n_heads
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.dh = cfg.d_model // cfg.n_heads
        self.kv_heads = int(cfg.n_kv_heads or cfg.n_heads)
        if cfg.n_heads % self.kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.kv_dim = self.kv_heads * self.dh
        self.kv_group = self.h // self.kv_heads
        self.qkv = nn.Linear(
            cfg.d_model, cfg.d_model + 2 * self.kv_dim, bias=False
        )
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(self.dh)
        self.k_norm = RMSNorm(self.dh)
        scaling = cfg.rope_scaling.lower()
        if scaling == "auto":
            scaling = "yarn" if cfg.ctx > cfg.rope_original_ctx else "none"
        factor = cfg.rope_factor
        if factor < 1.0:
            factor = max(1.0, cfg.ctx / max(1, cfg.rope_original_ctx))
        self.rope = build_rope(
            self.dh,
            base=cfg.rope_base,
            max_context=max(cfg.max_ctx, cfg.ctx),
            original_context=cfg.rope_original_ctx,
            scaling=scaling,
            factor=factor,
            beta_fast=cfg.rope_beta_fast,
            beta_slow=cfg.rope_beta_slow,
            attention_factor=cfg.rope_attention_factor,
        )
        self.drop = cfg.dropout
        self.is_global = layer % cfg.global_every == 0
        self.window = cfg.window
        self.force_manual = bool(cfg.force_manual_attention)
        self.gate = (
            nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            if cfg.attn_gate else None
        )

    def _manual_attention(self, q, k, v):
        """Dense attention fallback for backends without a working SDPA."""
        scale = self.dh ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale
        t = q.shape[-2]
        ones = torch.ones(t, t, device=q.device, dtype=torch.bool)
        mask = ones.tril() if self.is_global else (ones.tril() & ones.triu(1 - self.window))
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.drop if self.training else 0.0)
        return attn @ v

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(
            (d, self.kv_dim, self.kv_dim), dim=2
        )
        q = q.view(b, t, self.h, self.dh).transpose(1, 2)
        k = k.view(b, t, self.kv_heads, self.dh).transpose(1, 2)
        v = v.view(b, t, self.kv_heads, self.dh).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = self.rope(q)
        k = self.rope(k)
        if self.kv_group > 1:
            # Dense training path: expand KV heads for the attention kernel.
            # Inference can cache the smaller KV tensors instead.
            k = k.repeat_interleave(self.kv_group, dim=1)
            v = v.repeat_interleave(self.kv_group, dim=1)
        if self.force_manual or x.device.type == "xla":
            y = self._manual_attention(q, k, v)
        elif self.is_global:
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
        y = self.out(y)
        if self.gate is not None:
            y = y * torch.sigmoid(self.gate(x))
        return y


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
    """Backward-compatible wrapper around the grouped top-k MoE.

    The old implementation looped over every expert in Python.  The inner
    module dispatches expert groups to their routed tokens with batched
    einsums and bounded row chunks, while this wrapper keeps the
    ``(out, load_balance, router_z)`` contract used by Block.
    """

    def __init__(self, cfg: V3Config):
        super().__init__()
        self.cfg = cfg
        self.inner = VectorizedSparseMoE.from_config(cfg)

    def forward(self, x):
        out, aux = self.inner(x)
        return out, aux["load_balance"], aux["router_z"]


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


class MTPAdapter(nn.Module):
    """Small residual adapter that specializes frozen-trunk features for MTP."""

    def __init__(self, d_model: int, rank: int):
        super().__init__()
        if rank <= 0:
            raise ValueError("MTP adapter rank must be positive")
        self.norm = RMSNorm(d_model)
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.xavier_uniform_(self.up.weight)
        self.scale = nn.Parameter(torch.tensor(1e-1))

    def forward(self, x):
        return x + self.scale * self.up(F.silu(self.down(self.norm(x))))


class MTPAttention(nn.Module):
    """Trunk-grade causal attention for the MTP branch."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 max_ctx: int = 8192):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.h = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(self.dh)
        self.k_norm = RMSNorm(self.dh)
        self.rope = build_rope(
            self.dh, max_context=max_ctx, original_context=1024,
            scaling="none",
        )
        self.drop = dropout

    def _manual(self, q, k, v):
        scores = (q @ k.transpose(-2, -1)) * (self.dh ** -0.5)
        t = q.shape[-2]
        mask = torch.ones(t, t, device=q.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = F.dropout(
            torch.softmax(scores, dim=-1), p=self.drop, training=self.training
        )
        return attn @ v

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q = q.view(b, t, self.h, self.dh).transpose(1, 2)
        k = k.view(b, t, self.h, self.dh).transpose(1, 2)
        v = v.view(b, t, self.h, self.dh).transpose(1, 2)
        q = self.rope(self.q_norm(q))
        k = self.rope(self.k_norm(k))
        if x.device.type == "xla":
            y = self._manual(q, k, v)
        else:
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.drop if self.training else 0.0,
                is_causal=True,
            )
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        return self.out(y)


class MTPBlock(nn.Module):
    """One-block future branch with trunk-grade attention and its own MoE."""

    def __init__(self, d_model: int, n_heads: int, hidden: int = 1024,
                 n_experts: int = 8, topk: int = 2, shared: int = 1,
                 dropout: float = 0.1, max_ctx: int = 8192):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MTPAttention(
            d_model, n_heads, dropout=dropout, max_ctx=max_ctx
        )
        self.norm2 = RMSNorm(d_model)
        self.moe = VectorizedSparseMoE(
            d_model=d_model,
            n_experts=n_experts,
            topk=topk,
            shared_experts=shared,
            expert_hidden=hidden,
            dropout=dropout,
            group_size=4,
            max_rows=512,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        y, aux = self.moe(self.norm2(x))
        x = x + self.drop(y)
        return x, aux


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
        self.mtp_adapters = nn.ModuleList(
            [MTPAdapter(cfg.d_model, cfg.mtp_adapter_rank)
             if cfg.mtp_adapter_rank > 0 else nn.Identity()
             for _ in range(max(0, cfg.mtp - 1))]
        )
        n_mtp = max(0, cfg.mtp - 1)
        self.mtp_blocks = nn.ModuleList(
            [
                MTPBlock(
                    cfg.d_model,
                    cfg.n_heads,
                    hidden=cfg.mtp_block_hidden,
                    n_experts=cfg.mtp_block_experts,
                    topk=2,
                    shared=1,
                    dropout=cfg.dropout,
                    max_ctx=max(cfg.max_ctx, cfg.ctx),
                )
                for _ in range(n_mtp)
            ]
            if cfg.mtp_block else []
        )
        self.mtp_block = self.mtp_blocks[0] if self.mtp_blocks else None
        self.mtp_now = nn.ParameterList(
            [nn.Parameter(torch.zeros(cfg.d_model)) for _ in range(n_mtp)]
        )
        self.mtp_next_norm = RMSNorm(cfg.d_model) if cfg.mtp_block else None
        self.mtp_cond_proj = nn.ModuleList(
            [nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)
             for _ in range(n_mtp)]
            if cfg.mtp_block else []
        )
        self.mtp_aux_norm = (
            RMSNorm(cfg.d_model)
            if 0 <= cfg.mtp_aux_layer < cfg.n_layers else None
        )
        self.mtp_gate = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2, cfg.mtp_gate_hidden),
                    nn.SiLU(),
                    nn.Linear(cfg.mtp_gate_hidden, 1),
                )
                for _ in range(n_mtp)
            ]
            if cfg.mtp_block and cfg.mtp_gate else []
        )
        self.memory_path = None
        self.memory_cross = None
        self.memory_scale = None
        if cfg.use_memory:
            mem_heads = cfg.memory_heads or cfg.n_heads
            if cfg.d_model % mem_heads != 0:
                raise ValueError("d_model must be divisible by memory_heads")
            if cfg.use_ngram_memory:
                self.memory_scale = nn.Parameter(torch.ones(1))
                self.memory_path = V3MemoryPath(
                    cfg.d_model,
                    mem_heads,
                    cfg.vocab_size,
                    n_slots=cfg.memory_slots,
                    memory_size=cfg.memory_size,
                    order=cfg.memory_order,
                    min_count=cfg.memory_min_count,
                    alpha_init=cfg.memory_alpha_init,
                    dropout=cfg.memory_dropout,
                )
            else:
                self.memory_cross = MemoryCrossAttention(
                    cfg.d_model,
                    mem_heads,
                    n_slots=cfg.memory_slots,
                    dropout=cfg.memory_dropout,
                )
        self.apply(self._init)
        nn.init.normal_(self.tok.weight, mean=0.0, std=0.02)
        for proj, block in zip(self.mtp_cond_proj, self.mtp_blocks):
            with torch.no_grad():
                proj.weight.zero_()
                proj.weight[:, : self.cfg.d_model].copy_(torch.eye(self.cfg.d_model))
                proj.weight[:, self.cfg.d_model:].zero_()
            # Identity branch at step zero.
            nn.init.zeros_(block.attn.out.weight)
            block.moe.down_w.data.zero_()
            if block.moe.shared_down_w is not None:
                block.moe.shared_down_w.data.zero_()

    def build_memory(self, tokens, max_tokens: int | None = None) -> "V3LM":
        """Populate the hashed n-gram store from token IDs (no-op if disabled)."""
        if self.memory_path is not None:
            self.memory_path.build_memory(tokens, max_tokens=max_tokens)
        return self

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, tokens, mtp_teacher: bool = False,
                mtp_teacher_prob: float | None = None,
                mtp_next_tokens=None,
                mtp_gt_mix: float | None = None):
        x = self.tok(tokens)
        lbs, zs = [], []
        aux_hidden = None
        for layer_index, block in enumerate(self.blocks):
            x, lb, z = block(x)
            if layer_index == self.cfg.mtp_aux_layer:
                aux_hidden = x
            if lb is not None:
                lbs.append(lb)
                zs.append(z)
        mem_aux = {}
        if self.memory_path is not None:
            x, mem_aux = self.memory_path.cross(x)
        elif self.memory_cross is not None:
            x, mem_aux = self.memory_cross(x)
        x = self.norm(x)
        main = x @ self.tok.weight.T
        if self.memory_path is not None:
            # Sparse n-gram bias with a learned global scale.
            bias = self.memory_path.logit_bias(tokens)
            bias.value = bias.value * self.memory_scale
            bias.add_(main)
        mtp_aux_logits = None
        if self.mtp_aux_norm is not None and aux_hidden is not None and self.mtp_heads:
            mtp_aux_logits = self.mtp_heads[0](self.mtp_aux_norm(aux_hidden))
        if self.mtp_blocks:
            # Predicted t+1 distribution, mixed with ground truth during
            # training.  The same distribution is used at inference.
            pred_probs = F.softmax(main.detach(), dim=-1)
            gt_mix = self.cfg.mtp_gt_mix if mtp_gt_mix is None else mtp_gt_mix
            if mtp_next_tokens is not None:
                next_soft = F.one_hot(
                    mtp_next_tokens, num_classes=self.cfg.vocab_size
                ).to(dtype=pred_probs.dtype)
            elif mtp_teacher and gt_mix > 0.0:
                next_ids = torch.cat([tokens[:, 1:], tokens[:, -1:]], dim=1)
                gt = F.one_hot(
                    next_ids, num_classes=self.cfg.vocab_size
                ).to(dtype=pred_probs.dtype)
                next_soft = (1.0 - gt_mix) * pred_probs + gt_mix * gt
            else:
                next_soft = pred_probs
            next_emb = self.mtp_next_norm(next_soft @ self.tok.weight)
            ent = -(pred_probs * torch.log(pred_probs.clamp_min(1e-9))).sum(
                dim=-1, keepdim=True
            )
            maxp = pred_probs.max(dim=-1, keepdim=True).values
            gate_features = torch.cat([ent, maxp], dim=-1)
            t1_ok = (
                main.argmax(dim=-1)[:, :-1] == tokens[:, 1:]
            ).to(dtype=pred_probs.dtype)
            mtp = []
            gate_losses = []
            for idx, head in enumerate(self.mtp_heads):
                cond = (
                    self.mtp_cond_proj[idx](
                        torch.cat([x, next_emb], dim=-1)
                    )
                    + self.mtp_now[idx]
                )
                branch, _ = self.mtp_blocks[idx](cond)
                branch_logits = head(branch)
                base_logits = head(x)
                if self.mtp_gate:
                    gate_logits = self.mtp_gate[idx](gate_features)
                    gate = torch.sigmoid(gate_logits)
                    mtp.append(
                        gate * branch_logits + (1.0 - gate) * base_logits
                    )
                    gate_losses.append(
                        F.binary_cross_entropy_with_logits(
                            gate_logits[:, :-1].squeeze(-1), t1_ok
                        )
                    )
                else:
                    mtp.append(branch_logits)
            gate_loss = (
                torch.stack(gate_losses).mean() if gate_losses else None
            )
        else:
            mtp = [
                head(adapter(x))
                for head, adapter in zip(self.mtp_heads, self.mtp_adapters)
            ]
        aux = dict(mem_aux)
        if mtp_aux_logits is not None:
            aux["mtp_aux_logits"] = mtp_aux_logits
        if self.mtp_blocks and gate_loss is not None:
            aux["mtp_gate_loss"] = gate_loss
        if lbs:
            aux["load_balance"] = torch.stack(lbs).mean()
            aux["router_z"] = torch.stack(zs).mean()
        return main, mtp, aux

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
