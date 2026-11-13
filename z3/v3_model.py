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
    # Memory is on by default: learned global slots are cheap and act as a
    # persistent, causal context layer.  The hashed n-gram bias is opt-in
    # because it needs an explicit ``build_memory`` pass over training IDs.
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
    # Frontier MTP defaults.  The branch is a real, separately parameterised
    # block per lag, not one shared branch reused for every horizon.
    mtp: int = 3
    mtp_adapter_rank: int = 32
    mtp_block: bool = True
    mtp_block_experts: int = 8
    mtp_block_hidden: int = 1024
    mtp_attention_global: bool = True
    mtp_attention_gate: bool = False
    mtp_teacher_prob: float = 1.0
    mtp_gt_mix: float = 0.0
    mtp_softmax_temperature: float = 1.0
    mtp_gate: bool = True
    mtp_gate_hidden: int = 256
    mtp_gate_prior: float = 0.7
    mtp_gate_use_hidden: bool = True
    # Deep supervision.  ``mtp_aux_layer`` is the legacy single-layer knob;
    # ``mtp_aux_layers`` enables multiple intermediate MTP losses.
    mtp_aux_layer: int = -1
    mtp_aux_layers: tuple[int, ...] = (3, 7)
    mtp_aux_weight: float = 0.2
    mtp_aux_layer_weights: tuple[float, ...] | None = None
    mtp_position_embed: bool = True
    dropout: float = 0.1
    rope_base: float = 10000.0


def _rope_kwargs(cfg: V3Config, head_dim: int) -> dict:
    """Resolve the long-context RoPE configuration once for every block."""
    scaling = cfg.rope_scaling.lower()
    if scaling == "auto":
        scaling = "yarn" if cfg.ctx > cfg.rope_original_ctx else "none"
    factor = float(cfg.rope_factor)
    if factor < 1.0:
        factor = max(1.0, cfg.ctx / max(1, cfg.rope_original_ctx))
    return dict(
        head_dim=head_dim,
        base=cfg.rope_base,
        max_context=max(cfg.max_ctx, cfg.ctx),
        original_context=cfg.rope_original_ctx,
        scaling=scaling,
        factor=factor,
        beta_fast=cfg.rope_beta_fast,
        beta_slow=cfg.rope_beta_slow,
        attention_factor=cfg.rope_attention_factor,
    )


def _causal_mask(
    length: int, device, *, is_global: bool, window: int
) -> torch.Tensor:
    """Causal (and optionally sliding-window) boolean attention mask."""
    ones = torch.ones(length, length, device=device, dtype=torch.bool)
    mask = ones.tril()
    if not is_global:
        mask = mask & ones.triu(1 - max(1, int(window)))
    return mask


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
        self.rope = build_rope(**_rope_kwargs(cfg, self.dh))
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
        mask = _causal_mask(
            q.shape[-2], q.device, is_global=self.is_global, window=self.window
        )
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
            mask = _causal_mask(
                t, x.device, is_global=False, window=self.window
            )
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
    """Trunk-grade causal attention for the MTP branch.

    The branch uses the same stability and long-context machinery as the
    trunk: QK-RMSNorm, YaRN/NTK-capable RoPE, optional GQA, and either full
    causal or sliding-window causal attention.  Keeping these aligned avoids
    a situation where the trunk is stable at long context but the MTP branch
    silently degrades because it still uses vanilla attention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        max_ctx: int = 8192,
        n_kv_heads: int = 0,
        window: int = 512,
        is_global: bool = True,
        rope_kwargs: dict | None = None,
        force_manual: bool = False,
        attn_gate: bool = False,
    ):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.h = n_heads
        self.dh = d_model // n_heads
        self.kv_heads = int(n_kv_heads or n_heads)
        if self.h % self.kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.kv_dim = self.kv_heads * self.dh
        self.kv_group = self.h // self.kv_heads
        self.qkv = nn.Linear(
            d_model, d_model + 2 * self.kv_dim, bias=False
        )
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(self.dh)
        self.k_norm = RMSNorm(self.dh)
        if rope_kwargs is None:
            rope_kwargs = dict(
                max_context=max_ctx, original_context=1024,
                scaling="none",
            )
        rope_kwargs = dict(rope_kwargs)
        rope_kwargs.setdefault("head_dim", self.dh)
        self.rope = build_rope(**rope_kwargs)
        self.drop = dropout
        self.window = max(1, int(window))
        self.is_global = bool(is_global)
        self.force_manual = bool(force_manual)
        self.gate = (
            nn.Linear(d_model, d_model, bias=False) if attn_gate else None
        )

    def _manual(self, q, k, v):
        scores = (q @ k.transpose(-2, -1)) * (self.dh ** -0.5)
        mask = _causal_mask(
            q.shape[-2], q.device, is_global=self.is_global, window=self.window
        )
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = F.dropout(
            torch.softmax(scores, dim=-1), p=self.drop, training=self.training
        )
        return attn @ v

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(
            (d, self.kv_dim, self.kv_dim), dim=2
        )
        q = q.view(b, t, self.h, self.dh).transpose(1, 2)
        k = k.view(b, t, self.kv_heads, self.dh).transpose(1, 2)
        v = v.view(b, t, self.kv_heads, self.dh).transpose(1, 2)
        q = self.rope(self.q_norm(q))
        k = self.rope(self.k_norm(k))
        if self.kv_group > 1:
            k = k.repeat_interleave(self.kv_group, dim=1)
            v = v.repeat_interleave(self.kv_group, dim=1)
        if self.force_manual or x.device.type == "xla":
            y = self._manual(q, k, v)
        elif self.is_global:
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.drop if self.training else 0.0,
                is_causal=True,
            )
        else:
            mask = _causal_mask(
                t, x.device, is_global=False, window=self.window
            )
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask,
                dropout_p=self.drop if self.training else 0.0,
            )
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        y = self.out(y)
        if self.gate is not None:
            y = y * torch.sigmoid(self.gate(x))
        return y


class MTPBlock(nn.Module):
    """One-block future branch with trunk-grade attention and its own MoE."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        hidden: int = 1024,
        n_experts: int = 8,
        topk: int = 2,
        shared: int = 1,
        dropout: float = 0.1,
        max_ctx: int = 8192,
        n_kv_heads: int = 0,
        window: int = 512,
        is_global: bool = True,
        rope_kwargs: dict | None = None,
        force_manual: bool = False,
        attn_gate: bool = False,
        group_size: int = 4,
        max_rows: int = 512,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MTPAttention(
            d_model,
            n_heads,
            dropout=dropout,
            max_ctx=max_ctx,
            n_kv_heads=n_kv_heads,
            window=window,
            is_global=is_global,
            rope_kwargs=rope_kwargs,
            force_manual=force_manual,
            attn_gate=attn_gate,
        )
        self.norm2 = RMSNorm(d_model)
        self.moe = VectorizedSparseMoE(
            d_model=d_model,
            n_experts=n_experts,
            topk=topk,
            shared_experts=shared,
            expert_hidden=hidden,
            dropout=dropout,
            group_size=group_size,
            max_rows=max_rows,
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
        rope_kwargs = _rope_kwargs(cfg, cfg.d_model // cfg.n_heads)
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
                    n_kv_heads=cfg.n_kv_heads or cfg.n_heads,
                    window=cfg.window,
                    is_global=(
                        cfg.mtp_attention_global
                        or (lag % cfg.global_every == 0)
                    ),
                    rope_kwargs=rope_kwargs,
                    force_manual=cfg.force_manual_attention,
                    attn_gate=cfg.mtp_attention_gate,
                    group_size=cfg.moe_group_size,
                    max_rows=cfg.moe_max_rows,
                )
                for lag in range(n_mtp)
            ]
            if cfg.mtp_block else []
        )
        self.mtp_block = self.mtp_blocks[0] if self.mtp_blocks else None
        self.mtp_now = nn.ParameterList(
            [nn.Parameter(torch.zeros(cfg.d_model)) for _ in range(n_mtp)]
        )
        if cfg.mtp_position_embed and n_mtp:
            self.mtp_position_embed = nn.Embedding(
                max(cfg.max_ctx, cfg.ctx), cfg.d_model
            )
            nn.init.zeros_(self.mtp_position_embed.weight)
        else:
            self.mtp_position_embed = None
        self.mtp_next_norm = RMSNorm(cfg.d_model) if cfg.mtp_block else None
        self.mtp_cond_proj = nn.ModuleList(
            [nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)
             for _ in range(n_mtp)]
            if cfg.mtp_block else []
        )
        # Deep supervision can read several intermediate residual streams.
        aux_layers = [
            int(layer) for layer in cfg.mtp_aux_layers
            if 0 <= int(layer) < cfg.n_layers
        ]
        if 0 <= cfg.mtp_aux_layer < cfg.n_layers:
            aux_layers.append(int(cfg.mtp_aux_layer))
        aux_layers = sorted(set(aux_layers))
        self.mtp_aux_layers = tuple(aux_layers)
        self.mtp_aux_norms = nn.ModuleList(
            [RMSNorm(cfg.d_model) for _ in aux_layers]
        )
        self.mtp_aux_norm = (
            self.mtp_aux_norms[0] if len(self.mtp_aux_norms) else None
        )
        weights = cfg.mtp_aux_layer_weights
        if weights is None or len(weights) != len(aux_layers):
            weights = tuple(1.0 for _ in aux_layers)
        self.mtp_aux_layer_weights = tuple(float(w) for w in weights)
        gate_input = 3 + (cfg.d_model if cfg.mtp_gate_use_hidden else 0)
        self.mtp_gate = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(gate_input, cfg.mtp_gate_hidden),
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
        # Start the reliability gate near the observed t+1 accuracy rather
        # than at 0.5; the branch is identity at step zero, so this only
        # affects the gate's initial gradient conditioning.
        prior = min(0.99, max(0.01, float(self.cfg.mtp_gate_prior)))
        prior_logit = math.log(prior / (1.0 - prior))
        for gate in self.mtp_gate:
            with torch.no_grad():
                gate[-1].bias.fill_(prior_logit)

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
        aux_hidden: dict[int, torch.Tensor] = {}
        for layer_index, block in enumerate(self.blocks):
            x, lb, z = block(x)
            if layer_index in self.mtp_aux_layers:
                aux_hidden[layer_index] = x
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

        # Deep supervision: each selected intermediate stream predicts t+2
        # through the same small MTP head.  The residual stream is causal, so
        # no future token can leak into an earlier position.
        mtp_aux_logits: list[torch.Tensor] = []
        if self.mtp_heads and self.mtp_aux_norms:
            aux_head = self.mtp_heads[0]
            for layer, norm in zip(self.mtp_aux_layers, self.mtp_aux_norms):
                hidden = aux_hidden.get(layer)
                if hidden is not None:
                    mtp_aux_logits.append(aux_head(norm(hidden)))

        mtp_block_lb: list[torch.Tensor] = []
        mtp_block_rz: list[torch.Tensor] = []
        if self.mtp_blocks:
            # Predicted t+1 distribution.  This is a distribution in both
            # train and eval; ground truth is only ever a soft teacher term.
            temperature = max(1e-3, float(self.cfg.mtp_softmax_temperature))
            pred_probs = F.softmax(main.detach() / temperature, dim=-1)
            gt_mix = self.cfg.mtp_gt_mix if mtp_gt_mix is None else mtp_gt_mix
            gt_mix = min(1.0, max(0.0, float(gt_mix)))
            teacher_prob = (
                self.cfg.mtp_teacher_prob
                if mtp_teacher_prob is None else mtp_teacher_prob
            )
            teacher_prob = min(1.0, max(0.0, float(teacher_prob)))
            next_ids = torch.cat([tokens[:, 1:], tokens[:, -1:]], dim=1)
            gt = F.one_hot(
                next_ids.clamp_min(0), num_classes=self.cfg.vocab_size
            ).to(dtype=pred_probs.dtype)

            explicit_teacher = mtp_next_tokens is not None
            if explicit_teacher:
                # Diagnostics may inject a known/corrupted t+1 target.  Keep
                # the predicted distribution in the mix even then; a hard
                # one-hot is never the sole branch input.
                explicit = F.one_hot(
                    mtp_next_tokens.clamp_min(0).clamp_max(self.cfg.vocab_size - 1),
                    num_classes=self.cfg.vocab_size,
                ).to(dtype=pred_probs.dtype)
                mix = 1.0 if mtp_gt_mix is None else gt_mix
                mix = min(0.95, max(0.0, mix))
                next_soft = (1.0 - mix) * pred_probs + mix * explicit
            elif mtp_teacher and gt_mix > 0.0:
                mix = gt_mix
                if self.training and teacher_prob < 1.0:
                    keep = (
                        torch.rand(tokens.size(0), 1, 1, device=tokens.device)
                        < teacher_prob
                    ).to(dtype=pred_probs.dtype)
                    mix = mix * keep
                next_soft = (1.0 - mix) * pred_probs + mix * gt
            else:
                next_soft = pred_probs
            next_emb = self.mtp_next_norm(next_soft @ self.tok.weight)

            ent = -(pred_probs * torch.log(pred_probs.clamp_min(1e-9))).sum(
                dim=-1, keepdim=True
            )
            top2 = pred_probs.topk(2, dim=-1).values
            maxp = top2[..., :1]
            margin = (top2[..., 0:1] - top2[..., 1:2])
            gate_parts = [ent, maxp, margin]
            if self.cfg.mtp_gate_use_hidden:
                gate_parts.append(x)
            gate_features = torch.cat(gate_parts, dim=-1)
            valid_t1 = tokens[:, 1:] >= 0
            t1_ok = (
                main.argmax(dim=-1)[:, :-1] == tokens[:, 1:]
            ).to(dtype=pred_probs.dtype)
            positions = torch.arange(tokens.size(1), device=tokens.device)
            position_anchor = (
                self.mtp_position_embed(positions).to(dtype=x.dtype).unsqueeze(0)
                if self.mtp_position_embed is not None else 0.0
            )
            mtp = []
            gate_losses = []
            for idx, head in enumerate(self.mtp_heads):
                cond = (
                    self.mtp_cond_proj[idx](
                        torch.cat([x, next_emb], dim=-1)
                    )
                    + self.mtp_now[idx]
                    + position_anchor
                )
                branch, block_aux = self.mtp_blocks[idx](cond)
                if block_aux:
                    if block_aux.get("load_balance") is not None:
                        mtp_block_lb.append(block_aux["load_balance"])
                    if block_aux.get("router_z") is not None:
                        mtp_block_rz.append(block_aux["router_z"])
                branch_logits = head(branch)
                base_logits = head(x)
                if self.mtp_gate:
                    gate_logits = self.mtp_gate[idx](gate_features)
                    gate = torch.sigmoid(gate_logits)
                    mtp.append(
                        gate * branch_logits + (1.0 - gate) * base_logits
                    )
                    gate_target = gate_logits[:, :-1].squeeze(-1)
                    gate_loss = F.binary_cross_entropy_with_logits(
                        gate_target, t1_ok, reduction="none"
                    )
                    if valid_t1.any():
                        gate_loss = gate_loss[valid_t1].mean()
                    else:
                        gate_loss = gate_loss.sum() * 0.0
                    gate_losses.append(gate_loss)
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
            gate_loss = None
        aux = dict(mem_aux)
        if mtp_aux_logits:
            aux["mtp_aux_logits"] = mtp_aux_logits
            aux["mtp_aux_layers"] = self.mtp_aux_layers
            aux["mtp_aux_layer_weights"] = self.mtp_aux_layer_weights
        if self.mtp_blocks and gate_loss is not None:
            aux["mtp_gate_loss"] = gate_loss
        if mtp_block_lb:
            aux["mtp_load_balance"] = torch.stack(mtp_block_lb).mean()
        if mtp_block_rz:
            aux["mtp_router_z"] = torch.stack(mtp_block_rz).mean()
        if lbs:
            aux["load_balance"] = torch.stack(lbs).mean()
            aux["router_z"] = torch.stack(zs).mean()
        return main, mtp, aux

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
