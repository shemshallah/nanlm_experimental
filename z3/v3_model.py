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
  * tied main output embedding; optional factorized 64k embedding
  * optional MTP output tied to the main token geometry (64k-friendly)

This is an architecture prototype, not a frontier-scale training claim.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover - older torch
    SDPBackend = None  # type: ignore[assignment]
    sdpa_kernel = None  # type: ignore[assignment]

try:  # support both `python z3/train_v3.py` and `import z3.v3_model`
    from .combinatorial_objects import CombinatorialMemory
    from .concept_loop import (
        MAX_TOKEN_BYTES, CompositionHead, CompositionState,
        CompositionTokenizer, ConceptLoop, IncrementalCompositionDecoder,
        SpanSidecar,
    )
    from .memory_attention import MemoryCrossAttention, V3MemoryPath
    from .moe_vectorized import VectorizedSparseMoE
    from .rope_scaling import build_rope
except ImportError:  # pragma: no cover - script execution path
    from combinatorial_objects import CombinatorialMemory
    from concept_loop import (
        MAX_TOKEN_BYTES, CompositionHead, CompositionState,
        CompositionTokenizer, ConceptLoop, IncrementalCompositionDecoder,
        SpanSidecar,
    )
    from memory_attention import MemoryCrossAttention, V3MemoryPath
    from moe_vectorized import VectorizedSparseMoE
    from rope_scaling import build_rope


@dataclass
class V3Config:
    vocab_size: int = 64000
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
    # The v4 64k vocabulary is the default.  Factorizing the tied embedding
    # and sharing the MTP output geometry keeps the model near the byte-era
    # parameter budget.  Old checkpoints carry their original fields and are
    # migrated to the non-factorized layout by the loaders.
    factorized_embedding: bool = True
    embedding_rank: int = 128
    mtp_tie_heads: bool = True
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
    # Context-aware domain -> branch -> expert routing.  The branch router is
    # conditioned on an attention/routing summary that is carried through all
    # blocks and updated by every MoE block's selected branch/expert state.
    moe_hierarchical: bool = True
    moe_branches: int = 8
    moe_context_router: bool = True
    moe_route_feedback: bool = True
    moe_route_to_attention: bool = True
    # Gate the route->attention residual with the feedback curriculum.  Off
    # keeps the historical always-on context injection.
    gate_route_attention: bool = False
    # Hyperdimensional expert/branch signatures fused into the router.
    # Off by default until the HD codebook is A/B gated against the learned
    # MLP router.
    moe_hyperdimensional: bool = True
    moe_hd_dim: int = 256
    moe_hd_scale: float = 0.5
    moe_associative_memory: bool = True
    moe_memory_keys: int = 16
    moe_memory_dim: int = 0
    # Growable C(V,2)/C(V,3) object memory.  Only vertex key/value
    # hypervectors are stored; pair objects are bound on demand.
    use_combinatorial_memory: bool = False
    use_concept_loop: bool = True
    compute_concept: bool = True
    task_vocab_size: int = 16
    # Scale of the concept/object residual injected into attention and MoE
    # routers.  1.0 preserves the original full-strength loop; a small value
    # lets the concept loop warm up without disturbing the early trunk.
    concept_loop_scale: float = 1.0
    concept_loop_attention: bool = False
    concept_router_tokens: bool = True
    concept_router_global: bool = True
    # Coherent feedback bus.  Every path starts closed (tanh(0)=0), so a new
    # model is numerically identical to the pre-feedback architecture at step
    # zero, then gradients can open each path.  The loop is:
    #   attention summary + hidden summary + concept/object state -> route
    #   route -> next attention input + router context
    #   token concept -> gated residual stream
    #   object state -> next block's object retrieval
    #   memory summary -> MTP context
    #   previous MTP branch -> next future lag
    feedback_token_residual: bool = True
    feedback_route_concept: bool = True
    feedback_object_carry: bool = True
    feedback_memory_mtp: bool = True
    feedback_mtp_carry: bool = True
    # Learn one positive gate per router-context component (route, pooled
    # concept, token-aligned concept).  The gates start at 1.0, so the first
    # step is numerically identical to the fixed sum, then the model can learn
    # how much each resolution contributes.
    learned_router_mix: bool = False
    # Explicit learned embedding for selected C(256,2) object IDs.  This
    # turns the hard pointer readout into a real object-ID vocabulary entry
    # instead of exposing only bound vertex vectors.
    object_id_embedding: bool = False
    object_id_dim: int = 64
    concept_loop_vertices: int = 256
    concept_loop_dim: int = 128
    # Byte models can feed a token-aligned subword-ID embedding into the
    # byte -> subword -> word concept encoder.
    subword_embedding: bool = False
    subword_vocab_size: int = 64060
    concept_top_vertices: int = 8
    concept_top_pairs: int = 4
    concept_top_triples: int = 2
    # Initial pair-object gate probability.  The historical value is
    # sigmoid(-4) = 0.017986, effectively closed for short runs.
    object_gate_init: float = 0.017986
    composition_head: bool = True
    # Auxiliary multi-resolution probes.  They are trained on the shared
    # hidden state and do not feed generation.
    byte_probe: bool = False
    byte_probe_full: bool = False
    word_probe_size: int = 0
    probe_detach: bool = True
    grad_checkpoint: bool = False
    comb_vertices: int = 64
    comb_dim: int = 128
    comb_top_vertices: int = 16
    comb_top_pairs: int = 8
    comb_top_triples: int = 8
    force_manual_attention: bool = False
    attn_gate: bool = True
    # Frontier MTP defaults.  The branch is a real, separately parameterised
    # block per lag, not one shared branch reused for every horizon.
    mtp: int = 3
    compute_mtp: bool = True
    mtp_adapter_rank: int = 32
    # Low-rank MTP output width.  0 reuses the token embedding rank.
    mtp_output_rank: int = 0
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
    # Fixed hyperdimensional time rails: bundle context + <now>, then bind
    # with a fixed random permutation of <now> for each future lag.
    mtp_hd_time: bool = True
    mtp_branch_norm: bool = True
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


def _sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float,
    group: int,
    is_causal: bool = False,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """SDPA with optional native grouped-query attention.

    ``enable_gqa`` avoids materialising the repeated KV heads, which is a 5x
    copy for the 10-query/2-KV layout.  Older PyTorch builds do not accept the
    keyword; fall back to the explicit repeat there.  Masked calls keep the
    repeat because some backends reject GQA together with an arbitrary mask.
    """
    kwargs = {"dropout_p": dropout_p}
    if is_causal:
        kwargs["is_causal"] = True
    if attn_mask is not None:
        kwargs["attn_mask"] = attn_mask
    if group > 1 and attn_mask is None:
        try:
            return F.scaled_dot_product_attention(
                q, k, v, enable_gqa=True, **kwargs
            )
        except TypeError:
            pass
    if group > 1:
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
    return F.scaled_dot_product_attention(q, k, v, **kwargs)


def _attention_context(device: torch.device):
    """Keep CPU SDPA on the deterministic math kernel.

    Some CPU PyTorch builds select a fused attention kernel that is not
    bit-deterministic (and, on Android/Termux, not numerically stable).  CUDA
    keeps the default fused-kernel selection.
    """
    if (
        sdpa_kernel is not None
        and SDPBackend is not None
        and device.type == "cpu"
    ):
        return sdpa_kernel(SDPBackend.MATH)
    return contextlib.nullcontext()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # Match the autocast activation dtype.  Multiplying a bf16 activation
        # by an fp32 weight promotes the whole residual stream back to fp32
        # and prevents the fused RMSNorm kernel from firing on Blackwell.
        weight = self.weight.to(dtype=x.dtype)
        if hasattr(F, "rms_norm"):
            return F.rms_norm(
                x, (self.weight.numel(),), weight, self.eps
            )
        return weight * x * torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True) + self.eps
        )


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
        if self.force_manual or x.device.type == "xla":
            if self.kv_group > 1:
                k = k.repeat_interleave(self.kv_group, dim=1)
                v = v.repeat_interleave(self.kv_group, dim=1)
            y = self._manual_attention(q, k, v)
        else:
            with _attention_context(x.device):
                # With ctx <= window the sliding-window mask is exactly the
                # causal mask.  Passing a 512x512 bool mask here forces SDPA
                # onto a slower masked backend and disables GQA; use the
                # causal fast path instead.
                if self.is_global or t <= self.window:
                    y = _sdpa(
                        q, k, v,
                        dropout_p=self.drop if self.training else 0.0,
                        group=self.kv_group,
                        is_causal=True,
                    )
                else:
                    mask = _causal_mask(
                        t, x.device, is_global=False, window=self.window
                    )
                    y = _sdpa(
                        q, k, v,
                        dropout_p=self.drop if self.training else 0.0,
                        group=self.kv_group,
                        attn_mask=mask,
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

    def forward(self, x, context=None):
        out, aux = self.inner(x, context=context)
        return out, aux["load_balance"], aux["router_z"], aux


def _router_add(base: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
    """Add router context tensors with either global or token alignment."""
    if base is None:
        return value
    if base.ndim == value.ndim:
        return base + value
    if base.ndim == 2 and value.ndim == 3:
        return base[:, None, :] + value
    if base.ndim == 3 and value.ndim == 2:
        return base + value[:, None, :]
    return base + value


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
        self.concept_loop = bool(cfg.use_concept_loop)
        self.loop_scale = float(cfg.concept_loop_scale)
        self.loop_to_attention = bool(cfg.concept_loop_attention)
        self.router_global = bool(cfg.concept_router_global)
        self.learned_router_mix = bool(cfg.learned_router_mix and self.is_moe)
        if self.learned_router_mix:
            # softplus(raw) == 1.0 at initialization.
            self.router_mix = nn.Parameter(
                torch.full((3,), math.log(math.e - 1.0))
            )
        self.loop_to_state = (
            nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            if self.concept_loop else None
        )
        self.feedback_token_residual = bool(
            cfg.feedback_token_residual and self.loop_to_state is not None
        )
        self.token_feedback_gate = (
            nn.Parameter(torch.zeros(()))
            if self.feedback_token_residual else None
        )
        self.route_feedback = bool(cfg.moe_route_feedback)
        self.feedback_route = bool(
            cfg.feedback_route_concept and self.route_feedback
        )
        self.route_hidden_gate = (
            nn.Parameter(torch.zeros(())) if self.feedback_route else None
        )
        self.route_concept_gate = (
            nn.Parameter(torch.zeros(()))
            if self.feedback_route and self.concept_loop else None
        )
        self.route_to_attention = bool(
            cfg.moe_route_to_attention and self.route_feedback
        )
        self.gate_route_attention = bool(cfg.gate_route_attention)
        if self.route_feedback:
            self.route_attn_proj = nn.Linear(
                cfg.d_model, cfg.d_model, bias=False
            )
            self.route_moe_proj = (
                nn.Linear(cfg.d_model, cfg.d_model, bias=False)
                if self.is_moe else None
            )
            self.route_norm = RMSNorm(cfg.d_model)
            if self.route_to_attention:
                self.route_to_state = nn.Linear(
                    cfg.d_model, cfg.d_model, bias=False
                )

    def forward(
        self,
        x,
        route_ctx=None,
        loop_ctx=None,
        token_ctx=None,
        feedback_scale: float | torch.Tensor = 1.0,
    ):
        feedback_scale_t = torch.as_tensor(
            feedback_scale, device=x.device, dtype=x.dtype
        )
        # Token-level multi-resolution state enters the residual stream, not
        # only the router.  The gate starts at tanh(0)=0, so this is a no-op
        # until the model learns to use it.
        if (
            self.feedback_token_residual
            and token_ctx is not None
            and self.loop_to_state is not None
            and self.token_feedback_gate is not None
        ):
            x = x + (
                self.loop_scale
                * feedback_scale_t
                * torch.tanh(self.token_feedback_gate).to(dtype=x.dtype)
                * self.loop_to_state(token_ctx)
            )
        if (
            self.route_to_attention
            and route_ctx is not None
            and hasattr(self, "route_to_state")
        ):
            # Closed loop: the routing/context state modulates the
            # self-attention input, and the resulting attention summary
            # updates the context for the next layer.  The optional
            # curriculum gate keeps the path exactly zero at step zero.
            route_scale = (
                feedback_scale_t if self.gate_route_attention else 1.0
            )
            x = x + route_scale * self.route_to_state(route_ctx).unsqueeze(1)
        if (
            loop_ctx is not None
            and self.loop_to_state is not None
            and self.loop_to_attention
        ):
            # Multi-resolution concept/object state modulates attention.
            x = x + self.loop_scale * self.loop_to_state(loop_ctx).unsqueeze(1)
        attn_out = self.attn(self.n1(x))
        x = x + self.drop(attn_out)
        h = self.n2(x)
        lb = z = None
        aux: dict[str, Any] = {}
        if self.is_moe:
            if self.learned_router_mix:
                weights = F.softplus(self.router_mix).to(dtype=x.dtype)
                router_context = None
                if route_ctx is not None:
                    router_context = _router_add(
                        router_context, weights[0] * route_ctx
                    )
                if loop_ctx is not None and self.router_global:
                    router_context = _router_add(
                        router_context,
                        weights[1] * self.loop_scale * loop_ctx,
                    )
                if token_ctx is not None:
                    router_context = _router_add(
                        router_context,
                        weights[2] * self.loop_scale * token_ctx,
                    )
            else:
                router_context = route_ctx
                if loop_ctx is not None and self.router_global:
                    scaled_loop = self.loop_scale * loop_ctx
                    router_context = (
                        scaled_loop if router_context is None
                        else router_context + scaled_loop
                    )
                if token_ctx is not None:
                    token_part = self.loop_scale * token_ctx
                    if router_context is None:
                        router_context = token_part
                    elif router_context.ndim == 2:
                        router_context = (
                            router_context[:, None, :] + token_part
                        )
                    else:
                        router_context = router_context + token_part
            h, lb, z, aux = self.mlp(h, context=router_context)
            x = x + self.drop(h)
        else:
            x = x + self.mlp(h)
        if self.route_feedback:
            if route_ctx is None:
                route_ctx = attn_out.new_zeros(
                    attn_out.shape[0], attn_out.shape[-1]
                )
            summary = attn_out.mean(dim=1)
            if self.route_hidden_gate is not None:
                summary = summary + (
                    feedback_scale_t
                    * torch.tanh(self.route_hidden_gate).to(dtype=x.dtype)
                    * x.mean(dim=1)
                )
            feedback = self.route_attn_proj(summary)
            if aux is not None and aux.get("route_summary") is not None:
                feedback = feedback + self.route_moe_proj(
                    aux["route_summary"]
                )
            if (
                self.route_concept_gate is not None
                and loop_ctx is not None
            ):
                concept_summary = loop_ctx
                if token_ctx is not None:
                    concept_summary = concept_summary + token_ctx.mean(dim=1)
                feedback = feedback + (
                    feedback_scale_t
                    * torch.tanh(self.route_concept_gate).to(dtype=x.dtype)
                    * concept_summary
                )
            route_ctx = self.route_norm(route_ctx + feedback)
        if self.token_feedback_gate is not None:
            aux["feedback_token_gate"] = torch.tanh(
                self.token_feedback_gate
            )
        if self.route_hidden_gate is not None:
            aux["feedback_route_hidden_gate"] = torch.tanh(
                self.route_hidden_gate
            )
        if self.route_concept_gate is not None:
            aux["feedback_route_concept_gate"] = torch.tanh(
                self.route_concept_gate
            )
        if route_ctx is not None:
            aux["route_state_norm"] = route_ctx.norm(dim=-1).mean()
        return x, lb, z, route_ctx, aux


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
        else:
            with _attention_context(x.device):
                if self.is_global:
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
        hierarchical: bool = False,
        branches: int = 0,
        context_router: bool = False,
        route_to_attention: bool = False,
        hyperdimensional: bool = False,
        hd_dim: int = 256,
        hd_scale: float = 0.5,
        associative_memory: bool = False,
        memory_keys: int = 16,
        memory_dim: int = 0,
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
            hierarchical=hierarchical,
            branches=branches,
            context_router=context_router,
            hyperdimensional=hyperdimensional,
            hd_dim=hd_dim,
            hd_scale=hd_scale,
            associative_memory=associative_memory,
            memory_keys=memory_keys,
            memory_dim=memory_dim,
        )
        self.drop = nn.Dropout(dropout)
        self.route_to_state = (
            nn.Linear(d_model, d_model, bias=False)
            if route_to_attention else None
        )

    def forward(self, x, context=None):
        if self.route_to_state is not None and context is not None:
            projected = self.route_to_state(context)
            if context.ndim == 2:
                projected = projected.unsqueeze(1)
            x = x + projected
        attn_out = self.attn(self.norm1(x))
        x = x + self.drop(attn_out)
        if context is None:
            context = attn_out.mean(dim=1)
        y, aux = self.moe(self.norm2(x), context=context)
        x = x + self.drop(y)
        return x, aux


class V3LM(nn.Module):
    def __init__(self, cfg: V3Config):
        super().__init__()
        self.cfg = cfg
        self.grad_checkpoint = bool(cfg.grad_checkpoint)
        self.factorized_embedding = bool(cfg.factorized_embedding)
        if self.factorized_embedding:
            self.embedding_rank = int(cfg.embedding_rank)
            if self.embedding_rank < 1:
                raise ValueError("embedding_rank must be positive")
            self.tok = nn.Embedding(cfg.vocab_size, self.embedding_rank)
            self.tok_up = nn.Linear(
                self.embedding_rank, cfg.d_model, bias=False
            )
        else:
            self.embedding_rank = cfg.d_model
            self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.tok_up = None
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.route_init = nn.Parameter(torch.zeros(cfg.d_model))
        # 0 = every feedback gate is closed; 1 = full learned gate.  The
        # trainer advances this during warmup, and evaluation sets it to 1.
        self.register_buffer(
            "feedback_progress", torch.zeros((), dtype=torch.float32)
        )
        self.task_embedding = (
            nn.Embedding(cfg.task_vocab_size, cfg.d_model)
            if cfg.task_vocab_size > 0 else None
        )
        self.comb_memory = (
            CombinatorialMemory(
                d_model=cfg.d_model,
                vertices=cfg.comb_vertices,
                dim=cfg.comb_dim,
                top_vertices=cfg.comb_top_vertices,
                top_pairs=cfg.comb_top_pairs,
                top_triples=cfg.comb_top_triples,
            )
            if cfg.use_combinatorial_memory else None
        )
        self.concept_loop = (
            ConceptLoop(
                d_model=cfg.d_model,
                vertices=cfg.concept_loop_vertices,
                dim=cfg.concept_loop_dim,
                top_vertices=cfg.concept_top_vertices,
                top_pairs=cfg.concept_top_pairs,
                top_triples=cfg.concept_top_triples,
                object_id_dim=(
                    cfg.object_id_dim if cfg.object_id_embedding else 0
                ),
                subword_vocab_size=(
                    cfg.subword_vocab_size if cfg.subword_embedding else 0
                ),
                depth_carry=bool(cfg.feedback_object_carry),
                object_gate_init=float(cfg.object_gate_init),
            )
            if cfg.use_concept_loop else None
        )
        self.composition_head = (
            CompositionHead(cfg.d_model)
            if cfg.composition_head else None
        )
        self.byte_probe = (
            nn.Linear(cfg.d_model, 256, bias=False)
            if cfg.byte_probe else None
        )
        self.byte_sequence_probe = (
            nn.Linear(
                cfg.d_model, MAX_TOKEN_BYTES * 256, bias=False
            )
            if cfg.byte_probe and cfg.byte_probe_full else None
        )
        self.word_probe = (
            nn.Linear(cfg.d_model, cfg.word_probe_size, bias=False)
            if cfg.word_probe_size > 0 else None
        )
        # Tied main head.  MTP heads are either separate full-vocab
        # projections or, for 64k vocabularies, tied to the main token
        # geometry through project_logits().
        self.mtp_tie_heads = bool(cfg.mtp_tie_heads)
        n_mtp = max(0, cfg.mtp - 1)
        self.mtp_output_rank = int(
            cfg.mtp_output_rank or self.embedding_rank
        )
        if self.mtp_tie_heads and self.factorized_embedding:
            if not 1 <= self.mtp_output_rank <= self.embedding_rank:
                raise ValueError(
                    "mtp_output_rank must be between 1 and embedding_rank"
                )
            # The branch gets a tiny lag-specific projection into the shared
            # token rank.  This keeps v4 output tying without routing full
            # branch gradients through the input embedding projection.
            mtp_head_dim = self.mtp_output_rank
        elif self.mtp_tie_heads:
            mtp_head_dim = 0
        else:
            mtp_head_dim = cfg.vocab_size
        self.mtp_heads = nn.ModuleList(
            [nn.Linear(cfg.d_model, mtp_head_dim, bias=False)
             for _ in range(max(0, cfg.mtp - 1))]
            if mtp_head_dim > 0 else []
        )
        # A separate low-rank output basis for the MTP branch.  It starts as
        # a copy of the token geometry but has its own gradient path, so an
        # unstable future branch cannot corrupt the input/output embedding.
        self.mtp_output = (
            nn.Parameter(
                torch.empty(cfg.vocab_size, self.mtp_output_rank)
            )
            if (
                n_mtp > 0
                and self.mtp_tie_heads
                and self.factorized_embedding
            ) else None
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
                    hierarchical=cfg.moe_hierarchical,
                    # MTP blocks have fewer experts than the trunk.  Pick a
                    # layout with at least 2 experts per branch so top-2 is a
                    # real within-branch choice instead of selecting the whole
                    # branch (the old branches=0 auto-layout produced 4 x 2).
                    branches=max(
                        1,
                        int(cfg.mtp_block_experts)
                        // max(2, int(cfg.topk) * 2),
                    ),
                    context_router=cfg.moe_context_router,
                    route_to_attention=cfg.moe_route_to_attention,
                    hyperdimensional=cfg.moe_hyperdimensional,
                    hd_dim=cfg.moe_hd_dim,
                    hd_scale=cfg.moe_hd_scale,
                    associative_memory=cfg.moe_associative_memory,
                    memory_keys=cfg.moe_memory_keys,
                    memory_dim=cfg.moe_memory_dim,
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
        self.mtp_hd_time = bool(cfg.mtp_hd_time and n_mtp)
        if self.mtp_hd_time:
            now = torch.randn(cfg.d_model)
            self.register_buffer("hd_now", F.normalize(now, dim=-1))
            self.register_buffer(
                "hd_pos",
                F.normalize(
                    torch.randn(max(cfg.max_ctx, cfg.ctx), cfg.d_model),
                    dim=-1,
                ),
            )
            self.register_buffer(
                "hd_perm", torch.randperm(cfg.d_model)
            )
            self.mtp_hd_proj = nn.ModuleList(
                [nn.Linear(cfg.d_model, cfg.d_model, bias=False)
                 for _ in range(n_mtp)]
            )
        else:
            self.hd_now = None
            self.hd_pos = None
            self.hd_perm = None
            self.mtp_hd_proj = nn.ModuleList()
        self.mtp_next_norm = RMSNorm(cfg.d_model) if cfg.mtp_block else None
        self.mtp_branch_norm = (
            RMSNorm(cfg.d_model)
            if cfg.mtp_block and cfg.mtp_branch_norm else None
        )
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
        self.memory_feedback_gate = (
            nn.Parameter(torch.zeros(()))
            if cfg.feedback_memory_mtp
            and (self.memory_path is not None or self.memory_cross is not None)
            else None
        )
        self.mtp_carry_gate = (
            nn.Parameter(torch.zeros(()))
            if cfg.feedback_mtp_carry and n_mtp
            else None
        )
        self.apply(self._init)
        nn.init.normal_(self.route_init, mean=0.0, std=0.02)
        nn.init.normal_(self.tok.weight, mean=0.0, std=0.02)
        if self.task_embedding is not None:
            # A task/domain code must start as a no-op; the embedding then
            # learns only the domain signal that actually helps routing.
            nn.init.zeros_(self.task_embedding.weight)
        if self.tok_up is not None:
            # tok @ tok_up must have variance near the full-rank 0.02
            # embedding initialisation.
            nn.init.normal_(
                self.tok_up.weight,
                mean=0.0,
                std=1.0 / math.sqrt(self.embedding_rank),
            )
            if self.mtp_tie_heads:
                with torch.no_grad():
                    for head in self.mtp_heads:
                        head.weight.copy_(
                            self.tok_up.weight.T[: self.mtp_output_rank]
                        )
                    if self.mtp_output is not None:
                        self.mtp_output.copy_(
                            self.tok.weight[:, : self.mtp_output_rank]
                        )
        # The global apply() initializes every Linear, including the router
        # context and inter-block feedback projections.  Keep those neutral
        # at step zero so the hierarchy itself is exercised first.
        for module in self.modules():
            if getattr(module, "expert_context", None) is not None:
                with torch.no_grad():
                    module.expert_context.weight.mul_(0.01)
            if getattr(module, "branch_context", None) is not None:
                with torch.no_grad():
                    module.branch_context.weight.mul_(0.01)
        for block in self.blocks:
            if getattr(block, "route_feedback", False):
                with torch.no_grad():
                    # Small non-zero feedback breaks the zero-gradient
                    # symmetry while keeping the initial routing conditional
                    # close to the uncontextualized router.
                    block.route_attn_proj.weight.mul_(0.01)
                    if block.route_moe_proj is not None:
                        block.route_moe_proj.weight.mul_(0.01)
                    if getattr(block, "route_to_state", None) is not None:
                        block.route_to_state.weight.mul_(0.01)
        for block in self.mtp_blocks:
            if getattr(block, "route_to_state", None) is not None:
                with torch.no_grad():
                    block.route_to_state.weight.mul_(0.01)
        for proj in self.mtp_hd_proj:
            with torch.no_grad():
                proj.weight.zero_()
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

    @property
    def n_mtp_heads(self) -> int:
        """Number of future lags, independent of whether heads are tied."""
        return max(0, int(self.cfg.mtp) - 1)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.tok(tokens)
        return self.tok_up(x) if self.tok_up is not None else x

    def embed_distribution(self, probs: torch.Tensor) -> torch.Tensor:
        """Map a vocabulary distribution back into d_model space."""
        rank = probs @ self.tok.weight
        return self.tok_up(rank) if self.tok_up is not None else rank

    def embed_mixture(
        self,
        probs: torch.Tensor,
        target_ids: torch.Tensor,
        mix: torch.Tensor | float,
    ) -> torch.Tensor:
        """Embed ``(1-mix)*probs + mix*one_hot(target)`` without a dense one-hot.

        The MTP soft teacher only needs the embedding of the mixed next-token
        distribution.  Materializing the ``[B,T,V]`` one-hot costs gigabytes at
        64k vocabulary and forced the Blackwell smoke into a much smaller batch.
        """
        rank = probs @ self.tok.weight
        target = self.tok(
            target_ids.long().clamp_min(0).clamp_max(self.cfg.vocab_size - 1)
        ).to(dtype=rank.dtype)
        rank = (1.0 - mix) * rank + mix * target
        return self.tok_up(rank) if self.tok_up is not None else rank

    def project_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits through the token basis."""
        if self.tok_up is not None:
            hidden = hidden @ self.tok_up.weight
        return hidden @ self.tok.weight.T

    def _hd_future_signal(
        self, hidden: torch.Tensor, positions: torch.Tensor, steps: int
    ) -> torch.Tensor | None:
        if not self.mtp_hd_time:
            return None
        assert self.hd_now is not None
        assert self.hd_pos is not None
        assert self.hd_perm is not None
        future_now = self.hd_now
        for _ in range(max(1, int(steps))):
            future_now = future_now[self.hd_perm]
        # Bundle context with position and the fixed <now> anchor; bind the
        # result with the permuted <now> rail for this future lag.
        context = (
            F.normalize(hidden, dim=-1)
            + self.hd_pos[positions]
            + self.hd_now
        )
        return context * future_now

    def mtp_head_logits(
        self, head: nn.Module | None, hidden: torch.Tensor
    ) -> torch.Tensor:
        """Apply an MTP output adapter over the shared token basis."""
        if head is None:
            return self.project_logits(hidden)
        adapted = head(hidden)
        if (
            self.mtp_output is not None
            and adapted.shape[-1] == self.mtp_output_rank
        ):
            adapted = adapted @ self.mtp_output.T
        return adapted

    def build_memory(self, tokens, max_tokens: int | None = None) -> "V3LM":
        """Populate the hashed n-gram store from token IDs (no-op if disabled)."""
        if self.memory_path is not None:
            self.memory_path.build_memory(tokens, max_tokens=max_tokens)
        return self

    @torch.no_grad()
    def set_feedback_progress(self, value: float) -> "V3LM":
        """Scale every learned feedback gate by ``value`` in [0, 1].

        This is a curriculum knob, not an architecture change.  At 0 the model
        is the pre-feedback architecture; at 1 every gate uses its learned
        ``tanh(parameter)`` strength.
        """
        self.feedback_progress.fill_(
            max(0.0, min(1.0, float(value)))
        )
        return self

    def feedback_gate_scale(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        scale = self.feedback_progress
        return scale.to(dtype=dtype) if dtype is not None else scale

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, tokens, mtp_teacher: bool = False,
                mtp_teacher_prob: float | None = None,
                mtp_next_tokens=None,
                mtp_gt_mix: float | None = None,
                return_diagnostics: bool = False,
                sidecar=None,
                task_ids: torch.Tensor | None = None,
                compute_mtp_aux: bool = True,
                compute_byte_probe: bool = True,
                compute_byte_sequence: bool = True,
                compute_concept: bool | None = None,
                compute_mtp: bool | None = None,
                return_hidden: bool = False):
        if compute_concept is None:
            compute_concept = bool(self.cfg.compute_concept)
        if compute_mtp is None:
            compute_mtp = bool(self.cfg.compute_mtp)
        if return_hidden and compute_mtp:
            raise ValueError(
                "return_hidden is only supported with compute_mtp=False"
            )
        x = self.embed(tokens)
        feedback_scale = self.feedback_gate_scale(dtype=x.dtype)
        lbs, zs = [], []
        branch_lbs: list[torch.Tensor] = []
        hd_diversities: list[torch.Tensor] = []
        memory_gates: list[torch.Tensor] = []
        aux_hidden: dict[int, torch.Tensor] = {}
        route_ctx = self.route_init.to(dtype=x.dtype).unsqueeze(0).expand(
            x.shape[0], -1
        )
        task_ctx = None
        if task_ids is not None and self.task_embedding is not None:
            embedded = self.task_embedding(
                task_ids.to(device=tokens.device, dtype=torch.long)
            )
            if embedded.ndim == 1:
                task_ctx = embedded.unsqueeze(0)
            elif embedded.ndim == 2:
                task_ctx = embedded
            else:
                task_ctx = embedded.mean(dim=1)
            task_ctx = task_ctx.to(dtype=x.dtype)
        concept_carry = None
        object_carry = None
        object_stats: list[dict[str, torch.Tensor]] = []
        object_balances: list[torch.Tensor] = []
        feedback_stats: list[dict[str, torch.Tensor]] = []
        for layer_index, block in enumerate(self.blocks):
            concept_route = (
                route_ctx + task_ctx
                if task_ctx is not None else route_ctx
            )
            loop_ctx = None
            token_ctx = None
            if self.concept_loop is not None and compute_concept:
                concept, object_ctx, loop_ctx = self.concept_loop(
                    x,
                    sidecar,
                    concept_route,
                    concept_carry,
                    object_carry,
                    feedback_scale,
                    compute_histograms=bool(return_diagnostics),
                )
                concept_carry = self.concept_loop.last_carry
                object_carry = object_ctx
                stats = getattr(self.concept_loop, "last_object_stats", None)
                if stats:
                    object_stats.append(stats)
                balance = getattr(
                    self.concept_loop, "last_object_balance", None
                )
                if balance is not None:
                    object_balances.append(balance)
                if self.cfg.concept_router_tokens:
                    token_ctx = self.concept_loop.router_token_state(
                        concept, object_ctx
                    )
            block_route = route_ctx
            if task_ctx is not None:
                if self.concept_loop is not None:
                    # Task/domain conditioning is persistent in the concept
                    # loop's router/attention state; it cannot wash out when
                    # route feedback overwrites the initial route vector.
                    loop_ctx = (
                        task_ctx if loop_ctx is None
                        else loop_ctx + task_ctx
                    )
                elif layer_index == 0 or not getattr(
                    block, "route_feedback", False
                ):
                    # With no concept loop, add the task to the first route
                    # once; route feedback should preserve it thereafter.
                    block_route = route_ctx + task_ctx
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                x, lb, z, route_ctx, block_aux = checkpoint(
                    block, x, block_route, loop_ctx, token_ctx,
                    feedback_scale,
                    use_reentrant=False,
                )
            else:
                x, lb, z, route_ctx, block_aux = block(
                    x, block_route, loop_ctx, token_ctx, feedback_scale
                )
            if layer_index in self.mtp_aux_layers:
                aux_hidden[layer_index] = x
            if lb is not None:
                lbs.append(lb)
                zs.append(z)
            if block_aux:
                if block_aux.get("branch_load_balance") is not None:
                    branch_lbs.append(block_aux["branch_load_balance"])
                if block_aux.get("hd_code_diversity") is not None:
                    hd_diversities.append(block_aux["hd_code_diversity"])
                if block_aux.get("memory_gate_mean") is not None:
                    memory_gates.append(block_aux["memory_gate_mean"])
                feedback_entry = {
                    key: block_aux[key]
                    for key in (
                        "feedback_token_gate",
                        "feedback_route_hidden_gate",
                        "feedback_route_concept_gate",
                        "route_state_norm",
                    )
                    if block_aux.get(key) is not None
                }
                if feedback_entry:
                    feedback_stats.append(feedback_entry)
        if self.comb_memory is not None:
            x = x + self.comb_memory(x)
        mem_aux = {}
        if self.memory_path is not None:
            x, mem_aux = self.memory_path.cross(x)
        elif self.memory_cross is not None:
            x, mem_aux = self.memory_cross(x)
        memory_summary = (
            mem_aux.get("memory_summary")
            if isinstance(mem_aux, dict) else None
        )
        if memory_summary is not None and self.memory_feedback_gate is not None:
            # Global memory state re-enters the residual stream.  The gate is
            # closed at init, so this is a learned feedback path, not a
            # second unconditional residual.
            x = x + (
                feedback_scale
                * torch.tanh(self.memory_feedback_gate).to(dtype=x.dtype)
                * memory_summary.unsqueeze(1)
            )
        x = self.norm(x)
        if return_hidden:
            # Training can project this hidden state through the 64k head in
            # bounded chunks, so the (B, T, vocab) logits tensor is never
            # materialised in full.  The hashed n-gram bias is token-local;
            # reject the fast path rather than silently dropping it.
            if self.memory_path is not None:
                raise ValueError(
                    "return_hidden does not support the n-gram logit bias"
                )
            main = None
        else:
            main = self.project_logits(x)
            if self.memory_path is not None:
                # Sparse n-gram bias with a learned global scale.
                bias = self.memory_path.logit_bias(tokens)
                bias.value = bias.value * self.memory_scale
                bias.add_(main)

        # Deep supervision: each selected intermediate stream predicts t+2
        # through the same small MTP head.  The residual stream is causal, so
        # no future token can leak into an earlier position.
        mtp_aux_logits: list[torch.Tensor] = []
        if self.mtp_aux_norms and compute_mtp_aux and compute_mtp:
            aux_head = self.mtp_heads[0] if self.mtp_heads else None
            for layer, norm in zip(self.mtp_aux_layers, self.mtp_aux_norms):
                hidden = aux_hidden.get(layer)
                if hidden is not None:
                    projected = norm(hidden)
                    mtp_aux_logits.append(
                        self.mtp_head_logits(aux_head, projected)
                    )

        mtp_block_lb: list[torch.Tensor] = []
        mtp_block_rz: list[torch.Tensor] = []
        mtp_composition_logits: list[torch.Tensor] = []
        mtp_object_pointer_logits: list[torch.Tensor] = []
        mtp_object_triple_pointer_logits: list[torch.Tensor] = []
        gate_loss = None
        mtp: list[torch.Tensor] = []
        if not compute_mtp:
            pass
        elif self.mtp_blocks:
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
            explicit_teacher = mtp_next_tokens is not None
            if explicit_teacher:
                target_ids = mtp_next_tokens
                mix = 1.0 if mtp_gt_mix is None else gt_mix
                mix = min(0.95, max(0.0, mix))
            elif mtp_teacher and gt_mix > 0.0:
                target_ids = next_ids
                mix = gt_mix
                if self.training and teacher_prob < 1.0:
                    keep = (
                        torch.rand(tokens.size(0), 1, 1, device=tokens.device)
                        < teacher_prob
                    ).to(dtype=pred_probs.dtype)
                    mix = mix * keep
            else:
                target_ids = next_ids
                mix = 0.0
            next_emb = self.mtp_next_norm(
                self.embed_mixture(pred_probs, target_ids, mix)
            )

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
            positions = torch.arange(tokens.size(1), device=tokens.device)
            position_anchor = (
                self.mtp_position_embed(positions).to(dtype=x.dtype).unsqueeze(0)
                if self.mtp_position_embed is not None else 0.0
            )
            mtp = []
            gate_losses = []
            mtp_diagnostics: list[dict[str, torch.Tensor]] = []
            mtp_carry: torch.Tensor | None = None
            for idx in range(self.n_mtp_heads):
                head = self.mtp_heads[idx] if self.mtp_heads else None
                cond = (
                    self.mtp_cond_proj[idx](
                        torch.cat([x, next_emb], dim=-1)
                    )
                    + self.mtp_now[idx]
                    + position_anchor
                )
                if self.mtp_hd_time and idx < len(self.mtp_hd_proj):
                    hd_signal = self._hd_future_signal(
                        x, positions, idx + 2
                    )
                    if hd_signal is not None:
                        cond = cond + 0.1 * self.mtp_hd_proj[idx](hd_signal)
                mtp_context = route_ctx
                if self.concept_loop is not None and loop_ctx is not None:
                    scaled_loop = (
                        float(self.cfg.concept_loop_scale) * loop_ctx
                    )
                    mtp_context = (
                        mtp_context + scaled_loop
                        if mtp_context is not None else scaled_loop
                    )
                if memory_summary is not None and self.memory_feedback_gate is not None:
                    mtp_context = _router_add(
                        mtp_context,
                        feedback_scale
                        * torch.tanh(self.memory_feedback_gate).to(dtype=x.dtype)
                        * memory_summary,
                    )
                if mtp_carry is not None and self.mtp_carry_gate is not None:
                    mtp_context = _router_add(
                        mtp_context,
                        feedback_scale
                        * torch.tanh(self.mtp_carry_gate).to(dtype=x.dtype)
                        * mtp_carry,
                    )

                if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                    branch, block_aux = checkpoint(
                        self.mtp_blocks[idx],
                        cond,
                        mtp_context,
                        use_reentrant=False,
                    )
                else:
                    branch, block_aux = self.mtp_blocks[idx](
                        cond, context=mtp_context
                    )
                if self.mtp_branch_norm is not None:
                    branch = self.mtp_branch_norm(branch)
                if self.mtp_carry_gate is not None:
                    # Next future lag sees the previous branch's summary.
                    mtp_carry = branch.mean(dim=1)
                if self.composition_head is not None:
                    # The future branch predicts future word/sentence/math
                    # composition too, not only the next subword.  Sharing
                    # the main structural head avoids a second vocabulary.
                    mtp_composition_logits.append(
                        self.composition_head(branch)
                    )
                if self.concept_loop is not None:
                    # Future object identity is a shared codebook query, not
                    # another dense object vocabulary.
                    mtp_object_pointer_logits.append(
                        self.concept_loop.object_pointer(branch).view(
                            branch.shape[0], branch.shape[1], 2,
                            self.concept_loop.vertices,
                        )
                    )
                    mtp_object_triple_pointer_logits.append(
                        self.concept_loop.object_triple_pointer(branch).view(
                            branch.shape[0], branch.shape[1], 3,
                            self.concept_loop.vertices,
                        )
                    )
                if block_aux:
                    if block_aux.get("load_balance") is not None:
                        mtp_block_lb.append(block_aux["load_balance"])
                    if block_aux.get("router_z") is not None:
                        mtp_block_rz.append(block_aux["router_z"])
                if head is not None:
                    branch_logits = self.mtp_head_logits(head, branch)
                    base_logits = (
                        main if self.mtp_tie_heads
                        else self.mtp_head_logits(head, x)
                    )
                else:
                    branch_logits = self.project_logits(branch)
                    # Tied fallback is exactly the main prediction.
                    base_logits = main
                if self.mtp_gate:
                    gate_logits = self.mtp_gate[idx](gate_features)
                    gate = torch.sigmoid(gate_logits)
                    mtp.append(
                        gate * branch_logits + (1.0 - gate) * base_logits
                    )
                    if return_diagnostics:
                        mtp_diagnostics.append(
                            {
                                "branch": branch_logits.detach(),
                                "base": base_logits.detach(),
                                "gate": gate.detach(),
                            }
                        )
                    # Supervise the gate on the actual t+2 utility of the
                    # branch, not on t+1 correctness.  A correct t+1 token
                    # does not imply that the future branch is better; the
                    # old target was the reason correct-t1 positions could
                    # have worse MTP loss than wrong-t1 positions.
                    lag = idx + 2
                    target_len = tokens.size(1) - lag
                    if target_len > 0:
                        targets = tokens[:, lag:]
                        mask = targets >= 0
                        # Exact branch-vs-fallback CE advantage, chunked so
                        # 2k contexts do not create two more full-vocab
                        # softmax tensors.  The target is detached: it
                        # supervises the gate only.
                        advantage = torch.empty(
                            tokens.size(0),
                            target_len,
                            device=tokens.device,
                            dtype=pred_probs.dtype,
                        )
                        with torch.no_grad():
                            for start in range(0, target_len, 64):
                                end = min(target_len, start + 64)
                                tgt = targets[:, start:end]
                                b = branch_logits[:, start:end].reshape(
                                    -1, self.cfg.vocab_size
                                )
                                z = base_logits[:, start:end].reshape(
                                    -1, self.cfg.vocab_size
                                )
                                t = tgt.reshape(-1)
                                bce = F.cross_entropy(
                                    b, t, reduction="none",
                                    ignore_index=-100,
                                ).view(tokens.size(0), end - start)
                                zce = F.cross_entropy(
                                    z, t, reduction="none",
                                    ignore_index=-100,
                                ).view(tokens.size(0), end - start)
                                advantage[:, start:end] = zce - bce
                        target_prob = torch.sigmoid(advantage.detach() / 0.1)
                        gate_loss = F.binary_cross_entropy_with_logits(
                            gate_logits[:, :target_len].squeeze(-1),
                            target_prob,
                            reduction="none",
                        )
                        if mask.any():
                            gate_loss = gate_loss[mask].mean()
                        else:
                            gate_loss = gate_loss.sum() * 0.0
                        gate_losses.append(gate_loss)
                else:
                    mtp.append(branch_logits)
            gate_loss = (
                torch.stack(gate_losses).mean() if gate_losses else None
            )
        else:
            mtp = []
            for idx, adapter in enumerate(self.mtp_adapters):
                hidden = adapter(x)
                if self.composition_head is not None:
                    mtp_composition_logits.append(
                        self.composition_head(hidden)
                    )
                if self.concept_loop is not None:
                    mtp_object_pointer_logits.append(
                        self.concept_loop.object_pointer(hidden).view(
                            hidden.shape[0], hidden.shape[1], 2,
                            self.concept_loop.vertices,
                        )
                    )
                    mtp_object_triple_pointer_logits.append(
                        self.concept_loop.object_triple_pointer(hidden).view(
                            hidden.shape[0], hidden.shape[1], 3,
                            self.concept_loop.vertices,
                        )
                    )
                head = self.mtp_heads[idx] if self.mtp_heads else None
                mtp.append(self.mtp_head_logits(head, hidden))
            gate_loss = None
        aux = dict(mem_aux)
        if object_stats:
            keys = object_stats[0].keys()
            object_memory = {
                key: torch.stack([item[key] for item in object_stats]).mean()
                for key in keys
            }
            aux["object_memory"] = object_memory
            if object_balances:
                aux["object_balance"] = torch.stack(object_balances).mean()
            else:  # pragma: no cover - concept loop always exposes it
                aux["object_balance"] = object_memory["object_balance"]
        if feedback_stats:
            feedback_keys = sorted(
                {key for entry in feedback_stats for key in entry}
            )
            feedback = {
                key: torch.stack(
                    [entry[key] for entry in feedback_stats if key in entry]
                ).mean()
                for key in feedback_keys
            }
            if memory_summary is not None:
                feedback["memory_summary_norm"] = memory_summary.norm(
                    dim=-1
                ).mean()
            if self.memory_feedback_gate is not None:
                feedback["memory_feedback_gate"] = torch.tanh(
                    self.memory_feedback_gate
                )
            if self.mtp_carry_gate is not None:
                feedback["mtp_carry_gate"] = torch.tanh(
                    self.mtp_carry_gate
                )
            if object_stats:
                feedback["object_depth_carry_gate"] = object_memory[
                    "depth_carry_gate"
                ]
            feedback["progress"] = self.feedback_progress
            aux["feedback"] = feedback
        if mtp_aux_logits:
            aux["mtp_aux_logits"] = mtp_aux_logits
            aux["mtp_aux_layers"] = self.mtp_aux_layers
            aux["mtp_aux_layer_weights"] = self.mtp_aux_layer_weights
        if compute_mtp and self.mtp_blocks and gate_loss is not None:
            aux["mtp_gate_loss"] = gate_loss
        if mtp_block_lb:
            aux["mtp_load_balance"] = torch.stack(mtp_block_lb).mean()
        if mtp_block_rz:
            aux["mtp_router_z"] = torch.stack(mtp_block_rz).mean()
        if lbs:
            aux["load_balance"] = torch.stack(lbs).mean()
            aux["router_z"] = torch.stack(zs).mean()
        if branch_lbs:
            aux["branch_load_balance"] = torch.stack(branch_lbs).mean()
        if hd_diversities:
            aux["hd_code_diversity"] = torch.stack(hd_diversities).mean()
        if memory_gates:
            aux["memory_gate_mean"] = torch.stack(memory_gates).mean()
        if self.composition_head is not None:
            aux["composition_logits"] = self.composition_head(x)
            if mtp_composition_logits:
                aux["mtp_composition_logits"] = mtp_composition_logits
                aux["mtp_composition_lags"] = [
                    2 + idx for idx in range(len(mtp_composition_logits))
                ]
        probe_hidden = x.detach() if self.cfg.probe_detach else x
        if self.byte_probe is not None and compute_byte_probe:
            aux["byte_probe_logits"] = self.byte_probe(probe_hidden)
        if self.byte_sequence_probe is not None and compute_byte_sequence:
            seq = self.byte_sequence_probe(probe_hidden)
            aux["byte_sequence_logits"] = seq.view(
                *probe_hidden.shape[:2], MAX_TOKEN_BYTES, 256
            )
        if self.word_probe is not None:
            aux["word_probe_logits"] = self.word_probe(probe_hidden)
        if mtp_object_pointer_logits:
            aux["mtp_object_pointer_logits"] = mtp_object_pointer_logits
            aux["mtp_object_pointer_lags"] = [
                2 + idx for idx in range(len(mtp_object_pointer_logits))
            ]
        if mtp_object_triple_pointer_logits:
            aux["mtp_object_triple_pointer_logits"] = (
                mtp_object_triple_pointer_logits
            )
            aux["mtp_object_triple_pointer_lags"] = [
                2 + idx
                for idx in range(len(mtp_object_triple_pointer_logits))
            ]
        if self.concept_loop is not None and hasattr(
            self.concept_loop, "last_object_ids"
        ):
            aux["object_pair_ids"] = self.concept_loop.last_object_ids
            if getattr(self.concept_loop, "last_triple_ids", None) is not None:
                aux["object_triple_ids"] = self.concept_loop.last_triple_ids
            if getattr(self.concept_loop, "last_pointer_logits", None) is not None:
                aux["object_pointer_logits"] = (
                    self.concept_loop.last_pointer_logits
                )
            if getattr(
                self.concept_loop, "last_triple_pointer_logits", None
            ) is not None:
                aux["object_triple_pointer_logits"] = (
                    self.concept_loop.last_triple_pointer_logits
                )
        if return_diagnostics and "mtp_diagnostics" in locals():
            aux["mtp_diagnostics"] = mtp_diagnostics
        return (x if return_hidden else main), mtp, aux

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int = 16,
        temperature: float = 1.0,
        seed: int | None = None,
        tokenizer: Any = None,
        task_id: int | None = None,
        composition_thresholds: Sequence[float] | None = None,
        speculative: bool = False,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        repetition_window: int = 256,
    ):
        """Autoregressive loop with model and tokenizer composition events."""
        if speculative:
            return self.generate_speculative(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=seed,
                task_id=task_id,
            )
        if seed is not None:
            torch.manual_seed(seed)
        ids = tokens
        events: list[list[int]] = []
        layer_tokenizer = (
            getattr(tokenizer, "layered", tokenizer)
            if tokenizer is not None else None
        )
        decoder = (
            IncrementalCompositionDecoder(layer_tokenizer.subword)
            if layer_tokenizer is not None
            and hasattr(layer_tokenizer, "subword") else None
        )
        tokenizer_events: list[tuple[int, str]] = []
        object_events: list[list[int]] = []
        if decoder is not None:
            for token_id in tokens[0].detach().cpu().tolist():
                tokenizer_events.extend(decoder.feed(int(token_id)))
        max_ctx = max(1, int(self.cfg.max_ctx))

        def build_sidecar(window_ids: torch.Tensor):
            if tokenizer is None:
                return None
            try:
                return SpanSidecar.from_token_ids(
                    window_ids[0].detach().cpu().tolist(), layer_tokenizer
                )
            except Exception:
                return None

        task_ids = (
            torch.full(
                (ids.size(0),), int(task_id), device=ids.device,
                dtype=torch.long,
            )
            if task_id is not None else None
        )
        sidecar = build_sidecar(ids[:, -max_ctx:])
        structure = CompositionState()
        if sidecar is not None:
            structure.observe_multi(sidecar.event_multi[:-1])
        for _ in range(max_new_tokens):
            window = ids[:, -max_ctx:]
            logits, _mtp, aux = self(
                window, sidecar=sidecar, task_ids=task_ids
            )
            next_logits = logits[:, -1].float()
            if repetition_penalty and repetition_penalty != 1.0:
                next_logits = next_logits.clone()
                recent = ids[0, -max(1, int(repetition_window)):]
                if recent.numel():
                    unique = torch.unique(recent)
                    selected = next_logits[0, unique]
                    next_logits[0, unique] = torch.where(
                        selected > 0,
                        selected / float(repetition_penalty),
                        selected * float(repetition_penalty),
                    )
            if temperature > 0:
                next_logits = next_logits / max(1e-3, temperature)
            if top_k and int(top_k) > 0:
                k = min(int(top_k), next_logits.size(-1))
                values, indices = torch.topk(next_logits, k, dim=-1)
                filtered = torch.full_like(next_logits, float("-inf"))
                filtered.scatter_(1, indices, values)
                next_logits = filtered
            if top_p is not None and 0.0 < float(top_p) < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_logits, descending=True, dim=-1
                )
                probs = torch.softmax(sorted_logits, dim=-1)
                cumulative = torch.cumsum(probs, dim=-1)
                mask = cumulative - probs > float(top_p)
                sorted_logits = sorted_logits.masked_fill(
                    mask, float("-inf")
                )
                next_logits = torch.full_like(
                    next_logits, float("-inf")
                ).scatter_(1, sorted_indices, sorted_logits)
            if temperature <= 0:
                next_id = next_logits.argmax(dim=-1)
            else:
                probs = torch.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
            ids = torch.cat([ids, next_id[:, None]], dim=1)
            comp = aux.get("composition_logits")
            fired = None
            if comp is not None:
                probs = torch.sigmoid(comp[:, -1])
                if composition_thresholds is None:
                    fired = probs > 0.5
                else:
                    thresholds = torch.as_tensor(
                        composition_thresholds,
                        device=probs.device, dtype=probs.dtype,
                    ).reshape(1, -1)
                    if thresholds.numel() != probs.shape[-1]:
                        raise ValueError(
                            "composition_thresholds must have one value "
                            "per composition event"
                        )
                    fired = probs > thresholds
                fired_list = fired[0].tolist()
                if sidecar is not None:
                    fired_list = structure.filter(fired_list)
                    fired = torch.tensor(
                        [fired_list], device=fired.device,
                        dtype=torch.bool,
                    )
                events.append(
                    [
                        idx for idx, value in enumerate(fired_list)
                        if value
                    ]
                )
            if decoder is not None:
                tokenizer_events.extend(decoder.feed(int(next_id[0])))
            if aux.get("object_pair_ids") is not None:
                event = {
                    "pairs": aux["object_pair_ids"][:, -1].reshape(-1).tolist(),
                }
                if aux.get("object_triple_ids") is not None:
                    event["triples"] = (
                        aux["object_triple_ids"][:, -1].reshape(-1).tolist()
                    )
                if aux.get("object_pointer_logits") is not None:
                    pointer = aux["object_pointer_logits"][:, -1]
                    pu = pointer[:, 0].argmax(dim=-1)
                    pv_logits = pointer[:, 1].clone()
                    rows = torch.arange(
                        pv_logits.shape[0], device=pv_logits.device
                    )
                    pv_logits[rows, pu] = torch.finfo(pv_logits.dtype).min
                    pv = pv_logits.argmax(dim=-1)
                    lo = torch.minimum(pu, pv)
                    hi = torch.maximum(pu, pv)
                    event["pointer_pairs"] = (
                        lo + hi * (hi - 1) // 2
                    ).tolist()
                if aux.get("object_triple_pointer_logits") is not None:
                    triple = aux["object_triple_pointer_logits"][:, -1]
                    used = torch.zeros(
                        triple.shape[0], triple.shape[-1],
                        device=triple.device, dtype=torch.bool,
                    )
                    chosen = []
                    for slot in range(3):
                        logits = triple[:, slot].clone()
                        logits[used] = torch.finfo(logits.dtype).min
                        choice = logits.argmax(dim=-1)
                        chosen.append(choice)
                        used[
                            torch.arange(
                                used.shape[0], device=used.device
                            ),
                            choice,
                        ] = True
                    vertices = torch.stack(chosen, dim=-1).sort(dim=-1).values
                    a, bmid, c = (
                        vertices[:, 0], vertices[:, 1], vertices[:, 2]
                    )
                    event["pointer_triples"] = (
                        a + bmid * (bmid - 1) // 2
                        + c * (c - 1) * (c - 2) // 6
                    ).tolist()
                object_events.append(event)
            next_sidecar = build_sidecar(ids[:, -max_ctx:])
            if (
                next_sidecar is not None and fired is not None
                and next_sidecar.event_multi.shape[0] >= 2
            ):
                # ``fired`` describes the token that was just appended before
                # this sidecar rebuild: it is the previous row (-2), not the
                # new token row (-1).  Keep the deterministic tokenizer event
                # and add the model's predicted event to the same position.
                with torch.no_grad():
                    row_index = -2
                    row = next_sidecar.event_multi[row_index].to(
                        fired.device
                    )
                    fused = torch.maximum(
                        row, fired[0].to(row.dtype)
                    ).detach().cpu()
                    next_sidecar.event_multi[row_index] = fused
                    structure.observe_multi(fused)
            elif fired is not None:
                structure.observe_events(
                    idx for idx, value in enumerate(fired[0].tolist())
                    if value
                )
            sidecar = next_sidecar
        if decoder is not None:
            composition = None
            if hasattr(layer_tokenizer, "decode_token_ids"):
                try:
                    text = layer_tokenizer.decode_token_ids(
                        ids[0].detach().cpu().tolist()
                    )
                    composition = [
                        event.as_dict()
                        for event in CompositionTokenizer(
                            layer_tokenizer
                        ).encode(text)
                    ]
                except Exception:
                    composition = None
            return ids, {
                "model": events,
                "tokenizer": tokenizer_events,
                "objects": object_events,
                "composition": composition,
            }
        return ids, events

    @torch.no_grad()
    def generate_speculative(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int = 16,
        temperature: float = 1.0,
        seed: int | None = None,
        task_id: int | None = None,
    ):
        """Self-speculative generation using the first MTP head as a drafter.

        The main head proposes token t+1 and the MTP head proposes t+2.  One
        verification forward on context+draft1 checks the t+2 draft against
        the main head.  Accepted drafts append two tokens; a mismatch falls
        back to the verification head's token.  This is a correctness-preserving
        decode path, not a throughput claim; no tokenizer sidecar is built here.
        """
        if tokens.ndim != 2 or tokens.size(0) != 1:
            raise ValueError("speculative generation currently supports batch=1")
        if max_new_tokens < 1:
            return tokens, {"accepted": 0, "drafted": 0, "forwards": 0}
        if seed is not None:
            torch.manual_seed(seed)
        ids = tokens
        max_ctx = max(1, int(self.cfg.max_ctx))
        accepted = 0
        drafted = 0
        forwards = 0

        def choose(logits: torch.Tensor) -> torch.Tensor:
            if temperature <= 0:
                return logits.argmax(dim=-1)
            probs = torch.softmax(
                logits.float() / max(1e-3, float(temperature)), dim=-1
            )
            return torch.multinomial(probs, num_samples=1).squeeze(-1)

        task_ids = (
            torch.full(
                (1,), int(task_id), device=ids.device, dtype=torch.long
            )
            if task_id is not None else None
        )
        while ids.size(1) - tokens.size(1) < max_new_tokens:
            window = ids[:, -max_ctx:]
            logits, mtp_logits, _aux = self(window, task_ids=task_ids)
            forwards += 1
            draft1 = choose(logits[:, -1])
            remaining = max_new_tokens - (ids.size(1) - tokens.size(1))
            if (
                remaining >= 2
                and isinstance(mtp_logits, (list, tuple))
                and len(mtp_logits) > 0
            ):
                draft2 = choose(mtp_logits[0][:, -1])
                drafted += 1
                verify_window = torch.cat(
                    [window, draft1[:, None]], dim=1
                )[:, -max_ctx:]
                verify_logits, _verify_mtp, _verify_aux = self(
                    verify_window, task_ids=task_ids
                )
                forwards += 1
                verified2 = choose(verify_logits[:, -1])
                if int(verified2.item()) == int(draft2.item()):
                    accepted += 1
                    ids = torch.cat(
                        [ids, draft1[:, None], draft2[:, None]], dim=1
                    )
                else:
                    ids = torch.cat(
                        [ids, draft1[:, None], verified2[:, None]], dim=1
                    )
            else:
                ids = torch.cat([ids, draft1[:, None]], dim=1)
        return ids, {
            "accepted": accepted,
            "drafted": drafted,
            "forwards": forwards,
            "acceptance": accepted / max(1, drafted),
        }

    def grow_combinatorial_vertices(self, new_vertices: int) -> "V3LM":
        if self.comb_memory is not None:
            self.comb_memory.grow(new_vertices)
        return self

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
