"""Terminal-RSI: one recursively self-improving byte predictor on CPU, T4 or Blackwell.

Source of truth: same causal model and checkpoint schema for all three presets.
This is a new candidate lineage; legacy v103 C weights are NOT silently substituted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import math
import torch
from torch import nn
from torch.nn import functional as F

from .hierarchical_bus import CausalHierarchy, hierarchy_parameter_count
from .low_rank_moe import SparseRankOneMoE, moe_parameter_count
from .fast_weights import (
    FastWeightHead, RankAdapter, fast_weight_parameter_count,
    rank_adapter_parameter_count,
)
from .unit_heads import (
    SENTENCE_VOCAB_SIZE, TOKEN_VOCAB_SIZE, WORD_VOCAB_SIZE,
)
from .norm import RMSNorm


@dataclass(frozen=True)
class Preset:
    name: str
    width: int
    layers: int
    heads: int
    context: int
    microbatch: int
    recursive_passes: int
    lr: float
    grad_accum: int


PRESETS = {
    # One unified ~100M v4 graph.  CPU/T4/Blackwell are execution profiles:
    # parameter keys, shapes and counts are identical.  ``microbatch`` is the
    # hardware batch hint used by preflight planning; ``train_verbatim --batch``
    # controls the actual runtime batch (Blackwell: 8192).
    "cpu": Preset("cpu", 704, 12, 11, 256, 2, 2, 3e-4, 1),
    "t4": Preset("t4", 704, 12, 11, 256, 64, 2, 3e-4, 1),
    "blackwell": Preset("blackwell", 704, 12, 11, 256, 8192, 2, 3e-4, 1),
}

# Original v103 bit-path core, unchanged in every execution profile.
ORIGINAL_V103_PARAMETER_COUNT = 2_782_515


def fast_adapter_input_width(cfg: Preset,
                             word_vocab_size: int = WORD_VOCAB_SIZE) -> int:
    bus = max(16, cfg.width // 4)
    return (cfg.width + 3 * bus + 256 + 256
            + 2 * (TOKEN_VOCAB_SIZE + int(word_vocab_size))
            + SENTENCE_VOCAB_SIZE)


def symbiotic_parameter_count(cfg: Preset, hidden: int = 64,
                              word_vocab_size: int = WORD_VOCAB_SIZE) -> int:
    adapter = fast_adapter_input_width(cfg, word_vocab_size)
    return (adapter + 256) * hidden + hidden + hidden * cfg.width + cfg.width + 2


def moe_layer_count(cfg: Preset) -> int:
    """Layers that receive the 256-expert low-rank MoE (indices 1,4,7,10)."""
    return sum(1 for index in range(cfg.layers) if index % 3 == 1)


def static_upgrade_parameter_count(cfg: Preset,
                                   word_vocab_size: int = WORD_VOCAB_SIZE) -> int:
    """Exact TerminalRSI new-parameter count without allocating Blackwell weights."""
    d = cfg.width
    c = cfg.context
    if d % 4:
        raise ValueError("Critic width must be divisible by four")
    return (
        256 * d + (c + 4) * d +               # byte and position embeddings
        (4 + 16 + 64) * d + 3 * d * d +        # three causal conv specialists
        4 * d + 4 +                            # four-way routing
        cfg.layers * (16 * d * d + 2 * d) +    # causal SDPA/GLU, two RMSNorm
        d + 3 * 256 * d +                      # final norm, next/+2/+3
        d + d * (d // 4) + (d // 4) +          # critic norm + hidden
        (d // 4) + 1 +                        # critic scalar output
        2 * d + 1 +                           # feedback and refine scale
        d + 1 +                               # adaptive-depth head
        d + 1 +                               # per-token upgrade gate
        fast_weight_parameter_count(d) +       # low-rank weight-space RSI head
        rank_adapter_parameter_count(
            fast_adapter_input_width(cfg, word_vocab_size), rank=16) +  # side head
        symbiotic_parameter_count(
            cfg, word_vocab_size=word_vocab_size) +  # feedback refinement
        moe_layer_count(cfg) * moe_parameter_count(d) +  # 256-expert rank-1 MoE
        hierarchy_parameter_count(d, word_vocab_size)  # causal hierarchy
    )


def blackwell_memory_plan(*, physical_vram_gib: float = 96.0,
                          usable_fraction: float = 0.92) -> dict:
    """Preflight of the Blackwell execution profile for the unified v4 model.

    The model is the same 100M-parameter v3 graph on CPU, T4 and Blackwell.
    FP32 weights/grads/Adam moments occupy ~16 bytes/parameter.  BF16 autocast
    is used only for the additive upgrade branch; archived v103 logits, RSI
    actor/critic and inherited AdamW groups stay in their original FP32 path.
    """
    cfg = PRESETS["blackwell"]
    if not 0.5 <= usable_fraction <= 1.0 or physical_vram_gib <= 0:
        raise ValueError("Invalid Blackwell VRAM or reserve fraction")
    params = static_upgrade_parameter_count(cfg)
    total_params = params + ORIGINAL_V103_PARAMETER_COUNT + 1  # + residual gate
    state_gib = total_params * 16 / (1024 ** 3)
    # Two-layer checkpoint segments: 12 stored inputs for 2 passes, while only
    # two layers are recomputed at a time.  (Per-layer storage OOMs at 8192;
    # whole-stack recompute OOMs on the 12-layer activation peak.)
    segments = math.ceil(cfg.layers / 4)
    checkpoint_activation_floor = (
        cfg.recursive_passes * segments * cfg.microbatch * cfg.context *
        cfg.width * 2 / (1024 ** 3))
    estimated_recompute_peak = (
        4 * cfg.microbatch * cfg.context * cfg.width * 4 * 2 / (1024 ** 3))
    budget = physical_vram_gib * usable_fraction
    return {
        "preset": "blackwell",
        "parameter_count_additive_specialist": params,
        "parameter_count_original_v103": ORIGINAL_V103_PARAMETER_COUNT,
        "parameter_count_total": total_params,
        "hidden_width": cfg.width,
        "layers": cfg.layers,
        "context_bytes": cfg.context,
        "recursive_passes": cfg.recursive_passes,
        "microbatch": cfg.microbatch,
        "physical_vram_gib": physical_vram_gib,
        "usable_vram_budget_gib": budget,
        "fp32_weights_grad_adam_gib_approx": state_gib,
        "bf16_checkpoint_activation_floor_gib": checkpoint_activation_floor,
        "estimated_recompute_peak_gib": estimated_recompute_peak,
        "provisional_headroom_gib": (
            budget - state_gib - checkpoint_activation_floor - estimated_recompute_peak),
        "bf16_cuda_upgrade_only": True,
        "archived_v103_fp32_unchanged": True,
        "same_three_optimizer_groups": True,
        "native_pytorch_causal_sdpa": True,
        "gradient_checkpointing": True,
        "gpu_peak_verified": False,
        "note": "Same ~100M graph on all modes; Blackwell profile uses batch 8192, BF16 upgrade autocast and checkpointing",
    }


def device_for(mode: str) -> torch.device:
    if mode == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(f"{mode} preset requires CUDA; run cpu mode first")
    if mode not in ("t4", "blackwell"):
        raise ValueError(f"Unknown execution profile: {mode}")
    if mode == "blackwell":
        props = torch.cuda.get_device_properties(0)
        if props.total_memory < 8 * (1024 ** 3):
            raise RuntimeError(
                "Blackwell profile requires a CUDA GPU with at least 8 GiB; "
                f"found {props.total_memory / 2**30:.1f} GiB")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "Blackwell profile requires BF16 support; use --mode t4 or cpu")
    return torch.device("cuda")


class CausalConv(nn.Module):
    """Truncated left-only convolution: no right-hand information."""
    def __init__(self, width: int, kernel: int):
        super().__init__()
        self.kernel = kernel
        self.depthwise = nn.Conv1d(width, width, kernel, groups=width, bias=False)
        self.pointwise = nn.Linear(width, width, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = F.pad(h.transpose(1, 2), (self.kernel - 1, 0))
        return self.pointwise(self.depthwise(x).transpose(1, 2))


class CausalBlock(nn.Module):
    def __init__(self, width: int, heads: int, *, use_moe: bool = False):
        super().__init__()
        self.heads = heads
        self.norm_attn = RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.project = nn.Linear(width, width, bias=False)
        self.norm_ff = RMSNorm(width)
        # Fused 8x GLU: one matmul instead of separate gate/value launches.
        # The split pair is kept in checkpoint migration for v4 compatibility.
        self.gate_value = nn.Linear(width, 8 * width, bias=False)
        self.down = nn.Linear(4 * width, width, bias=False)
        self.moe = SparseRankOneMoE(width) if use_moe else None

    def forward(self, x: torch.Tensor, cache: dict | None = None) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm_attn(x)).reshape(b, t, 3, self.heads, d // self.heads).unbind(2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        if cache is not None:
            if cache.get("k") is not None:
                k = torch.cat((cache["k"], k), dim=2)
                v = torch.cat((cache["v"], v), dim=2)
            cache["k"] = k
            cache["v"] = v
            a = F.scaled_dot_product_attention(
                q, k, v, is_causal=(q.shape[2] > 1))
        else:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.project(a.transpose(1, 2).contiguous().reshape(b, t, d))
        normed = self.norm_ff(x)
        gate, value = self.gate_value(normed).chunk(2, dim=-1)
        x = x + self.down(F.silu(gate).mul_(value))
        if self.moe is not None:
            x = self.moe(x)
        return x


class TerminalRSI(nn.Module):
    """Shared parameterized RSI core. No CPU-only or GPU-only model heads.

    Hierarchical specialists feed a learned router; causal self-attention updates NOW.
    Recursive feedback uses *previous* observed prediction error, never the future.
    A learned critic estimates surprise and feeds the next refinement pass.
    Next, +2 and +3 have distinct proper predictive heads.
    """
    def __init__(self, cfg: Preset, word_vocab_size: int = WORD_VOCAB_SIZE):
        super().__init__()
        self.cfg = cfg
        self.word_vocab_size = int(word_vocab_size)
        self.gradient_checkpointing = (
            cfg.name in ("t4", "blackwell") and cfg.width >= 512)
        d = cfg.width
        assert d % cfg.heads == 0
        self.byte = nn.Embedding(256, d)
        self.position = nn.Embedding(cfg.context + 4, d)
        self.specialists = nn.ModuleList([CausalConv(d, k) for k in (4, 16, 64)])
        self.router = nn.Linear(d, 4)
        self.backbone = nn.ModuleList([
            CausalBlock(d, cfg.heads, use_moe=(index % 3 == 1))
            for index in range(cfg.layers)])
        self.norm = RMSNorm(d)
        self.output = nn.ModuleList([nn.Linear(d, 256, bias=False) for _ in range(3)])
        self.critic = nn.Sequential(RMSNorm(d), nn.Linear(d, d // 4), nn.SiLU(), nn.Linear(d // 4, 1))
        self.feedback = nn.Linear(2, d, bias=False)
        self.refine_scale = nn.Parameter(torch.tensor(-3.0))
        self.fast_weight = FastWeightHead(d)
        self.depth_head = nn.Linear(d, 1)
        # Per-token parent/upgrade mixing gate.  Zero-initialized so the
        # inherited v103 logits stay exact until this head learns.
        self.upgrade_gate = nn.Linear(d, 1)
        self.hierarchy = CausalHierarchy(d, word_vocab_size=self.word_vocab_size)
        adapter_dim = (d + 3 * self.hierarchy.bus_width + 256 + 256
                       + 2 * (TOKEN_VOCAB_SIZE + self.word_vocab_size)
                       + SENTENCE_VOCAB_SIZE)
        self.fast_adapter = RankAdapter(adapter_dim, rank=16)
        self.fast_adapter.requires_grad_(False)
        self.symbiotic_feedback = nn.Sequential(
            nn.Linear(adapter_dim + 256, 64),
            nn.GELU(),
            nn.Linear(64, d),
        )
        self.symbiosis_gate = nn.Parameter(torch.zeros(()))
        self.symbiosis_output_gate = nn.Parameter(torch.zeros(()))
        # RSI-v4 intervention target.  This is deliberately a plain Python
        # attribute, not a buffer, so old v103 checkpoints load unchanged.
        self.router_temperature = 1.0
        with torch.no_grad():
            self.router.bias.copy_(torch.tensor([2.0, 0.0, 0.0, 0.0]))
            self.feedback.weight.zero_()
            self.depth_head.bias.fill_(4.0)  # start close to the two-pass path
            # Start the upgrade branch uniform (8 bits/byte), not with large
            # random logits.  Gate 0 still returns exact v103; direct aux CE
            # now starts at 8 instead of ~65 and trains immediately.
            for head in self.output:
                head.weight.zero_()
            nn.init.zeros_(self.upgrade_gate.weight)
            nn.init.zeros_(self.upgrade_gate.bias)

    @staticmethod
    def migrate_state_dict(state_dict: dict) -> dict:
        """Migrate v4 split-GLU keys to the fused v5 GLU without data loss."""
        migrated = dict(state_dict)
        for key in list(migrated):
            if ((key.startswith("upgrades.backbone.")
                 or key.startswith("backbone."))
                    and key.endswith(".gate.weight")):
                prefix = key[:-len("gate.weight")]
                value_key = prefix + "value.weight"
                if value_key in migrated:
                    migrated[prefix + "gate_value.weight"] = torch.cat(
                        (migrated.pop(key), migrated.pop(value_key)), dim=0)
        return migrated

    def load_state_dict(self, state_dict, strict: bool = True,
                        assign: bool = False):
        migrated = self.migrate_state_dict(dict(state_dict))
        missing, unexpected = super().load_state_dict(
            migrated, strict=False, assign=assign)
        allowed = ("fast_weight.", "depth_head.", "fast_adapter.",
                   "hierarchy.pair_memory.", "hierarchy.unit_heads.",
                   "upgrades.fast_weight.", "upgrades.depth_head.",
                   "upgrades.hierarchy.pair_memory.",
                   "upgrades.hierarchy.unit_heads.")
        bad_missing = [
            key for key in missing
            if not key.startswith(allowed) and ".moe." not in key]
        if strict and (unexpected or bad_missing):
            raise RuntimeError(
                f"Checkpoint mismatch: missing={bad_missing[:4]} "
                f"unexpected={unexpected[:4]}")
        return torch.nn.modules.module._IncompatibleKeys(missing, unexpected)

    def _backbone_forward(self, h: torch.Tensor,
                          caches: list | None = None) -> torch.Tensor:
        for index, layer in enumerate(self.backbone):
            h = layer(h, cache=caches[index] if caches is not None else None)
        return h

    def _segment_forward(self, h: torch.Tensor, start: int, end: int) -> torch.Tensor:
        for layer in self.backbone[start:end]:
            h = layer(h)
        return h

    def forward(self, x: torch.Tensor, passes: Optional[int] = None) -> Dict[str, torch.Tensor]:
        if x.dtype != torch.long or x.ndim != 2:
            raise ValueError("x must be [batch,time] torch.long byte indices")
        if x.shape[1] > self.cfg.context + 1:
            raise ValueError("sequence exceeds preset context")
        rounds = self.cfg.recursive_passes if passes is None else passes
        if not 1 <= rounds <= self.cfg.recursive_passes:
            raise ValueError("passes must be between 1 and preset recursion count")
        b, t = x.shape
        loc = torch.arange(t, device=x.device)
        h = self.byte(x) + self.position(loc)[None, :, :]
        past_surprise = torch.zeros((b, t), dtype=h.dtype, device=x.device)
        predicted_surprise = torch.zeros_like(past_surprise)
        route_entropies = []
        h_first = None
        for r in range(rounds):
            basis = self.norm(h)
            temperature = float(getattr(self, "router_temperature", 1.0))
            choice = F.softmax(self.router(basis) / max(temperature, 1e-3),
                               dim=-1).to(h.dtype)
            route_entropies.append(-(choice.float() * choice.float().clamp_min(1e-9).log()).sum(-1).mean())
            branches = [basis] + [expert(basis) for expert in self.specialists]
            mixed = sum(choice[..., j:j+1] * branches[j] for j in range(4))
            h = h + 0.25 * (mixed - basis)
            if self.training and self.gradient_checkpointing:
                from torch.utils.checkpoint import checkpoint
                for start in range(0, len(self.backbone), 4):
                    h = checkpoint(
                        self._segment_forward, h, start,
                        min(start + 4, len(self.backbone)),
                        use_reentrant=True)
            else:
                h = self._backbone_forward(h)
            if r == 0:
                h_first = h
            if r + 1 < rounds:
                # At offset i, token x[i] is already known. The previous prediction
                # at i-1 can be scored; predicting x[i+1] may use that error.
                intermediate = self.norm(h)
                raw_next = self.output[0](intermediate)
                predicted_surprise = F.softplus(
                    self.critic(intermediate).squeeze(-1))
                if t > 1:
                    previous_error = F.cross_entropy(
                        raw_next[:, :-1].transpose(1, 2).float(),
                        x[:, 1:], reduction="none").detach()
                    past_surprise = F.pad(previous_error, (1, 0))
                signal = torch.stack([past_surprise.clamp_max(16), predicted_surprise], dim=-1)
                h = h + torch.sigmoid(self.refine_scale) * self.feedback(signal.to(h.dtype))
        if rounds > 1 and h_first is not None:
            depth_prob = torch.sigmoid(self.depth_head(self.norm(h_first)))
            h = h_first + depth_prob * (h - h_first)
        else:
            depth_prob = torch.ones((b, t, 1), dtype=h.dtype, device=h.device)
        # Final causal sweep: byte -> subword -> word -> sentence, then the
        # top-down and lateral priors.  Every hierarchy edge is zero-gated, so
        # at initialization this is exactly the previous byte-state path.
        final_state = self.norm(h)
        bus = self.hierarchy(final_state)
        enhanced = bus.state
        fast = self.fast_weight(enhanced)
        zeros_t = torch.zeros((enhanced.shape[0], TOKEN_VOCAB_SIZE),
                              device=enhanced.device, dtype=enhanced.dtype)
        zeros_w = torch.zeros((enhanced.shape[0], self.word_vocab_size),
                              device=enhanced.device, dtype=enhanced.dtype)
        zeros_s = torch.zeros((enhanced.shape[0], SENTENCE_VOCAB_SIZE),
                              device=enhanced.device, dtype=enhanced.dtype)
        cu_tok = bus.unit_logits.get("current_token", zeros_t)
        cu_word = bus.unit_logits.get("current_word", zeros_w)
        nx_tok = bus.unit_logits.get("token", zeros_t)
        nx_word = bus.unit_logits.get("word", zeros_w)
        nx_sent = bus.unit_logits.get("sentence", zeros_s)
        adapter_input = torch.cat(
            (enhanced[:, -1], bus.subword[:, -1], bus.word[:, -1],
             bus.sentence[:, -1], bus.unit_byte_raw, fast[:, -1],
             cu_tok, cu_word, nx_tok, nx_word, nx_sent), dim=-1)
        adapter_logits = self.fast_adapter(adapter_input)
        feedback = torch.tanh(self.symbiosis_gate) * self.symbiotic_feedback(
            torch.cat((adapter_input, adapter_logits), dim=-1))
        refined = enhanced[:, -1] + feedback
        raw_next = self.output[0](enhanced) + bus.byte_prior + fast
        refined_last = (self.output[0](refined) + bus.byte_prior[:, -1]
                        + fast[:, -1])
        raw_next = torch.cat((raw_next[:, :-1], refined_last[:, None]), dim=1)
        base_last_logits = self.output[0](enhanced)[:, -1]
        predicted_surprise = F.softplus(self.critic(enhanced).squeeze(-1))
        moe_stats = [block.moe.last_load_balance.to(enhanced.device)
                     for block in self.backbone if block.moe is not None]
        moe_losses = [block.moe.load_balance_loss.to(enhanced.device)
                      for block in self.backbone if block.moe is not None]
        moe_losses.extend((
            self.hierarchy.top_moe.load_balance_loss.to(enhanced.device),
            self.hierarchy.bottom_extensions.load_balance_loss.to(
                enhanced.device)))
        moe_entropies = [block.moe.last_entropy.to(enhanced.device)
                         for block in self.backbone if block.moe is not None]
        return {
            "logits": torch.stack(
                [raw_next, self.output[1](enhanced), self.output[2](enhanced)],
                dim=0),
            "critic": predicted_surprise,
            "past_surprise": past_surprise,
            "route_entropy": torch.stack(route_entropies).mean(),
            "boundary_logits": bus.boundary_logits,
            "boundary_probs": bus.boundary_probs,
            "hierarchy_gates": bus.gates,
            "unit_logits": bus.unit_logits,
            "unit_byte_raw": bus.unit_byte_raw,
            "unit_byte_logits": bus.unit_byte_logits,
            "adapter_input": adapter_input,
            "last_base_logits": base_last_logits,
            "last_refined_logits": refined_last,
            "gate_features": enhanced[:, -1],
            "word_state": bus.word,
            "byte_state": enhanced,
            "pair_entropy": bus.pair_entropy,
            "pair_gate": bus.pair_gate,
            "depth_prob": depth_prob.mean(),
            "moe_load_balance": (torch.stack(moe_losses).mean()
                                 if moe_losses else torch.zeros((), device=enhanced.device)),
            "moe_load_balance_telemetry": (torch.stack(moe_stats).mean()
                                           if moe_stats else torch.zeros((), device=enhanced.device)),
            "moe_entropy": (torch.stack(moe_entropies).mean()
                            if moe_entropies else torch.zeros((), device=enhanced.device)),
        }

    def forward_prefill(self, x: torch.Tensor) -> dict:
        """One-pass prefill that returns per-layer K/V and hierarchy buffers."""
        if x.dtype != torch.long or x.ndim != 2:
            raise ValueError("prefill expects [batch,time] byte ids")
        b, t = x.shape
        h = self.byte(x) + self.position(
            torch.arange(t, device=x.device))[None, :, :]
        basis = self.norm(h)
        temperature = float(getattr(self, "router_temperature", 1.0))
        choice = F.softmax(self.router(basis) / max(temperature, 1e-3),
                           dim=-1).to(h.dtype)
        branches = [basis] + [expert(basis) for expert in self.specialists]
        mixed = sum(choice[..., j:j + 1] * branches[j] for j in range(4))
        h = h + 0.25 * (mixed - basis)
        caches: list[dict] = [{"k": None, "v": None}
                              for _ in self.backbone]
        h = self._backbone_forward(h, caches)
        states = self.norm(h)
        bus = self.hierarchy(states)
        current = (self.output[0](states) + bus.byte_prior
                   + self.fast_weight(states))
        return {
            "current": current[:, -1],
            "caches": caches,
            "basis": basis,
            "states": states,
            "bus": bus,
            "gate_features": states[:, -1],
        }

    def forward_token(self, x_new: torch.Tensor, position: int,
                      caches: list[dict], basis_buffer: torch.Tensor,
                      state_buffer: torch.Tensor) -> dict:
        """One-pass cached next-byte step; updates the rolling buffers."""
        if x_new.ndim != 2 or x_new.shape[1] != 1:
            raise ValueError("cached token step expects [batch,1]")
        device = x_new.device
        position = min(int(position), self.position.num_embeddings - 1)
        h = self.byte(x_new) + self.position(
            torch.tensor([position], device=device))[None, :, :]
        basis_new = self.norm(h)
        window = torch.cat((basis_buffer[:, -63:], basis_new), dim=1)
        branches = [basis_new] + [
            expert(window)[:, -1:] for expert in self.specialists]
        temperature = float(getattr(self, "router_temperature", 1.0))
        choice = F.softmax(self.router(basis_new) / max(temperature, 1e-3),
                           dim=-1).to(h.dtype)
        mixed = sum(choice[..., j:j + 1] * branches[j] for j in range(4))
        h = h + 0.25 * (mixed - basis_new)
        h = self._backbone_forward(h, caches)
        final = self.norm(h)
        basis_buffer = torch.cat((basis_buffer, basis_new), dim=1)[:, -256:]
        state_buffer = torch.cat((state_buffer, final), dim=1)[:, -256:]
        bus = self.hierarchy(state_buffer)
        enhanced = bus.state[:, -1:]
        current = (self.output[0](enhanced) + bus.byte_prior[:, -1:]
                   + self.fast_weight(enhanced))
        return {
            "current": current[:, -1],
            "bus": bus,
            "basis_buffer": basis_buffer,
            "state_buffer": state_buffer,
            "gate_features": enhanced[:, 0],
        }

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def loss_and_stats(out: dict, raw: torch.Tensor, valid: Optional[torch.Tensor] = None):
    """raw includes input bytes plus three future bytes; valid rejects book padding."""
    x_len = out["logits"].shape[2]
    if raw.shape[1] < x_len + 3:
        raise ValueError("raw needs input plus three lookahead bytes")
    if valid is None:
        valid = torch.ones_like(raw, dtype=torch.bool)
    losses, stats = [], {}
    for horizon in range(1, 4):
        logits = out["logits"][horizon - 1].float()
        targets = raw[:, horizon:horizon + x_len]
        mask = valid[:, :x_len] & valid[:, horizon:horizon + x_len]
        ce = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
        den = mask.sum().clamp_min(1)
        sums = (ce * mask).sum()
        acc = (logits.argmax(-1) == targets) & mask
        top5 = (logits.topk(5, -1).indices == targets.unsqueeze(-1)).any(-1) & mask
        stats[f"h{horizon}_bits"] = (sums / den / math.log(2)).detach()
        stats[f"h{horizon}_top1"] = (acc.sum() / den).detach()
        stats[f"h{horizon}_top5"] = (top5.sum() / den).detach()
        stats[f"h{horizon}_bytes"] = den.detach()
        losses.append(sums / den)
    # Critic learns expected *future* prediction error; targets are detached.
    with torch.no_grad():
        next_logits = out["logits"][0].float()
        expected_error = F.cross_entropy(
            next_logits.transpose(1, 2), raw[:, 1:x_len + 1], reduction="none")
    mask1 = valid[:, :x_len] & valid[:, 1:x_len + 1]
    critic_loss = ((out["critic"].float() - expected_error) ** 2 * mask1).sum() / mask1.sum().clamp_min(1)
    training_loss = losses[0] + 0.15 * (losses[1] + losses[2]) + 0.03 * critic_loss
    stats["critic_mse"] = critic_loss.detach()
    stats["route_entropy"] = out["route_entropy"].detach()
    return training_loss, stats


class RSIController:
    """Conservative self-modifying *training policy*, based only on DEV data.

    This does not edit executable source and does not mutate accepted checkpoints.
    It changes learning rate and recursion passes; every decision is auditable and
    reversible from the full optimizer/controller checkpoint.
    """
    def __init__(self, cfg: Preset):
        self.learning_rate = cfg.lr
        self.passes = cfg.recursive_passes
        self.max_passes = cfg.recursive_passes
        self.best_dev_bpb = float("inf")
        self.regressions = 0
        self.history = []

    def observe(self, optimizer: torch.optim.Optimizer, step: int, dev_bpb: float):
        if not math.isfinite(dev_bpb):
            raise RuntimeError("Non-finite DEV measurement; training halted")
        improved = dev_bpb < self.best_dev_bpb - 1e-5
        if improved:
            self.best_dev_bpb = dev_bpb
            self.regressions = 0
            if self.passes < self.max_passes:
                self.passes += 1
            action = "accept_dev"
        else:
            self.regressions += 1
            if self.regressions >= 2:
                self.learning_rate = max(1e-6, self.learning_rate * 0.5)
                self.regressions = 0
                action = "halve_lr"
            else:
                action = "hold"
        for group in optimizer.param_groups:
            group["lr"] = self.learning_rate
        self.history.append({"step": step, "dev_bpb": dev_bpb,
                             "action": action, "lr": self.learning_rate,
                             "passes": self.passes})
        return improved
