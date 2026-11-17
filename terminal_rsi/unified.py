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


def static_upgrade_parameter_count(cfg: Preset) -> int:
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
        2 * d + 1                             # feedback and refine scale
    )


def blackwell_memory_plan(*, physical_vram_gib: float = 96.0,
                          usable_fraction: float = 0.92) -> dict:
    """Preflight of the Blackwell execution profile for the unified v4 model.

    The model is the same 6.5M-parameter v3 graph on CPU, T4 and Blackwell.
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
    # One BF16 checkpoint tensor per layer per recursive pass; underestimate,
    # because attention kernels, router branches, and temporary buffers add.
    checkpoint_activation_floor = (
        cfg.microbatch * cfg.context * cfg.width *
        cfg.layers * cfg.recursive_passes * 2 / (1024 ** 3))
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
        "provisional_headroom_gib": budget - state_gib - checkpoint_activation_floor,
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


class RMSNorm(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.scale)


class CausalBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm_attn = RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.project = nn.Linear(width, width, bias=False)
        self.norm_ff = RMSNorm(width)
        self.up = nn.Linear(width, 8 * width, bias=False)
        self.down = nn.Linear(4 * width, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm_attn(x)).reshape(b, t, 3, self.heads, d // self.heads).unbind(2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.project(a.transpose(1, 2).contiguous().reshape(b, t, d))
        g, u = self.up(self.norm_ff(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(g) * u)


class TerminalRSI(nn.Module):
    """Shared parameterized RSI core. No CPU-only or GPU-only model heads.

    Hierarchical specialists feed a learned router; causal self-attention updates NOW.
    Recursive feedback uses *previous* observed prediction error, never the future.
    A learned critic estimates surprise and feeds the next refinement pass.
    Next, +2 and +3 have distinct proper predictive heads.
    """
    def __init__(self, cfg: Preset):
        super().__init__()
        self.cfg = cfg
        self.gradient_checkpointing = (
            cfg.name in ("t4", "blackwell") and cfg.width >= 512)
        d = cfg.width
        assert d % cfg.heads == 0
        self.byte = nn.Embedding(256, d)
        self.position = nn.Embedding(cfg.context + 4, d)
        self.specialists = nn.ModuleList([CausalConv(d, k) for k in (4, 16, 64)])
        self.router = nn.Linear(d, 4)
        self.backbone = nn.ModuleList([CausalBlock(d, cfg.heads) for _ in range(cfg.layers)])
        self.norm = RMSNorm(d)
        self.output = nn.ModuleList([nn.Linear(d, 256, bias=False) for _ in range(3)])
        self.critic = nn.Sequential(RMSNorm(d), nn.Linear(d, d // 4), nn.SiLU(), nn.Linear(d // 4, 1))
        self.feedback = nn.Linear(2, d, bias=False)
        self.refine_scale = nn.Parameter(torch.tensor(-3.0))
        # RSI-v4 intervention target.  This is deliberately a plain Python
        # attribute, not a buffer, so old v103 checkpoints load unchanged.
        self.router_temperature = 1.0
        with torch.no_grad():
            self.router.bias.copy_(torch.tensor([2.0, 0.0, 0.0, 0.0]))
            self.feedback.weight.zero_()

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
        for r in range(rounds):
            basis = self.norm(h)
            temperature = float(getattr(self, "router_temperature", 1.0))
            choice = F.softmax(self.router(basis) / max(temperature, 1e-3),
                               dim=-1).to(h.dtype)
            route_entropies.append(-(choice.float() * choice.float().clamp_min(1e-9).log()).sum(-1).mean())
            branches = [basis] + [expert(basis) for expert in self.specialists]
            mixed = sum(choice[..., j:j+1] * branches[j] for j in range(4))
            h = h + 0.25 * (mixed - basis)
            for layer in self.backbone:
                if self.training and self.gradient_checkpointing:
                    from torch.utils.checkpoint import checkpoint
                    h = checkpoint(layer, h, use_reentrant=False)
                else:
                    h = layer(h)
            state = self.norm(h)
            raw_next = self.output[0](state)
            predicted_surprise = F.softplus(self.critic(state).squeeze(-1))
            if r + 1 < rounds:
                # At offset i, token x[i] is already known. The previous prediction
                # at i-1 can be scored; predicting x[i+1] may use that error.
                if t > 1:
                    previous_error = F.cross_entropy(
                        raw_next[:, :-1].transpose(1, 2).float(),
                        x[:, 1:], reduction="none").detach()
                    past_surprise = F.pad(previous_error, (1, 0))
                signal = torch.stack([past_surprise.clamp_max(16), predicted_surprise], dim=-1)
                h = h + torch.sigmoid(self.refine_scale) * self.feedback(signal.to(h.dtype))
        return {
            "logits": torch.stack([head(self.norm(h)) for head in self.output], dim=0),
            "critic": predicted_surprise,
            "past_surprise": past_surprise,
            "route_entropy": torch.stack(route_entropies).mean(),
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
