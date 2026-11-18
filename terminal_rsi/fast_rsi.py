"""Cached fast-adapter RSI: thousands of proposals without 100M clones.

The frozen model is run once per tick on a branch batch and the probe set.
All candidate proposals then train/evaluate a rank-r byte-logit adapter on the
cached hierarchy states with small matmuls.  An accepted adapter persists in
the model as a final side head.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class FastCache:
    inputs: torch.Tensor
    base_bits: torch.Tensor
    bits: torch.Tensor
    targets: torch.Tensor

    def to(self, device: torch.device) -> "FastCache":
        return FastCache(
            self.inputs.to(device), self.base_bits.to(device),
            self.bits.to(device), self.targets.to(device))


@torch.no_grad()
def build_fast_cache(model, history: torch.Tensor, target: torch.Tensor,
                     aux: torch.Tensor, *, chunk: int = 256) -> FastCache:
    """Run the frozen model once and cache adapter inputs + base bit logits."""
    device = next(model.parameters()).device
    rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(target), chunk):
            part_history = history[start:start + chunk].to(device)
            part_target = target[start:start + chunk].to(device)
            part_aux = aux[start:start + chunk].to(device)
            logits, bits, _current, out = model(
                part_history, part_target, part_aux,
                return_aux=True, return_hierarchy=True,
                use_fast_adapter=False)
            rows.append((out["adapter_input"].detach(), logits.detach(),
                         bits.detach(), part_target.detach()))
    finally:
        model.train(was_training)
    return FastCache(
        inputs=torch.cat([row[0] for row in rows], dim=0),
        base_bits=torch.cat([row[1] for row in rows], dim=0),
        bits=torch.cat([row[2] for row in rows], dim=0),
        targets=torch.cat([row[3] for row in rows], dim=0),
    )


def _adapter_bits(model, logits: torch.Tensor,
                  target: torch.Tensor) -> torch.Tensor:
    candidates, length, _vocab = logits.shape
    flat = logits.reshape(candidates * length, -1)
    repeated = target[None, :].expand(candidates, length).reshape(-1)
    bits = model._byte_to_conditional_bit_logits(flat, repeated)
    return bits.reshape(candidates, length, 8)


def _adapter_loss(model, A: torch.Tensor, B: torch.Tensor,
                  bias: torch.Tensor, cache: FastCache,
                  *, per_sample: bool = False) -> torch.Tensor:
    z = torch.einsum("nd,kdr->knr", cache.inputs, A)
    logits = torch.einsum("knr,krv->knv", z, B) + bias[:, None, :]
    bits = _adapter_bits(model, logits, cache.targets)
    target_bits = cache.bits[None].expand(
        logits.shape[0], -1, -1)
    loss = F.binary_cross_entropy_with_logits(
        cache.base_bits[None] + bits, target_bits, reduction="none")
    if per_sample:
        return loss.mean(dim=-1)
    return loss.mean(dim=(1, 2))


class AdapterPopulation(nn.Module):
    def __init__(self, A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.A = nn.Parameter(A.clone())
        self.B = nn.Parameter(B.clone())
        self.bias = nn.Parameter(bias.clone())

    def loss(self, model, cache: FastCache, *,
             per_sample: bool = False) -> torch.Tensor:
        return _adapter_loss(model, self.A, self.B, self.bias, cache,
                             per_sample=per_sample)


def _noop_loss(model, adapter, cache: FastCache,
               *, per_sample: bool = False) -> torch.Tensor:
    A = adapter.A.detach()[None]
    B = adapter.B.detach()[None]
    bias = adapter.bias.detach()[None]
    return _adapter_loss(model, A, B, bias, cache,
                         per_sample=per_sample)[0]


def run_fast_adapter_rsi(model, branch: FastCache, probe: FastCache, *,
                         candidates: int = 128, rounds: int = 4,
                         steps: int = 2, lr: float = 0.05, rank: int = 8,
                         min_gain: float = 0.0005, z_score: float = 1.5,
                         shards: int = 8, noise: float = 0.005,
                         top_k: int = 8, seed: int | None = None) -> dict:
    """Train/evaluate cached rank-r adapters and persist the best accepted one."""
    if candidates < 1 or rounds < 1 or steps < 1:
        return {"proposals": 0, "accepted": 0, "rounds": 0}
    device = branch.inputs.device
    adapter = model.upgrades.fast_adapter
    if seed is not None:
        torch.manual_seed(int(seed))
    proposals = 0
    accepted = 0
    best_delta = -float("inf")
    best_se = float("inf")
    shard_count = max(2, min(int(shards), len(probe.targets)))
    # Selection uses the first half of the probe; confirmation the second half.
    half = max(1, len(probe.targets) // 2)
    probe_sel = FastCache(probe.inputs[:half], probe.base_bits[:half],
                          probe.bits[:half], probe.targets[:half])
    probe_conf = FastCache(probe.inputs[half:], probe.base_bits[half:],
                           probe.bits[half:], probe.targets[half:])
    conf_shards = max(2, min(int(shards), len(probe_conf.targets)))
    with torch.no_grad():
        noop_conf_shards = _noop_loss(
            model, adapter, probe_conf, per_sample=True).reshape(
                conf_shards, -1).mean(-1)
    grad_steps = (0.01, 0.03, 0.1, 0.3, 1.0)
    branch_shards = max(1, min(4, len(branch.targets)))
    for _round in range(rounds):
        with torch.no_grad():
            A0 = adapter.A.detach()
            B0 = adapter.B.detach()
            bias0 = adapter.bias.detach()
            # Nonzero probe point breaks the A=B=0 saddle.
            A_ref = A0 + noise * torch.randn(
                A0.shape, device=device)
            B_ref = B0 + noise * torch.randn(
                B0.shape, device=device)
            bias_ref = bias0 + 0.01 * torch.randn(
                bias0.shape, device=device)
        candidate_A = [A0[None], A_ref[None]]
        candidate_B = [B0[None], B_ref[None]]
        candidate_bias = [bias0[None], bias_ref[None]]
        branch_size = max(1, len(branch.targets) // branch_shards)
        for shard_index in range(branch_shards):
            lo = shard_index * branch_size
            hi = len(branch.targets) if shard_index + 1 == branch_shards else (shard_index + 1) * branch_size
            part = FastCache(
                branch.inputs[lo:hi], branch.base_bits[lo:hi],
                branch.bits[lo:hi], branch.targets[lo:hi])
            A = A_ref[None].clone().requires_grad_(True)
            B = B_ref[None].clone().requires_grad_(True)
            bias = bias_ref[None].clone().requires_grad_(True)
            loss = _adapter_loss(model, A, B, bias, part).sum()
            if not torch.isfinite(loss):
                continue
            gA, gB, gb = torch.autograd.grad(loss, [A, B, bias])
            gA = gA / gA.norm().clamp_min(1e-8)
            gB = gB / gB.norm().clamp_min(1e-8)
            gb = gb / gb.norm().clamp_min(1e-8)
            for step in grad_steps:
                candidate_A.append((A_ref[None] - step * gA).detach())
                candidate_B.append((B_ref[None] - step * gB).detach())
                candidate_bias.append((bias_ref[None] - step * gb).detach())
        A_pop = torch.cat(candidate_A, dim=0)
        B_pop = torch.cat(candidate_B, dim=0)
        bias_pop = torch.cat(candidate_bias, dim=0)
        with torch.no_grad():
            selection_loss = _adapter_loss(
                model, A_pop, B_pop, bias_pop, probe_sel,
                per_sample=True).mean(-1)
            keep = min(max(1, int(top_k)), len(selection_loss))
            top = torch.argsort(selection_loss)[:keep]
            weights = torch.softmax(-selection_loss[top], dim=0)
            A_ens = (A_pop[top] * weights[:, None, None]).sum(0)[None]
            B_ens = (B_pop[top] * weights[:, None, None]).sum(0)[None]
            bias_ens = (bias_pop[top] * weights[:, None]).sum(0)[None]
            ensemble_loss = _adapter_loss(
                model, A_ens, B_ens, bias_ens, probe_conf,
                per_sample=True)[0]
            candidate_shards = ensemble_loss.reshape(
                conf_shards, -1).mean(-1)
            diff = noop_conf_shards - candidate_shards
            delta = float(diff.mean().item())
            standard_error = (float(diff.std(unbiased=True).item()
                                    / math.sqrt(conf_shards))
                              if conf_shards > 1 else float("inf"))
            threshold = max(float(min_gain), float(z_score) * standard_error)
            proposals += max(0, len(selection_loss) - 1)
            if delta > best_delta:
                best_delta = delta
                best_se = standard_error
            if delta > threshold:
                current_full = _noop_loss(model, adapter, probe)
                candidate_full = _adapter_loss(
                    model, A_ens, B_ens, bias_ens, probe)[0]
                full_delta = float(current_full.item() - candidate_full.item())
                if full_delta > 0:
                    accepted += 1
                    adapter.A.copy_(A_ens[0].detach())
                    adapter.B.copy_(B_ens[0].detach())
                    adapter.bias.copy_(bias_ens[0].detach())
                    best_delta = max(best_delta, full_delta)
    with torch.no_grad():
        if (not torch.isfinite(adapter.A).all()
                or not torch.isfinite(adapter.B).all()
                or not torch.isfinite(adapter.bias).all()
                or adapter.A.norm() > 10.0 or adapter.B.norm() > 10.0):
            adapter.A.zero_(); adapter.B.zero_(); adapter.bias.zero_()
    noop_bpb = float(
        _noop_loss(model, adapter, probe).item() * 8 / math.log(2))
    return {
        "proposals": proposals,
        "accepted": accepted,
        "rounds": rounds,
        "best_delta": best_delta,
        "best_standard_error": (best_se if math.isfinite(best_se) else None),
        "noop_probe_bpb": noop_bpb,
        "adapter_gate": float(adapter.A.abs().sum().item()),
    }
