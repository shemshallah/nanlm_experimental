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
                         shards: int = 8, noise: float = 0.02,
                         seed: int | None = None) -> dict:
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
    with torch.no_grad():
        noop_shards = _noop_loss(
            model, adapter, probe, per_sample=True).reshape(
                shard_count, -1).mean(-1)
    for _round in range(rounds):
        with torch.no_grad():
            A0 = adapter.A.detach()
            B0 = adapter.B.detach()
            bias0 = adapter.bias.detach()
            A = A0[None] + noise * torch.randn(
                candidates, A0.shape[0], A0.shape[1], device=device)
            B = B0[None] + noise * torch.randn(
                candidates, B0.shape[0], B0.shape[1], device=device)
            bias = bias0[None] + 0.01 * torch.randn(
                candidates, bias0.shape[0], device=device)
        population = AdapterPopulation(A, B, bias).to(device)
        optimizer = torch.optim.Adam(population.parameters(), lr=lr)
        for _step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = population.loss(model, branch).sum()
            if not torch.isfinite(loss):
                break
            loss.backward()
            nn.utils.clip_grad_norm_(population.parameters(), 1.0)
            optimizer.step()
        with torch.no_grad():
            sample_loss = population.loss(
                model, probe, per_sample=True)  # [K,N]
            candidate_shards = sample_loss.reshape(
                candidates, shard_count, -1).mean(-1)
            diff = noop_shards[None] - candidate_shards
            mean_diff = diff.mean(-1)
            if shard_count > 1:
                se = diff.std(-1, unbiased=True) / math.sqrt(shard_count)
            else:
                se = torch.full_like(mean_diff, float("inf"))
            index = int(torch.argmax(mean_diff).item())
            delta = float(mean_diff[index].item())
            standard_error = float(se[index].item())
            threshold = max(float(min_gain), float(z_score) * standard_error)
            proposals += candidates
            if delta > threshold:
                accepted += 1
                adapter.A.copy_(population.A[index].detach())
                adapter.B.copy_(population.B[index].detach())
                adapter.bias.copy_(population.bias[index].detach())
                best_delta = max(best_delta, delta)
                best_se = min(best_se, standard_error)
    noop_bpb = float(
        _noop_loss(model, adapter, probe).item() * 8 / math.log(2))
    return {
        "proposals": proposals,
        "accepted": accepted,
        "rounds": rounds,
        "best_delta": best_delta if math.isfinite(best_delta) else 0.0,
        "best_standard_error": best_se if math.isfinite(best_se) else None,
        "noop_probe_bpb": noop_bpb,
        "adapter_gate": float(adapter.A.abs().sum().item()),
    }
