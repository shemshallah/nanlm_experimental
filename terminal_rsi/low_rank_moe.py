"""Sparse 256-expert low-rank mixture for the v5 upgrade backbone.

A full 256 x 704 x 704 MoE is ~127M parameters per layer.  Instead each expert
is rank-one: ``expert_i(x) = (x . u_i) * v_i``.  A shared router selects top-2
experts per byte, and the result is projected back through a zero-gated
residual.  This is the "256 extension heads" idea at a feasible parameter and
VRAM cost.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SparseRankOneMoE(nn.Module):
    def __init__(self, width: int, *, experts: int = 256, top_k: int = 2,
                 chunk: int = 16384):
        super().__init__()
        self.width = width
        self.experts = experts
        self.top_k = top_k
        self.chunk = chunk
        self.router = nn.Linear(width, experts, bias=False)
        self.u = nn.Parameter(torch.randn(experts, width) * 0.02)
        self.v = nn.Parameter(torch.randn(experts, width) * 0.02)
        self.out = nn.Linear(width, width, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.router.weight, std=0.02)
        nn.init.normal_(self.out.weight, std=0.02)
        # Detached telemetry; never part of state_dict.
        self.last_load_balance = torch.zeros(())
        self.last_entropy = torch.zeros(())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.width:
            raise ValueError("MoE expects [batch,time,width]")
        batch, time, width = x.shape
        flat = x.reshape(-1, width)
        scores = self.router(flat)
        top_values, top_indices = scores.topk(self.top_k, dim=-1)
        weights = F.softmax(top_values, dim=-1)
        output = torch.zeros_like(flat)
        for start in range(0, len(flat), self.chunk):
            end = min(start + self.chunk, len(flat))
            part = flat[start:end]
            part_weights = weights[start:end]
            part_indices = top_indices[start:end]
            part_output = torch.zeros_like(part)
            for slot in range(self.top_k):
                expert = part_indices[:, slot]
                direction = self.u[expert]
                magnitude = (part * direction).sum(-1, keepdim=True)
                value = self.v[expert]
                part_output = part_output + (
                    part_weights[:, slot:slot + 1] * magnitude * value)
            output[start:end] = part_output
        with torch.no_grad():
            probabilities = F.softmax(scores.float(), dim=-1).mean(0)
            counts = torch.bincount(
                top_indices[:, 0].reshape(-1), minlength=self.experts).float()
            fraction = counts / counts.sum().clamp_min(1)
            self.last_load_balance = (
                self.experts * (probabilities * fraction).sum()).detach()
            entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum()
            self.last_entropy = entropy.detach()
        return flat + torch.tanh(self.gate) * self.out(output).reshape(
            batch, time, width)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def moe_parameter_count(width: int, experts: int = 256) -> int:
    return (width * experts + 2 * experts * width + width * width + 1)
