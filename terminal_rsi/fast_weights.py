"""Low-rank fast-weight head for weight-space RSI branches.

The branch runner trains these parameters on its branch batch.  If the branch
wins, the low-rank delta persists in the main model; otherwise the snapshot is
discarded.  ``scale`` starts at zero, so the base model is unchanged until the
branch/RSI opens it.
"""
from __future__ import annotations

import torch
from torch import nn


class FastWeightHead(nn.Module):
    def __init__(self, width: int, vocab: int = 256, rank: int = 8):
        super().__init__()
        self.width = width
        self.vocab = vocab
        self.rank = rank
        self.down = nn.Parameter(torch.randn(width, rank) * 0.02)
        self.up = nn.Parameter(torch.randn(rank, vocab) * 0.02)
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] != self.width:
            raise ValueError("Fast-weight head expects [.., width]")
        delta = (state @ self.down) @ self.up
        return torch.tanh(self.scale) * delta

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def fast_weight_parameter_count(width: int, vocab: int = 256,
                                rank: int = 8) -> int:
    return width * rank + rank * vocab + 1


class RankAdapter(nn.Module):
    """Zero-initialized rank-r byte-logit side head for fast RSI proposals."""

    def __init__(self, input_dim: int, vocab: int = 256, rank: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.vocab = vocab
        self.rank = rank
        self.A = nn.Parameter(torch.zeros(input_dim, rank))
        self.B = nn.Parameter(torch.zeros(rank, vocab))
        self.bias = nn.Parameter(torch.zeros(vocab))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return (state @ self.A) @ self.B + self.bias

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def rank_adapter_parameter_count(input_dim: int, vocab: int = 256,
                                 rank: int = 8) -> int:
    return input_dim * rank + rank * vocab + vocab
