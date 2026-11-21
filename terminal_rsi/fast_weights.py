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
    """Rank-r byte-logit side head that starts at zero but can learn.

    Both factors used to start at zero.  That makes the output zero *and*
    kills the gradient: ``dL/dB = (state @ A)^T dL/dout`` is zero when ``A``
    is zero, and ``dL/dA`` is zero when ``B`` is zero.  The adapter therefore
    stayed exactly zero even when unfrozen.  Keep the output zero at init with
    a LoRA-style split: random ``A``, zero ``B``.  ``B`` then receives a real
    gradient on the first step while the base logits are unchanged.
    """

    def __init__(self, input_dim: int, vocab: int = 256, rank: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.vocab = vocab
        self.rank = rank
        self.A = nn.Parameter(torch.randn(input_dim, rank) * 0.02)
        self.B = nn.Parameter(torch.zeros(rank, vocab))
        self.bias = nn.Parameter(torch.zeros(vocab))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return (state @ self.A) @ self.B + self.bias

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def rank_adapter_parameter_count(input_dim: int, vocab: int = 256,
                                 rank: int = 8) -> int:
    return input_dim * rank + rank * vocab + vocab


@torch.no_grad()
def reopen_fast_adapter(model: torch.nn.Module, *, seed: int = 103) -> bool:
    """Re-seed a legacy all-zero adapter so its first gradient is non-zero.

    Old checkpoints were trained with ``A == B == 0``.  Simply unfreezing that
    adapter does not revive it.  Replace only the dead ``A`` factor; ``B`` and
    ``bias`` stay at their checkpoint values, so the adapter output is still
    zero until it actually learns.
    """
    adapter = getattr(getattr(model, "upgrades", None), "fast_adapter", None)
    if adapter is None:
        return False
    a = adapter.A.detach()
    b = adapter.B.detach()
    if float(a.abs().max()) != 0.0 or float(b.abs().max()) != 0.0:
        return False
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    replacement = torch.randn(a.shape, generator=generator) * 0.02
    adapter.A.copy_(replacement.to(device=a.device, dtype=a.dtype))
    return True
