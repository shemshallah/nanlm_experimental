"""Paired concept memory for the v5 hierarchy.

The geometry closes exactly:

    C(256, 2)                    = 32,640 unordered mirror pairs
    2 * C(256, 2) + 256          = 65,536 = 256^2 = 2 * 32^3
    bottom 32^3 lattice          = 32,640 pairs + 128 diagonal points
    top 32^3 lattice             = the 32,640 mirrors + the other 128 points

The bank stores one low-dimensional concept per ordered pair address.  The
bottom (subword) state selects/writes a concept; the top (sentence) state reads
it back through a zero-gated residual.  This is a sparse point-cloud
hypergraph, not a dense 256-expert matrix.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

PAIR_SLOTS = 256
N_PAIR = PAIR_SLOTS * (PAIR_SLOTS - 1) // 2
N_PALINDROME = PAIR_SLOTS
PAIR_CONCEPT_COUNT = 2 * (32 ** 3)  # 65,536 = 2 * C(256,2) + 256


def pair_addresses(left: torch.Tensor, right: torch.Tensor,
                   slots: int = PAIR_SLOTS) -> torch.Tensor:
    """Map two slot ids onto the closed 2*C(slots,2)+slots address space.

    Ranges (for slots=256):
      0..32639       bottom triangular pairs (left < right)
      32640..32767   bottom special points (left == right < 128)
      32768..65407   top mirrored pairs (left > right)
      65408..65535   top special points (left == right >= 128)
    """
    lo = torch.minimum(left, right)
    hi = torch.maximum(left, right)
    triangular = (lo * slots - lo * (lo + 1) // 2
                  + (hi - lo - 1).clamp_min(0))
    top_base = PAIR_CONCEPT_COUNT // 2  # 32,768 = one 32^3 lattice
    return torch.where(
        left < right, triangular,
        torch.where(left > right, top_base + triangular,
                    torch.where(left < slots // 2,
                                N_PAIR + left,
                                top_base + N_PAIR + (left - slots // 2))))


@dataclass
class PairMemoryState:
    sentence: torch.Tensor
    pair_index: torch.Tensor
    pair_entropy: torch.Tensor
    pair_gate: torch.Tensor


class PairConceptMemory(nn.Module):
    def __init__(self, bus_width: int, *, slots: int = PAIR_SLOTS,
                 concept_dim: int = 16, top_k: int = 2):
        super().__init__()
        if slots != PAIR_SLOTS:
            raise ValueError("The 2*32^3 closed geometry requires 256 slots")
        if PAIR_CONCEPT_COUNT != 2 * N_PAIR + N_PALINDROME:
            raise AssertionError("pair/palindrome/lattice count mismatch")
        self.bus_width = bus_width
        self.slots = slots
        self.concept_dim = concept_dim
        self.top_k = top_k
        self.bank = nn.Embedding(PAIR_CONCEPT_COUNT, concept_dim)
        self.router = nn.Linear(bus_width, slots, bias=False)
        self.write = nn.Linear(bus_width, concept_dim, bias=False)
        self.read = nn.Linear(concept_dim, bus_width, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.bank.weight, std=0.02)
        nn.init.normal_(self.router.weight, std=0.02)
        nn.init.normal_(self.write.weight, std=0.02)
        nn.init.normal_(self.read.weight, std=0.02)

    def forward(self, bottom: torch.Tensor, top: torch.Tensor) -> PairMemoryState:
        if bottom.shape != top.shape or bottom.shape[-1] != self.bus_width:
            raise ValueError("Pair memory expects matching [B,T,bus_width] states")
        scores = self.router(bottom)
        top_values, top_indices = scores.topk(self.top_k, dim=-1)
        left = top_indices[..., 0]
        right = top_indices[..., 1]
        # 128+128 special-point split:
        # bottom = 32,640 triangular pairs + 128 diagonal points
        # top    = 32,640 mirrored pairs + the other 128 diagonal points
        pair_index = pair_addresses(left, right, self.slots)
        concept = self.bank(pair_index) + self.write(bottom)
        prior = self.read(concept)
        gate = torch.tanh(self.gate)
        sentence = top + gate * prior
        weights = F.softmax(top_values, dim=-1)
        entropy = -(weights * weights.clamp_min(1e-9).log()).sum(-1).mean()
        return PairMemoryState(
            sentence=sentence,
            pair_index=pair_index,
            pair_entropy=entropy,
            pair_gate=gate.detach(),
        )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def pair_parameter_count(bus_width: int, concept_dim: int = 16) -> int:
    return (PAIR_CONCEPT_COUNT * concept_dim + bus_width * PAIR_SLOTS
            + bus_width * concept_dim + concept_dim * bus_width + 1)
