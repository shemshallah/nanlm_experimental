"""FA2 R28: identity-initialized mirror-aware recurrent residual.

Called inside the existing PAL GRU recurrence (not a separate predictor).
Causal state-only transformation; additional capacity can learn from BPTT across
multiple consecutive, un-detached 12-word chunks.
"""
import torch
from torch import nn
from torch.nn import functional as F

class MirrorRecurrentResidual(nn.Module):
    def __init__(self, dim=96, rank=64):
        super().__init__()
        self.dim=dim
        self.norm=nn.LayerNorm(dim)
        self.up=nn.Linear(2*dim,2*rank)
        self.down=nn.Linear(rank,dim)
        self.channel_gate=nn.Linear(dim,dim)
        self.amplitude=nn.Parameter(torch.zeros(()))

    def forward(self, state):
        if state.ndim != 2 or state.shape[-1] != self.dim:
            raise ValueError('FA2 R28 state shape')
        z=self.norm(state)
        a,b=self.up(torch.cat((z,z.flip(-1)),dim=-1)).chunk(2,dim=-1)
        correction=torch.tanh(self.down(F.silu(a)*torch.sigmoid(b)))
        channels=torch.sigmoid(self.channel_gate(z))
        return state+torch.tanh(self.amplitude)*channels*correction


def bind_recurrent(carry,unified):
    if not hasattr(unified,'r28'):
        raise ValueError('missing unified R28 module')
    carry.recurrent_refiner=unified.r28
    return carry
