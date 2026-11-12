#!/usr/bin/env python3
"""PAL R7.12: exact causal streaming for the *directed correction*.

Persistent state: ONE legacy 64x64 matrix plus a 768-token ring of old tokens.
The directed matrix is transiently reconstructed when reading; it is not a
second persistent memory or a second model. The legacy matrix evolves by its
inherited CrossWindowMirror.advance. The incoming token leaves the 2048-token
recent context and is therefore causal, not the current prediction target.

At initial time the output equals R7.11's full-window initialization exactly.
At later times it equals the specified recurrence: legacy state advanced with
CrossWindowMirror.advance and the correction recomputed from the exact sliding
ring. DO NOT compare the legacy recurrence to reinitializing all 96 blocks: the
original legacy model intentionally uses different block-vs-token updates.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from r74_stream import CrossWindowMirror,OLDER
from r711_selective_writer import HORIZON,SelectivePairMirror

@dataclass
class StreamingState:
    legacy: torch.Tensor
    ring: torch.Tensor

class DirectedStreaming:
    def __init__(self,model):
        self.model=model
        self.outer=model.r74.outer
        if not isinstance(self.outer,SelectivePairMirror):
            raise TypeError('R7.12 requires the same R7.11 trained writer')

    def initial(self,history):
        if history.ndim!=2 or history.shape[1]!=OLDER:
            raise ValueError('history must contain exactly the older 6144 tokens')
        E=self.model.trunk.E
        legacy=CrossWindowMirror.initial(self.outer,history,E)
        return StreamingState(legacy,history[:,-HORIZON:].long().clone())

    def matrix(self,state):
        if not isinstance(state,StreamingState): raise TypeError('expected StreamingState')
        g=torch.tanh(self.outer.dir_gate)
        # This branch skips the expensive directed read, and is bitwise equal
        # to the accepted memory at exactly zero. Enable the branch during
        # joint gate training using force_gradient=True below.
        if not self.force_gradient and float(g.detach())==0.:
            return state.legacy
        directed,_,_=self.outer.pair_state(state.ring,self.model.trunk.E)
        return state.legacy+g*(directed-state.legacy)

    force_gradient=False

    def advance(self,state,leaving):
        if leaving.ndim!=1 or len(leaving)!=len(state.ring):
            raise ValueError('leaving must be a vector with one token per batch')
        next_legacy=CrossWindowMirror.advance(self.outer,state.legacy,leaving,self.model.trunk.E)
        next_ring=torch.cat((state.ring[:,1:],leaving[:,None].long()),dim=1)
        return StreamingState(next_legacy,next_ring)

    def forward(self,ex,long,ctx,pieces,lens,state,history=None,target=None,**kwargs):
        return self.model(ex,long,ctx,pieces,lens,self.matrix(state),history,target,**kwargs)


def history_shift(history,leaving):
    return torch.cat((history[:,1:],leaving[:,None].long()),dim=1)
