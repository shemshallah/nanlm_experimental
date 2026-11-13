#!/usr/bin/env python3
"""R7.29 sparse context-to-triangle geometric memory, same original R7 model.

No replacement trunk, tokenizer, embedding bank, pair register, apex or decoder.
One new scalar controls a sparse attention read of the already-trained signed
S+/S- triangle codes. At zero gain, every inherited prediction is unchanged.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from fa1_r724_geometry import OrientedSimplicialGeometry

ROUND='FA1_R7_29_CONTEXTUAL_GEOMETRY_V1'
PARAMETER='diamond.geometry.context_read_gain'
READ_TOPK=8


class ContextualGeometry(OrientedSimplicialGeometry):
    """Query signed persistent simplicial addresses using the current context.

    The exact 2,763,520 possible triangle codes stay sparse: each example
    addresses only C(8,3)=56 triangles and reads the top eight. The read is
    delivered *to triangles*, so the existing apex -> pair -> point route
    remains the only means of reaching the inherited word decoder.
    """

    def pairs(self,trunk,vertex_ids,pair_ix,ids_i,ids_j,pair,q):
        pair,state=super().pairs(trunk,vertex_ids,pair_ix,ids_i,ids_j,pair,q)
        state['context_query']=q
        return pair,state

    def triangles_up(self,vertex_ids,tri_ids,tri,state):
        tri,state=super().triangles_up(vertex_ids,tri_ids,tri,state)
        query=F.normalize(state['context_query'],dim=-1)
        keys=F.normalize(tri,dim=-1)
        even=self.face_plus(tri_ids).squeeze(-1)
        odd=self.face_minus(tri_ids).squeeze(-1)
        signed_value=even+odd*torch.sin(state['face_curvature'])
        # Target-free memory selection: context query and signed stored values
        # are computed only from previously observed word/byte/sentence history.
        scores=4.*(keys*query.unsqueeze(1)).sum(-1)+.1*torch.tanh(signed_value)
        chosen=torch.topk(scores,k=min(READ_TOPK,scores.shape[1]),dim=-1).indices
        masked=torch.full_like(scores,-1.e4)
        masked.scatter_(1,chosen,scores.gather(1,chosen))
        alpha=F.softmax(masked,dim=-1)
        values=tri*(1.+.2*torch.tanh(signed_value).unsqueeze(-1))
        retrieved=(alpha.unsqueeze(-1)*values).sum(1)
        # A signed associative residual only at the addressed triangles.
        # It propagates into the *existing* learned apex and downward channel.
        # Do not send (retrieved - local triangle) here: because retrieved is
        # itself alpha-weighted, sum_i alpha_i*(retrieved-tri_i) is exactly
        # ZERO. That cancels almost all geometric signal at the apex pool.
        # Sending the signed retrieved VALUE instead makes the read visible
        # to the apex while retaining a sparse, normalized top-k allocation.
        contribution=alpha.unsqueeze(-1)*retrieved.unsqueeze(1)*READ_TOPK
        tri=tri+torch.tanh(self.context_read_gain)*contribution
        state['metrics']['context_read_entropy']=(-(alpha*alpha.clamp_min(1.e-20).log()).sum(-1).mean()).detach()
        state['metrics']['context_read_norm']=retrieved.detach().norm(dim=-1).mean()
        state['metrics']['context_read_gain']=self.context_read_gain.detach().clone()
        return tri,state


def attach_contextual(model, optimizer, learning_rate=2.e-4):
    """Function-preserving in-place migration: every inherited Parameter ID stays.

    The one public optimizer retains its 13 inherited groups and moments; one
    new AdamW group holds only the new read gain. This avoids rebuilding any
    state and makes the old optimizer `step()` operate on one model as before.
    """
    geometry=model.diamond.geometry
    if type(geometry) is not OrientedSimplicialGeometry:
        raise TypeError('R7.29 only migrates the exact R7.27/R7.28 geometry class')
    if hasattr(geometry,'context_read_gain'):
        raise ValueError('contextual geometry has already been migrated')
    existing={id(p) for p in model.parameters()}
    geometry.__class__=ContextualGeometry
    gate=nn.Parameter(torch.zeros(()))
    geometry.register_parameter('context_read_gain',gate)
    if id(gate) in existing or PARAMETER not in dict(model.named_parameters()):
        raise AssertionError('R7.29 scalar was not uniquely registered')
    optimizer._name_by_id[id(gate)]=PARAMETER
    optimizer._group_names.append({'algorithm':'adamw','names':[PARAMETER]})
    optimizer._adamw.add_param_group({'params':[gate], 'lr':learning_rate,
                                      'weight_decay':0.})
    optimizer._source_group_counts[-1]+=1
    optimizer.param_groups=optimizer._legacy_groups+optimizer._adamw.param_groups
    return model,optimizer


def load_contextual(parent,root,threads=2):
    """Migration of R7.28 trained candidate without any optimizer reset."""
    from fa1_r727_checkpoint import load_unified
    model,opt,base,comp,metadata=load_unified(parent,root,threads)
    if metadata.get('architecture_round')==ROUND:
        if type(model.diamond.geometry) is not ContextualGeometry:
            raise AssertionError('R7.29 restoration failed')
        return model,opt,base,comp,metadata
    attach_contextual(model,opt)
    return model,opt,base,comp,metadata
