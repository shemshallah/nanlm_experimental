#!/usr/bin/env python3
"""R7.30: point-factorized causal triangle keys/values, inside ONE R7 Diamond.

The already-learned R7.29 geometry and optimizer retain their Python Parameter
identities and moments.  Two zero-initialized gains preserve exact parent word
predictions; the second gain exposes the retrieved signal to the existing
branch decoder *after* the apex->pair->point descent.  No gold target enters
either the query or the retrieved value.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from fa1_r729_contextual_geometry import ContextualGeometry, attach_contextual

ROUND = 'FA1_R7_30_FACTORIZED_GEOMETRY_V1'
FACTOR_TOPK = 8
FACTOR_GROUP = 'factorized_readout'


class FactorizedContextualGeometry(ContextualGeometry):
    """O(256 * d) learned point factors; O(56 * d) addressed triangles/query."""

    def __init__(self, width=32):
        super().__init__(width=width)
        self._install_factors(width)

    def _install_factors(self, width):
        # This method also serves an in-place migration.  Never reinitialize
        # Module.__init__ on an existing trained geometry object.
        self.factor_key = nn.Embedding(256, width)
        self.factor_value = nn.Embedding(256, width)
        nn.init.normal_(self.factor_key.weight, std=.02)
        nn.init.normal_(self.factor_value.weight, std=.02)
        self.factor_query = nn.Linear(width, width, bias=False)
        self.factor_readout = nn.Sequential(
            nn.Linear(3 * width, 2 * width), nn.GELU(),
            nn.Linear(2 * width, 96, bias=False))
        self.factor_trust = nn.Linear(3 * width, 1)
        nn.init.zeros_(self.factor_trust.weight)
        nn.init.zeros_(self.factor_trust.bias)
        self.factor_up_gain = nn.Parameter(torch.zeros(()))
        self.factor_decoder_gain = nn.Parameter(torch.zeros(()))

    def pairs(self, trunk, vertex_ids, pair_ix, ids_i, ids_j, pair, q):
        pair, state = super().pairs(trunk, vertex_ids, pair_ix, ids_i, ids_j, pair, q)
        state['factor_vertex_ids'] = vertex_ids
        return pair, state

    def triangles_up(self, vertex_ids, tri_ids, tri, state):
        tri, state = super().triangles_up(vertex_ids, tri_ids, tri, state)
        # Factorize each exact C(256,3) address using its three inherited
        # vertex IDs.  Orientation-sensitive S-/curvature modulates VALUES;
        # the original exact-address signed face embeddings are still read.
        ids = vertex_ids[:, self.tri_local].long()
        keys = self.factor_key(ids).sum(2) / math.sqrt(3.)
        values = self.factor_value(ids).sum(2) / math.sqrt(3.)
        even = self.face_plus(tri_ids).squeeze(-1)
        odd = self.face_minus(tri_ids).squeeze(-1)
        signed = even + torch.sin(state['face_curvature']) * odd
        values = values * (1. + .25 * torch.tanh(signed).unsqueeze(-1))
        query = F.normalize(self.factor_query(state['context_query']), dim=-1)
        keys = F.normalize(keys + .25 * tri, dim=-1)
        scores = 4. * (query.unsqueeze(1) * keys).sum(-1)
        indices = scores.topk(min(FACTOR_TOPK, scores.shape[1]), dim=1).indices
        sparse = torch.full_like(scores, -1.e4).scatter(1, indices, scores.gather(1, indices))
        alpha = F.softmax(sparse, dim=1)
        retrieved = (alpha.unsqueeze(-1) * values).sum(1)
        # A second, explicitly measured triangle->apex->pair->point route.
        tri = tri + torch.tanh(self.factor_up_gain) * (FACTOR_TOPK * alpha.unsqueeze(-1) *
                                                    retrieved.unsqueeze(1))
        state['factor_retrieved'] = retrieved
        state['metrics'].update({
            'factor_read_entropy': (-(alpha * alpha.clamp_min(1.e-20).log()).sum(1).mean()).detach(),
            'factor_value_norm': retrieved.detach().norm(dim=-1).mean(),
            'factor_up_gain': self.factor_up_gain.detach().clone(),
            'factor_decoder_gain': self.factor_decoder_gain.detach().clone(),
        })
        return tri, state

    def decoder_residual(self, state, point_summary, apex, query):
        """Inject only at inherited 96D branch hidden, never into parent trunk.

        Both apex and descended point message are required: ablating either
        makes this *additional* signal exactly zero.  The zero-gain migration
        also remains strictly function-preserving.
        """
        retrieved = state['factor_retrieved']
        fused = torch.cat([retrieved * torch.tanh(point_summary),
                           point_summary, query], dim=-1)
        confidence = torch.sigmoid(self.factor_trust(fused))
        route = (torch.tanh(point_summary.norm(dim=-1, keepdim=True)) *
                 torch.tanh(apex.norm(dim=-1, keepdim=True)))
        return (.5 * torch.tanh(self.factor_decoder_gain) * confidence * route *
                self.factor_readout(fused))


def attach_factorized(model, optimizer, learning_rate=2.e-4):
    """Append one AdamW group, preserving every R7.29 parameter and moment."""
    geometry = model.diamond.geometry
    if type(geometry) is not ContextualGeometry:
        raise TypeError('R7.30 migrates only an exact R7.29 geometry')
    previous = {id(p) for p in model.parameters()}
    width = geometry.point_offset if hasattr(geometry, 'point_offset') else geometry.edge_plus.out_features
    geometry.__class__ = FactorizedContextualGeometry
    geometry._install_factors(width)
    added = [(name, p) for name, p in model.named_parameters() if id(p) not in previous]
    if len(added) != 10 or any(id(p) in previous for _, p in added):
        raise AssertionError('R7.30 factorized Parameter registration changed')
    names = [name for name, _ in added]
    if not all(n.startswith('diamond.geometry.factor_') for n in names):
        raise AssertionError('unexpected R7.30 optimizer parameter')
    if any(id(p) not in {id(x) for x in model.parameters()} for group in optimizer.param_groups
           for p in group['params']):
        raise AssertionError('inherited optimizer detached')
    optimizer._name_by_id.update({id(p): name for name, p in added})
    optimizer._group_names.append({'algorithm': 'adamw', 'names': names})
    optimizer._adamw.add_param_group({'params': [p for _, p in added],
                                      'lr': learning_rate, 'weight_decay': 0.})
    optimizer._source_group_counts[-1] += 1
    optimizer.param_groups = optimizer._legacy_groups + optimizer._adamw.param_groups
    return model, optimizer


def load_factorized(parent, root, threads=2):
    """Load a full R7.29/R7.30 state; never throw away inherited moments."""
    from fa1_r727_checkpoint import load_unified
    model, optimizer, base, comp, metadata = load_unified(parent, root, threads)
    if metadata.get('architecture_round') == ROUND:
        if type(model.diamond.geometry) is not FactorizedContextualGeometry:
            raise AssertionError('R7.30 checkpoint restoration failed')
    else:
        if type(model.diamond.geometry) is not ContextualGeometry:
            attach_contextual(model, optimizer)
        attach_factorized(model, optimizer)
    return model, optimizer, base, comp, metadata
