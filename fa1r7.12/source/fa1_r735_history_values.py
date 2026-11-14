#!/usr/bin/env python3
"""R7.35 causal far-history values inside the inherited FA1 Diamond.

Older retrieved modal *contents* are written to the selected sparse triangle
values. Selection remains target-free; all other geometric/LM weights and
their optimizer state survive an in-place, zero-gated migration.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

ROUND = 'FA1_R7_35_CAUSAL_HISTORY_VALUES_V1'


class FarHistoryValueWriter(nn.Module):
    """Per-triangle cross-attention to four *causal* far modal values."""

    def __init__(self, width=32):
        super().__init__()
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.write = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(),
                                   nn.Linear(width, width, bias=False))
        self.gain = nn.Parameter(torch.zeros(()))

    def forward(self, selected_keys, context, far_values, far_mask):
        """[B,K,D] writes; all-masked rows produce EXACTLY zero values."""
        batch, count, width = selected_keys.shape
        if (context.shape != (batch, width) or
                far_values.shape != (batch, 4, width) or
                far_mask.shape != (batch, 4)):
            raise ValueError('expected triangle/context and four causal far modes')
        q = self.query(selected_keys + context.unsqueeze(1))
        k = self.key(far_values)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(width)
        scores = scores.masked_fill(~far_mask.bool()[:, None, :], -1.e4)
        valid = far_mask.bool().unsqueeze(1).to(scores.dtype)
        alpha = F.softmax(scores, dim=-1) * valid
        alpha = alpha / alpha.sum(-1, keepdim=True).clamp_min(1.e-12)
        attended = torch.matmul(alpha, self.value(far_values))
        # Only observed far tokens can create a value. In particular, learned
        # biases/triangle codes cannot fabricate history for an empty mask.
        signal = self.write(torch.cat([selected_keys, attended], dim=-1))
        return signal * far_mask.any(-1)[:, None, None].to(signal.dtype)


def attach_history_values(model, optimizer, learning_rate=1.5e-4):
    """Append one AdamW group while preserving every inherited Parameter ID."""
    from fa1_r730_factorized_geometry import FactorizedContextualGeometry

    geometry = model.diamond.geometry
    if type(geometry) is not FactorizedContextualGeometry:
        raise TypeError('far-history writer requires existing R7.30 geometry')
    if hasattr(geometry, 'history_writer'):
        raise ValueError('history value writer is already attached')
    old_ids = {id(p) for p in model.parameters()}
    geometry.history_writer = FarHistoryValueWriter(geometry.factor_key.embedding_dim)
    added = [(name, p) for name, p in model.named_parameters() if id(p) not in old_ids]
    if not added or not all(n.startswith('diamond.geometry.history_writer.') for n, _ in added):
        raise AssertionError('unexpected history writer Parameter registration')
    optimizer._name_by_id.update({id(p): n for n, p in added})
    optimizer._group_names.append({'algorithm': 'adamw', 'names': [n for n, _ in added]})
    optimizer._adamw.add_param_group({'params': [p for _, p in added],
                                     'lr': learning_rate, 'weight_decay': 0.})
    optimizer._source_group_counts[-1] += 1
    optimizer.param_groups = optimizer._legacy_groups + optimizer._adamw.param_groups
    if {id(p) for g in optimizer.param_groups for p in g['params']} != {
            id(p) for p in model.parameters()}:
        raise AssertionError('history writer detached inherited optimizer state')
    return model, optimizer


def open_new_causal_gates(model, source_round, history_gain=0.08,
                          apex_gain=0.06, resume_override=False):
    """Candidate-only warm start; never reset existing learned moments.

    A zero-attached module learns only its scalar gate on the first update.
    Opening new gates *after* identity verification also trains the internal
    attention/value weights. Existing R7.35 gates are retained unless the
    caller explicitly opts into a recorded continuation experiment.
    """
    if not (0 <= history_gain < .5 and 0 <= apex_gain < .5):
        raise ValueError('causal gate warm starts must be in [0, 0.5)')
    h = getattr(model.diamond, 'hierarchy', None)
    writer = getattr(model.diamond.geometry, 'history_writer', None)
    if writer is None:
        raise ValueError('cannot warm-start an absent history value writer')
    result = {'source_round':source_round,
              'writer_before':float(writer.gain.detach()),
              'apex_before':float(h.apex_feedback_gain.detach())
                             if h is not None and hasattr(h,'apex_feedback_gain') else None,
              'resumed_override':bool(resume_override)}
    with torch.no_grad():
        if source_round not in (ROUND, 'FA1_R7_36_DEEP_WIRING_V1',
                               'FA1_R7_37_CLOSED_FEEDBACK_V1') or resume_override:
            writer.gain.fill_(math.atanh(history_gain))
        if h is not None and hasattr(h,'apex_feedback_gain') and (
                source_round not in (ROUND,'FA1_R7_36_DEEP_WIRING_V1',
                                     'FA1_R7_37_CLOSED_FEEDBACK_V1',
                                     'FA1_R7_UNIFIED_APEX_FEEDBACK_V1')
                or resume_override):
            h.apex_feedback_gain.fill_(math.atanh(apex_gain))
    result['writer_after']=float(writer.gain.detach())
    result['apex_after']=(float(h.apex_feedback_gain.detach())
                          if h is not None and hasattr(h,'apex_feedback_gain') else None)
    return result


def different_book_writer_values(data, seed=735, preserve_receiver_mask=False):
    """Counterfactual far VALUES only, with genuinely eligible foreign donors.

    A masked donor is not counterfactual *content*: it would turn a content
    comparison into an accidental writer-off comparison. Require a distinct
    eligible book for every receiver with real far-history evidence.
    """
    from fa1_r725_controls import different_book_donors

    ids=data.get('diamond_modal_ids',data.get('memory_modal_ids'))
    mask=data.get('diamond_modal_mask',data.get('memory_modal_mask'))
    books=torch.as_tensor(data['book_ids']).detach().cpu().reshape(-1)
    if ids is None or mask is None or ids.shape!=(len(books),12) or mask.shape!=ids.shape:
        raise ValueError('expected twelve original near/far modal slots')
    donors=different_book_donors(books,seed)
    eligible=mask[:,8:].bool().any(1).detach().cpu()
    rng=torch.Generator(device='cpu').manual_seed(int(seed)+103)
    for book in torch.unique(books[eligible]):
        receivers=torch.where((books==book)&eligible)[0]
        candidates=torch.where((books!=book)&eligible)[0]
        if not candidates.numel():
            raise ValueError('writer-value contrast requires two distinct far-eligible books')
        selected=torch.randint(len(candidates),(len(receivers),),generator=rng)
        donors[receivers]=candidates[selected]
    wrong=dict(data)
    foreign_ids=ids[donors.to(ids.device),8:].clone()
    foreign_mask=mask[donors.to(mask.device),8:].bool()
    if preserve_receiver_mask:
        # R7.36 isolates CONTENT rather than accidentally measuring missing
        # memory coverage. Keep each target's original eligibility pattern,
        # filling its valid slots only with genuine eligible foreign tokens.
        receiver_mask=mask[:,8:].bool()
        for row in range(len(books)):
            if not bool(receiver_mask[row].any()):
                continue
            eligible_foreign=foreign_ids[row,foreign_mask[row]]
            if not eligible_foreign.numel():
                raise AssertionError('eligible-donor selection was violated')
            count=int(receiver_mask[row].sum())
            foreign_ids[row,receiver_mask[row]]=eligible_foreign[
                torch.arange(count,device=ids.device)%len(eligible_foreign)]
        wrong['diamond_history_value_mask']=receiver_mask.clone()
    else:
        wrong['diamond_history_value_mask']=foreign_mask.clone()
    wrong['diamond_history_value_ids']=foreign_ids
    return wrong,donors
