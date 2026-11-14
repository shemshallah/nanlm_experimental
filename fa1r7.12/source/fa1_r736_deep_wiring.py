#!/usr/bin/env python3
"""R7.36 migration-safe sparse-history and apex-feedback wiring.

This is an experiment on the same inherited FA1 network, not a replacement
decoder. Two zero-initialized scalar controls make the historical read scale
and query-only feedback measurable without altering an R7.35 parent logit.
"""
from __future__ import annotations

import contextlib
import torch
from torch import nn

ROUND = 'FA1_R7_36_DEEP_WIRING_V1'


def normalized_sparse_history_read(triangle_weights, writes, selected_indices):
    """Normalize attention over selected written faces, not unwritten slots."""
    if (triangle_weights.ndim != 2 or writes.ndim != 3 or
            selected_indices.ndim != 2 or
            writes.shape[:2] != triangle_weights.shape or
            selected_indices.size(0) != triangle_weights.size(0)):
        raise ValueError('sparse read expects [B,T] weights, [B,T,D] values and [B,K] indices')
    if selected_indices.size(1) == 0:
        raise ValueError('sparse history read must select at least one triangle')
    mass=triangle_weights.gather(1,selected_indices)
    written=writes.gather(1,selected_indices.unsqueeze(-1).expand(
        -1,-1,writes.size(-1)))
    return (mass.unsqueeze(-1)*written).sum(1)/mass.sum(1,keepdim=True).clamp_min(1.e-12)


def gradient_path_health(model):
    """Track individual trainable pathway gradients before clipping."""
    d=model.diamond
    paths={'history_query':d.geometry.history_writer.query,
           'history_key':d.geometry.history_writer.key,
           'history_value':d.geometry.history_writer.value,
           'history_write':d.geometry.history_writer.write,
           'factor_key':d.geometry.factor_key,
           'factor_value':d.geometry.factor_value,
           'apex_feedback':d.hierarchy.apex_feedback,
           'byte_decoder':d.hierarchy.byte_head,
           'point_loop':d.hierarchy.point_loop,
           'pair_loop':d.hierarchy.pair_loop,
           'triangle_loop':d.hierarchy.triangle_loop,
           'sparse_read_gate':d.geometry.history_read_mix,
           'feedback_center_gate':d.hierarchy.apex_feedback_center_gain}
    result={}
    for name,object_ in paths.items():
        params=[object_] if isinstance(object_,nn.Parameter) else list(object_.parameters())
        grads=[p.grad.detach() for p in params if p.grad is not None]
        square=sum(float(g.float().square().sum()) for g in grads)
        result[name]={'active':bool(grads) and square>0.,
                      'gradient_l2':square**.5,
                      'finite':bool(all(torch.isfinite(g).all().item() for g in grads))}
    return result


@contextlib.contextmanager
def control_disabled(parameter):
    """Restore even after a failed counterfactual; preserve optimizer state."""
    saved=parameter.detach().clone()
    with torch.no_grad():
        parameter.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            parameter.copy_(saved)


def require_eligible_minibatch(data, indices, rng):
    """Include causal far values from two books for isolated writer contrast."""
    mask=data.get('diamond_modal_mask',data.get('memory_modal_mask'))
    if mask is None or mask.ndim!=2 or mask.shape[1]!=12:
        raise ValueError('R7.36 requires twelve causal modal eligibility masks')
    if indices.numel()<2:
        raise ValueError('writer contrast requires minibatch size of at least two')
    eligible=mask[:,8:].bool().any(1)
    books=data['book_ids']
    distinct=books[eligible].unique()
    if len(distinct)<2:
        raise ValueError('writer contrast requires two far-eligible training books')
    first,second=distinct[
        torch.as_tensor(rng.choice(len(distinct),size=2,replace=False),dtype=torch.long)]
    first_positions=torch.where(eligible & (books==first))[0]
    second_positions=torch.where(eligible & (books==second))[0]
    updated=indices.clone()
    updated[0]=first_positions[int(rng.integers(len(first_positions)))]
    updated[1]=second_positions[int(rng.integers(len(second_positions)))]
    return updated


def attach_deep_wiring(model, optimizer, learning_rate=1.5e-4):
    """Append one group, retaining all model Parameter identities and moments."""
    from fa1_r730_factorized_geometry import FactorizedContextualGeometry
    from fa1_r735_history_values import FarHistoryValueWriter
    from fa1_r731_layered_decoder import LayeredDiamondDecoder

    geometry = model.diamond.geometry
    hierarchy = getattr(model.diamond, 'hierarchy', None)
    if (type(geometry) is not FactorizedContextualGeometry or
            not isinstance(getattr(geometry, 'history_writer', None), FarHistoryValueWriter) or
            not isinstance(hierarchy, LayeredDiamondDecoder) or
            not hasattr(hierarchy, 'apex_feedback')):
        raise TypeError('R7.36 requires the full trained R7.35 causal writer and apex feedback')
    if hasattr(geometry, 'history_read_mix') or hasattr(hierarchy, 'apex_feedback_center_gain'):
        raise ValueError('R7.36 deep wiring is already attached')

    previous = {id(p) for p in model.parameters()}
    geometry.history_read_mix = nn.Parameter(torch.zeros(()))
    hierarchy.apex_feedback_center_gain = nn.Parameter(torch.zeros(()))
    added = [(n, p) for n, p in model.named_parameters() if id(p) not in previous]
    expected = {'diamond.geometry.history_read_mix',
                'diamond.hierarchy.apex_feedback_center_gain'}
    if {n for n, _ in added} != expected:
        raise AssertionError('unexpected R7.36 Parameter registration')
    optimizer._name_by_id.update({id(p): n for n, p in added})
    optimizer._group_names.append({'algorithm': 'adamw', 'names': [n for n, _ in added]})
    optimizer._adamw.add_param_group({'params': [p for _, p in added],
                                     'lr': learning_rate, 'weight_decay': 0.})
    optimizer._source_group_counts[-1] += 1
    optimizer.param_groups = optimizer._legacy_groups + optimizer._adamw.param_groups
    actual = [id(p) for group in optimizer.param_groups for p in group['params']]
    if len(actual) != len(set(actual)) or set(actual) != {id(p) for p in model.parameters()}:
        raise AssertionError('R7.36 optimizer has missing or duplicated model parameters')
    return model, optimizer
