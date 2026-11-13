#!/usr/bin/env python3
"""FA1 R7.27: one optimizer interface and serialized state for the entire R7.

The original first optimizer is *not Adam*: it is PAL6's exact AdaGrad-style
LegacyAccumulator.  Folding it into AdamW would silently destroy the accepted
training algorithm.  This class runs that original update rule for its first
group and a single native AdamW over all remaining inherited parameter groups.

It retains original group order, hyperparameters, weights and tensor moments.
The historical five-optimizer checkpoint is only an input format; successors
save exactly one optimizer state_dict and do not overwrite an accepted parent.
"""
from __future__ import annotations

import copy

import torch


FORMAT = 'FA1_R7_27_UNIFIED_HYBRID_OPTIMIZER_V1'


def _deep_equal(a, b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys() and all(
            _deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(
            _deep_equal(x, y) for x, y in zip(a, b))
    return a == b


class UnifiedR7Optimizer:
    """Single public step/zero_grad/state_dict over the inherited R7 groups."""

    def __init__(self, model, inherited, allow_empty_inherited=False):
        if len(inherited) != 5:
            raise ValueError('R7.27 requires five inherited optimizer histories')
        if any(type(o) is not torch.optim.AdamW for o in inherited[1:]):
            raise TypeError('R7.27 can merge only the four verified AdamW engines')
        if type(inherited[0]).__name__ != 'LegacyAccumulator':
            raise TypeError('the first optimizer must be the exact PAL6 LegacyAccumulator')

        self.model = model
        self._name_by_id = {id(p): name for name, p in model.named_parameters()}
        groups = [(0, g) for g in inherited[0].param_groups] + [
            (i, g) for i, opt in enumerate(inherited[1:], 1) for g in opt.param_groups]
        seen = set()
        self._group_names = []
        for index, group in groups:
            names = []
            for param in group['params']:
                if id(param) not in self._name_by_id or id(param) in seen:
                    raise ValueError('unregistered or overlapping R7 optimizer parameter')
                seen.add(id(param))
                names.append(self._name_by_id[id(param)])
            self._group_names.append({'algorithm': 'legacy_adagrad' if index == 0 else 'adamw',
                                      'names': names})
        if seen != {id(p) for p in model.parameters()}:
            missing = [n for n, p in model.named_parameters() if id(p) not in seen]
            raise ValueError('unoptimized R7 model parameters: ' + repr(missing[:12]))

        self._legacy_groups = [dict(g, params=list(g['params'])) for g in inherited[0].param_groups]
        self._legacy_state = {p: copy.deepcopy(s) for p, s in inherited[0].state.items()}
        for group in self._legacy_groups:
            for p in group['params']:
                if set(self._legacy_state.get(p, {})) != {'sum'}:
                    if allow_empty_inherited and p not in self._legacy_state:
                        # Only when constructing an empty optimizer shell for
                        # a complete R7.27 load: the saved moments replace
                        # these temporary tensors before any update is legal.
                        self._legacy_state[p] = {'sum': torch.zeros_like(p)}
                    else:
                        raise ValueError('missing or unexpected legacy accumulator state')
                if self._legacy_state[p]['sum'].shape != p.shape:
                    raise ValueError('legacy accumulator shape mismatch')

        # PyTorch AdamW already stores full settings on each original group.
        # Retain those exact dictionaries; its per-parameter state and steps
        # are independent across groups, so one backend is lossless.
        adam_groups = [dict(g, params=list(g['params'])) for opt in inherited[1:]
                       for g in opt.param_groups]
        self._adamw = torch.optim.AdamW(adam_groups)
        for opt in inherited[1:]:
            for p, state in opt.state.items():
                self._adamw.state[p] = copy.deepcopy(state)
        self.param_groups = self._legacy_groups + self._adamw.param_groups
        self._legacy_count = len(self._legacy_groups)
        self._source_group_counts = [len(o.param_groups) for o in inherited]
        self._verify_source(inherited)

    def _verify_source(self, inherited):
        for current, original in zip(self._legacy_groups, inherited[0].param_groups):
            for k in original:
                if k != 'params' and not _deep_equal(current[k], original[k]):
                    raise AssertionError('legacy optimizer group changed: ' + k)
        for p, s in inherited[0].state.items():
            if not _deep_equal(self._legacy_state[p], s):
                raise AssertionError('legacy accumulator changed')
        adam_groups = [g for opt in inherited[1:] for g in opt.param_groups]
        for current, original in zip(self._adamw.param_groups, adam_groups):
            for k, value in original.items():
                if k != 'params' and not _deep_equal(current[k], value):
                    raise AssertionError('AdamW group setting changed: ' + k)
        for opt in inherited[1:]:
            for p, state in opt.state.items():
                if not _deep_equal(self._adamw.state[p], state):
                    raise AssertionError('AdamW moment changed')

    def zero_grad(self, set_to_none=True):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('R7 legacy optimizer does not support closures')
        for group in self._legacy_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.to_dense() if p.grad.is_sparse else p.grad
                acc = self._legacy_state[p]['sum']
                acc.addcmul_(grad, grad)
                p.addcdiv_(grad, acc.sqrt().clamp_min_(1.e-15), value=-group['lr'])
        self._adamw.step()

    def state_dict(self):
        index = 0
        legacy_groups = []
        legacy_states = {}
        for group in self._legacy_groups:
            params = list(range(index, index + len(group['params'])))
            legacy_groups.append({**{k: copy.deepcopy(v) for k, v in group.items() if k != 'params'},
                                  'params': params})
            for i, p in zip(params, group['params']):
                legacy_states[i] = copy.deepcopy(self._legacy_state[p])
            index += len(params)
        return {'format': FORMAT,
                'optimizer_count': 1,
                'historical_engine_count': 5,
                'source_group_counts': list(self._source_group_counts),
                'groups': copy.deepcopy(self._group_names),
                'legacy': {'state': legacy_states, 'param_groups': legacy_groups},
                'adamw': copy.deepcopy(self._adamw.state_dict())}

    def load_state_dict(self, saved):
        if saved.get('format') != FORMAT or saved.get('optimizer_count') != 1:
            raise ValueError('not an R7.27 unified optimizer checkpoint')
        if saved['groups'] != self._group_names or saved['source_group_counts'] != self._source_group_counts:
            raise ValueError('R7 optimizer parameter inventory/order changed')
        legacy = saved['legacy']
        if len(legacy['param_groups']) != self._legacy_count:
            raise ValueError('legacy group count changed')
        new_legacy = {}
        for group, src in zip(self._legacy_groups, legacy['param_groups']):
            for key, value in src.items():
                if key != 'params':
                    group[key] = copy.deepcopy(value)
            if len(group['params']) != len(src['params']):
                raise ValueError('legacy group size changed')
            for p, i in zip(group['params'], src['params']):
                state = copy.deepcopy(legacy['state'][i])
                if set(state) != {'sum'} or state['sum'].shape != p.shape:
                    raise ValueError('invalid legacy accumulator tensor')
                new_legacy[p] = state
        self._legacy_state = new_legacy
        self._adamw.load_state_dict(copy.deepcopy(saved['adamw']))
        self.param_groups = self._legacy_groups + self._adamw.param_groups


def load_r727_geometric(parent, root, threads=2):
    """Read a protected R7.24 input; return the SAME model + one optimizer."""
    from fa1_r724_unified import load_geometric
    model, inherited, base, comp, metadata = load_geometric(parent, root, threads=threads)
    unified = UnifiedR7Optimizer(model, inherited)
    return model, unified, base, comp, metadata
