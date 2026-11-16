"""Shared FA2 geometric-address counterfactuals for diagnostics and training.

All controls are *temporary* interventions on the inherited model. Trainable
parameters, Adam state, learned embeddings and saved checkpoint schema remain
untouched. Exact-address tokens stay causal, and all hooks/buffers restore even
when a forward pass raises an exception.
"""
from contextlib import contextmanager

import numpy as np
import torch


MODES = ('exact', 'pair_address_scrambled', 'triangle_address_scrambled',
         'incidence_scrambled', 'all_scrambled', 'reciprocal_disabled')


def fixed_permutations(seed):
    """Deterministic bijections (use a *different* seed each TRAIN step)."""
    rng = np.random.default_rng(seed + 50017)
    result = {name: torch.from_numpy(rng.permutation(size).astype(np.int64))
              for name, size in (('points', 8), ('edges', 28), ('faces', 56),
                                 ('tetra', 70), ('pair_addresses', 32640),
                                 ('triangle_hashes', 8192))}
    if any(torch.equal(p, torch.arange(len(p))) for p in result.values()):
        raise AssertionError('A deterministic shuffle unexpectedly is identity')
    return result


def chain_relabelled(d1, d2, d3, permutations):
    """Relabel chain complex, break the fixed point/edge/face token alignment."""
    p, e, f, t = (permutations[x] for x in ('points', 'edges', 'faces', 'tetra'))
    first = d1[e][:, p].clone()
    second = d2[f][:, e].clone()
    third = d3[t][:, f].clone()
    if not (torch.equal(second @ first, torch.zeros_like(second @ first)) and
            torch.equal(third @ second, torch.zeros_like(third @ second))):
        raise AssertionError('Scrambling must preserve the boundary-of-boundary identities')
    return {'d1': first, 'd2': second, 'd3': third}


@contextmanager
def address_ablation(model, mode, permutations):
    """Remap lookups/buffers reversibly, including during autograd TRAIN graphs."""
    if mode not in MODES:
        raise ValueError(f'Unknown geometric address ablation: {mode}')
    hooks, originals = [], []
    reciprocal = model.reciprocal
    if reciprocal is None:
        raise ValueError('Address controls need the inherited reciprocal model')
    previous_enabled = reciprocal.enabled
    reciprocal.enabled = mode != 'reciprocal_disabled'
    try:
        if mode not in ('exact', 'reciprocal_disabled'):
            geometries = [m for m in model.modules() if getattr(m, 'simplex', None)
                          is not None and hasattr(m.simplex, 'pair_shared')]
            if not geometries:
                raise ValueError('No inherited simplex geometry to ablate')
            for geometry in geometries:
                if mode in ('pair_address_scrambled', 'all_scrambled'):
                    perm = permutations['pair_addresses']

                    def remap_pair(_module, args, address_map=perm):
                        return (address_map[args[0]],) + args[1:]

                    hooks.append(geometry.simplex.pair_shared.register_forward_pre_hook(remap_pair))
                    hooks.append(geometry.simplex.pair_anti.register_forward_pre_hook(remap_pair))
                if mode in ('triangle_address_scrambled', 'all_scrambled'):
                    perm = permutations['triangle_hashes']

                    def remap_triangle(_module, args, address_map=perm):
                        return (address_map[args[0]],) + args[1:]

                    if getattr(geometry, 'triangle_table', None) is None:
                        raise ValueError('Inherited geometry lacks triangle-hash table')
                    hooks.append(geometry.triangle_table.register_forward_pre_hook(remap_triangle))
                if mode in ('incidence_scrambled', 'all_scrambled'):
                    simplex = geometry.simplex
                    # Swap buffer *references*: an earlier exact forward may
                    # still need the original tensors for its backward pass.
                    # In-place copy_/restore would increment their autograd
                    # version counters and invalidate that pending graph.
                    before = {name: getattr(simplex, name)
                              for name in ('d1', 'd2', 'd3')}
                    altered = chain_relabelled(before['d1'], before['d2'],
                                               before['d3'], permutations)
                    originals.append((simplex, before))
                    for name, value in altered.items():
                        setattr(simplex, name, value)
        yield
    finally:
        for hook in hooks:
            hook.remove()
        for simplex, buffers in reversed(originals):
            for name, value in buffers.items():
                setattr(simplex, name, value)
        reciprocal.enabled = previous_enabled
