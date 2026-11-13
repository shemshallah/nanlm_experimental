#!/usr/bin/env python3
"""R7.27 learned-weight optimizer parity: actual R7.24 candidate, no training.

Runs one deterministic synthetic-gradient step on an ephemeral second model.
No real checkpoint is modified or promoted, and no benchmark is inferred from
this non-language-model gradient probe.
"""
from __future__ import annotations

import argparse
import json

import torch

from fa1_r724_unified import load_geometric
from fa1_r727_optimizer import UnifiedR7Optimizer, _deep_equal
from fa1_r727_checkpoint import digest


@torch.no_grad()
def run(args):
    original_digest = digest(args.candidate)
    model_old, opts_old, _, _, meta_old = load_geometric(args.candidate, args.root, args.threads)
    model_new, opts_new, _, _, meta_new = load_geometric(args.candidate, args.root, args.threads)
    if meta_old['format'] != meta_new['format']:
        raise AssertionError('model lineage mismatch')
    names_old = dict(model_old.named_parameters())
    names_new = dict(model_new.named_parameters())
    if names_old.keys() != names_new.keys():
        raise AssertionError('model parameter names changed')
    for name in names_old:
        if not torch.equal(names_old[name], names_new[name]):
            raise AssertionError('initial weight mismatch: ' + name)
    unified = UnifiedR7Optimizer(model_new, opts_new)
    touched = []
    for j, (group_old, group_new) in enumerate(zip(
            [g for opt in opts_old for g in opt.param_groups], unified.param_groups)):
        a = group_old['params'][0]
        b = group_new['params'][0]
        assert names_old[next(k for k, p in names_old.items() if p is a)] is a
        grad = torch.full_like(a, (j + 1) * 1e-5)
        a.grad = grad
        b.grad = grad.clone()
        touched.append((a, b))
    for opt in opts_old:
        opt.step()
    unified.step()
    for a, b in touched:
        if not torch.equal(a, b):
            raise AssertionError('real-weight parameter update mismatch')
    for a, b in zip(model_old.parameters(), model_new.parameters()):
        if not torch.equal(a, b):
            raise AssertionError('one-step model mismatch')
    for a, b in zip(opts_old[0].param_groups[0]['params'], unified._legacy_groups[0]['params']):
        if not _deep_equal(opts_old[0].state[a], unified._legacy_state[b]):
            raise AssertionError('legacy accumulator drift')
    for opt in opts_old[1:]:
        for p, state in opt.state.items():
            name = next(k for k, v in names_old.items() if v is p)
            if not _deep_equal(state, unified._adamw.state[names_new[name]]):
                raise AssertionError('AdamW state drift at ' + name)

    result = {
        'format': 'FA1_R7_27_REAL_WEIGHT_UPDATE_PARITY_V1',
        'source_checkpoint_sha256': original_digest,
        'accepted_step': 14416, 'promoted': False,
        'model_parameter_tensors': len(names_old),
        'merged_parameter_groups': len(unified.param_groups),
        'old_optimizer_instances': len(opts_old), 'new_optimizer_instances': 1,
        'old_legacy_accumulator_algorithm_preserved': True,
        'all_original_weights_preserved_at_migration': True,
        'one_step_update_bitwise_equal': True,
        'all_optimizer_moments_bitwise_equal': True,
        'synthetic_probe_groups': len(touched),
    }
    # Do not persist the synthetic-gradient weights. Saving the original
    # checkpoint's pre-step values is a separate, explicitly opt-in action.
    if digest(args.candidate) != original_digest:
        raise AssertionError('source candidate checkpoint changed')
    if args.output:
        from pathlib import Path
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--output', default='')
    run(p.parse_args())
