#!/usr/bin/env python3
"""Audit a FA2 continuation's weights, 194 inherited Adam histories and corpus.

Read-only. Designed for the R40 two-speed candidate and usable on all FA2
one-architecture continuations; never reads the sealed TEST predictions.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(before, after):
    issues = []
    if before.get('data') != after.get('data'):
        issues.append('Training/validation corpus metadata changed')
    if before.get('config') != after.get('config'):
        issues.append('Architecture/config changed during this continuation')
    initial_step, final_step = before['step'], after['step']
    if final_step <= initial_step:
        issues.append('No positive training step advance')
    for field in ('rng_numpy', 'rng_torch', 'rng_python'):
        if field not in before or field not in after:
            issues.append(f'RNG state missing: {field}')

    original_weights, candidate_weights = before['model'], after['model']
    if original_weights.keys() != candidate_weights.keys():
        issues.append('Model tensor names differ')
    shifted = 0
    greatest_relative_shift = 0.
    for name, old in original_weights.items():
        if name not in candidate_weights:
            continue
        new = candidate_weights[name]
        if old.shape != new.shape or old.dtype != new.dtype:
            issues.append(f'Model tensor shape/dtype changed: {name}')
            continue
        if not torch.isfinite(new).all():
            issues.append(f'Non-finite model tensor: {name}')
        if torch.is_floating_point(old):
            difference = (new.float() - old.float()).norm().item()
            base = old.float().norm().item()
            if difference:
                shifted += 1
            greatest_relative_shift = max(greatest_relative_shift,
                                          difference / max(base, 1e-10))

    old_opt, new_opt = before['optimizer'], after['optimizer']
    old_groups, new_groups = old_opt['param_groups'], new_opt['param_groups']
    if len(old_groups) != len(new_groups) or any(
            x['params'] != y['params'] for x, y in zip(old_groups, new_groups)):
        issues.append('Optimizer parameter-group identity changed')
    old_states, new_states = old_opt['state'], new_opt['state']
    if not old_states.keys() <= new_states.keys():
        issues.append('Inherited optimizer state IDs were dropped')
    states_advanced = 0
    for key, old in old_states.items():
        if key not in new_states:
            continue
        new = new_states[key]
        for moment in ('exp_avg', 'exp_avg_sq'):
            if moment not in old or moment not in new:
                issues.append(f'Adam moment missing: {key}/{moment}')
                continue
            a, b = old[moment], new[moment]
            if a.shape != b.shape or a.dtype != b.dtype:
                issues.append(f'Adam moment shape/dtype changed: {key}/{moment}')
            elif not torch.isfinite(b).all():
                issues.append(f'Non-finite Adam moment: {key}/{moment}')
        if 'step' not in old or 'step' not in new:
            issues.append(f'Adam counter missing: {key}')
            continue
        previous = float(old['step'])
        current = float(new['step'])
        if not math.isfinite(current) or current < previous:
            issues.append(f'Adam step counter regressed: {key}')
        if current > previous:
            states_advanced += 1
    if not states_advanced:
        issues.append('No inherited Adam step counter advanced')

    return {
        'lineage_pass': not issues,
        'issues': issues,
        'parent_step': initial_step,
        'candidate_step': final_step,
        'steps_added': final_step - initial_step,
        'inherited_adam_states': len(old_states),
        'candidate_adam_states': len(new_states),
        'inherited_adam_states_advanced': states_advanced,
        'model_tensors_changed': shifted,
        'max_relative_weight_delta': greatest_relative_shift,
        'split_manifest_sha256': before['data'].get('split_manifest_sha256'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.before.resolve() == args.after.resolve():
        parser.error('Parent and candidate files must differ')
    before = torch.load(args.before, map_location='cpu', weights_only=False)
    after = torch.load(args.after, map_location='cpu', weights_only=False)
    report = audit(before, after)
    report['parent_sha256'] = sha256(args.before)
    report['candidate_sha256'] = sha256(args.after)
    text = json.dumps(report, indent=2) + '\n'
    print(text, end='')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    if not report['lineage_pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
