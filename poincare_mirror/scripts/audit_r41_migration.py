#!/usr/bin/env python3
"""Read-only R41 function-preserving trust-migration and Adam lineage audit."""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poincare_frontier import Config, MirrorLM
from audit_r40_lineage import sha256


TRUST_NAME = 'reciprocal.context_trust.weight'


def audit(before, after):
    issues = []
    previous = Config(**before['config'])
    current = Config(**after['config'])
    if previous.reciprocal_content_trust or not current.reciprocal_content_trust:
        issues.append('Trust migration must be false to true')
    if replace(previous, reciprocal_content_trust=True) != current:
        issues.append('Unrelated architecture changed')
    if before.get('data') != after.get('data'):
        issues.append('Book corpus or split changed')
    if after['step'] <= before['step']:
        issues.append('No training steps advanced')
    if not any(x.get('kind') == 'causal_reciprocal_content_trust'
               for x in after.get('growth_history', [])):
        issues.append('Trust migration provenance absent')

    original_names = list(dict(MirrorLM(previous).named_parameters()))
    candidate_names = list(dict(MirrorLM(current).named_parameters()))
    if set(candidate_names) - set(original_names) != {TRUST_NAME}:
        issues.append('Unexpected new parameter identity')
    if set(original_names) - set(candidate_names):
        issues.append('Inherited parameter missing')
    if set(after['model']) - set(before['model']) != {TRUST_NAME}:
        issues.append('Unexpected new model tensor identity')
    if set(before['model']) - set(after['model']):
        issues.append('Inherited model tensor missing')

    opt0, opt1 = before['optimizer'], after['optimizer']
    groups0, groups1 = opt0['param_groups'], opt1['param_groups']
    if len(groups0) != 1 or len(groups1) != 1:
        issues.append('Expected single inherited Adam parameter group')
        return {'lineage_pass': False, 'issues': issues}
    ids0, ids1 = groups0[0]['params'], groups1[0]['params']
    if len(ids0) != len(original_names) or len(ids1) != len(candidate_names):
        issues.append('Optimizer parameter/name mapping mismatch')
        return {'lineage_pass': False, 'issues': issues}
    named0, named1 = dict(zip(original_names, ids0)), dict(zip(candidate_names, ids1))
    retained = advanced = 0
    for name, old_id in named0.items():
        if old_id not in opt0['state']:
            continue
        new_id = named1.get(name)
        if new_id not in opt1['state']:
            issues.append(f'Inherited optimizer state lost: {name}')
            continue
        old = opt0['state'][old_id]
        new = opt1['state'][new_id]
        retained += 1
        for key in ('exp_avg', 'exp_avg_sq'):
            if key not in old or key not in new or old[key].shape != new[key].shape:
                issues.append(f'Adam moment changed shape/missing: {name}/{key}')
            elif not torch.isfinite(new[key]).all():
                issues.append(f'Nonfinite inherited Adam moment: {name}/{key}')
        if float(new['step']) < float(old['step']):
            issues.append(f'Adam step reset: {name}')
        elif float(new['step']) > float(old['step']):
            advanced += 1
    trust_id = named1.get(TRUST_NAME)
    trust_trained = (trust_id in opt1['state'] and
        bool(torch.count_nonzero(after['model'][TRUST_NAME]).item()))
    if not trust_trained:
        issues.append('New trust gate did not receive an effective update')
    if retained != len(opt0['state']):
        issues.append('Not all inherited Adam states retained')
    if not advanced:
        issues.append('No inherited Adam counter advanced')
    return {'lineage_pass': not issues, 'issues': issues,
            'parent_step': before['step'], 'candidate_step': after['step'],
            'inherited_adam_states': len(opt0['state']),
            'inherited_adam_states_retained': retained,
            'inherited_adam_states_advanced': advanced,
            'candidate_adam_states': len(opt1['state']),
            'new_gate_trained': trust_trained,
            'split_manifest_sha256': before['data'].get('split_manifest_sha256')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before', type=Path, required=True)
    p.add_argument('--after', type=Path, required=True)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.before.resolve() == args.after.resolve():
        p.error('Expected separate protected parent and unpromoted candidate')
    result = audit(torch.load(args.before, map_location='cpu', weights_only=False),
                   torch.load(args.after, map_location='cpu', weights_only=False))
    result['parent_sha256'] = sha256(args.before)
    result['candidate_sha256'] = sha256(args.after)
    text = json.dumps(result, indent=2) + '\n'
    print(text, end='')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    if not result['lineage_pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
