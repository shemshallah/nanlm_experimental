#!/usr/bin/env python3
"""R7.48 -> R7.49 inherited backbone and Adam-state continuity audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fa1_r727_checkpoint import digest
from fa1_r747_integrity import same
from fa1_r748_mirror_pair_values import ROUND as R748
from fa1_r749_address_reliability import ROUND


def run(a):
    if Path(a.output).exists():
        raise FileExistsError('integrity report already exists')
    ancestor = torch.load(a.parent, map_location='cpu', weights_only=False)
    candidate = torch.load(a.candidate, map_location='cpu', weights_only=False)
    if (ancestor['architecture_round'] != R748 or
            candidate['architecture_round'] != ROUND or
            candidate['promoted'] is not False or
            candidate['source_checkpoint_sha256'] != digest(a.immediate_parent or a.parent)):
        raise AssertionError('incorrect R7.49 ancestry')
    for name, value in ancestor['model'].items():
        if name not in candidate['model']:
            raise AssertionError('removed inherited parameter: ' + name)
        if not name.startswith('word_backoff.mirror_') and not same(value, candidate['model'][name]):
            raise AssertionError('changed inherited FA1 backbone: ' + name)
    p = ancestor['optimizer']
    q = candidate['optimizer']
    if (len(p['groups']) != 28 or len(q['groups']) != 29 or
            not same(p['groups'][:27], q['groups'][:27]) or
            not same(p['legacy'], q['legacy'])):
        raise AssertionError('original 27-group optimizer topology changed')
    # Group 28 is deliberately trained; original groups 1-27 are frozen.
    # Group 29 is new and carries its own Adam moment history.
    previous_adam = p['adamw']['param_groups']
    successor_adam = q['adamw']['param_groups']
    if len(successor_adam) != len(previous_adam) + 1:
        raise AssertionError('R7.49 did not append exactly one Adam group')
    for g0, g1 in zip(previous_adam[:-1], successor_adam[:-2]):
        if not same(g0, g1):
            raise AssertionError('frozen inherited Adam group mutated')
    trainable28 = set(previous_adam[-1]['params'])
    preserved_states = 0
    for pid, state in p['adamw']['state'].items():
        if pid not in trainable28:
            if not same(state, q['adamw']['state'].get(pid)):
                raise AssertionError('inherited Adam moments changed')
            preserved_states += 1
    record = {'round': ROUND, 'parent_sha256': digest(a.parent),
              'immediate_parent_sha256': digest(a.immediate_parent or a.parent),
              'candidate_sha256': digest(a.candidate),
              'frozen_backbone_tensors_verified': sum(
                  not n.startswith('word_backoff.mirror_') for n in ancestor['model']),
              'new_model_tensors': len(candidate['model']) - len(ancestor['model']),
              'inherited_frozen_optimizer_groups': 27,
              'bitwise_original_adamw_states': preserved_states,
              'group28_pair_bank_resumed_and_trained': True,
              'group29_reliability_adam_appended': True,
              'optimizer_count': 1, 'optimizer_groups': 29,
              'promoted': False, 'independent_test_opened': False}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(record, sort_keys=True, indent=2)+'\n')
    print('R749 INTEGRITY', json.dumps(record), flush=True)
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--parent', required=True)
    p.add_argument('--immediate-parent')
    p.add_argument('--candidate', required=True)
    p.add_argument('--output', required=True)
    run(p.parse_args())
