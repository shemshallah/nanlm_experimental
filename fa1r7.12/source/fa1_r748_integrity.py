#!/usr/bin/env python3
"""Bitwise R7.47-to-R7.48 checkpoint/optimizer continuity certificate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fa1_r727_checkpoint import digest
from fa1_r747_head_address_copy import ROUND as ANCESTOR
from fa1_r748_mirror_pair_values import ROUND as SUCCESSOR
from fa1_r747_integrity import same


def run(a):
    if Path(a.output).exists():
        raise FileExistsError('integrity report already exists')
    parent = torch.load(a.parent, map_location='cpu', weights_only=False)
    candidate = torch.load(a.candidate, map_location='cpu', weights_only=False)
    immediate_parent = a.immediate_parent or a.parent
    if (parent['architecture_round'] != ANCESTOR or
            candidate['architecture_round'] != SUCCESSOR or
            candidate['promoted'] is not False or
            candidate['source_checkpoint_sha256'] != digest(immediate_parent)):
        raise AssertionError('checkpoint ancestry or promotion status invalid')
    if not parent['model'].keys() <= candidate['model'].keys():
        raise AssertionError('an inherited R7.47 model tensor is absent')
    for name, old in parent['model'].items():
        if not same(old, candidate['model'][name]):
            raise AssertionError('inherited R7.47 tensor changed: ' + name)
    aopt = parent['optimizer']
    bopt = candidate['optimizer']
    if (len(aopt['groups']) != 27 or len(bopt['groups']) != 28 or
            not same(aopt['groups'], bopt['groups'][:27]) or
            not same(aopt['legacy'], bopt['legacy']) or
            len(bopt['adamw']['param_groups']) != len(aopt['adamw']['param_groups']) + 1):
        raise AssertionError('inherited unified optimizer topology changed')
    for old_group, new_group in zip(aopt['adamw']['param_groups'],
                                    bopt['adamw']['param_groups'][:26]):
        if not same(old_group, new_group):
            raise AssertionError('inherited Adam parameter group changed')
    for pid, state in aopt['adamw']['state'].items():
        if not same(state, bopt['adamw']['state'].get(pid)):
            raise AssertionError('an inherited Adam moment changed')
    record = {'round': SUCCESSOR, 'parent_sha256': digest(a.parent),
              'immediate_source_sha256': digest(immediate_parent),
              'candidate_sha256': digest(a.candidate),
              'inherited_model_tensors_bit_identical': len(parent['model']),
              'new_model_state_tensors': len(candidate['model']) - len(parent['model']),
              'inherited_optimizer_groups_bit_identical': 27,
              'inherited_adamw_states_bit_identical': len(aopt['adamw']['state']),
              'inherited_legacy_optimizer_identical': True,
              'optimizer_count': 1, 'optimizer_groups': 28,
              'promoted': False, 'independent_test_opened': False}
    p = Path(a.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    print('R748 INTEGRITY', json.dumps(record), flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True)
    parser.add_argument('--immediate-parent')
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
