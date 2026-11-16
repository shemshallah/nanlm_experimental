#!/usr/bin/env python3
"""Bitwise certification of every inherited R7.49 tensor and Adam moment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fa1_r727_checkpoint import digest
from fa1_r747_integrity import same
from fa1_r749_address_reliability import ROUND as R749
from fa1_r750_calibrated_inhibition import ROUND


def run(a):
    if Path(a.output).exists():
        raise FileExistsError('integrity output already exists')
    p=torch.load(a.parent,map_location='cpu',weights_only=False)
    q=torch.load(a.candidate,map_location='cpu',weights_only=False)
    if (p['architecture_round']!=R749 or q['architecture_round']!=ROUND or
            q['source_checkpoint_sha256']!=digest(a.parent) or q['promoted'] is not False):
        raise AssertionError('R7.50 has invalid one-checkpoint ancestry')
    for name, tensor in p['model'].items():
        if name not in q['model'] or not same(tensor,q['model'][name]):
            raise AssertionError('R7.49 inherited tensor mutated: '+name)
    old_opt,new_opt=p['optimizer'],q['optimizer']
    if (len(old_opt['groups'])!=29 or len(new_opt['groups'])!=30 or
            not same(old_opt['groups'],new_opt['groups'][:29]) or
            not same(old_opt['legacy'],new_opt['legacy']) or
            len(new_opt['adamw']['param_groups'])!=len(old_opt['adamw']['param_groups'])+1):
        raise AssertionError('unified optimizer history was replaced')
    for g0,g1 in zip(old_opt['adamw']['param_groups'],new_opt['adamw']['param_groups'][:-1]):
        if not same(g0,g1):
            raise AssertionError('inherited group config mutated')
    for pid,moments in old_opt['adamw']['state'].items():
        if not same(moments,new_opt['adamw']['state'].get(pid)):
            raise AssertionError('inherited Adam moment altered')
    report={'round':ROUND,'source_sha256':digest(a.parent),
            'candidate_sha256':digest(a.candidate),
            'inherited_tensors_bit_identical':len(p['model']),
            'new_model_tensors':len(q['model'])-len(p['model']),
            'inherited_optimizer_groups_bit_identical':29,
            'inherited_adamw_states_bit_identical':len(old_opt['adamw']['state']),
            'optimizer_count':1,'optimizer_groups':30,
            'promoted':False,'independent_test_opened':False}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print('R750 INTEGRITY',json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--parent',required=True)
    p.add_argument('--candidate',required=True)
    p.add_argument('--output',required=True)
    run(p.parse_args())
