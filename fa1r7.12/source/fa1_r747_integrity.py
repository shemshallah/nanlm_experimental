#!/usr/bin/env python3
"""Verify R7.47 inherited tensors AND all 26 inherited Adam groups/moments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fa1_r727_checkpoint import digest
from fa1_r747_head_address_copy import ROUND


def same(a,b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and torch.equal(a,b)
    if type(a) is not type(b):
        return False
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def run(a):
    if Path(a.output).exists():
        raise FileExistsError('R7.47 integrity report exists')
    ancestor=torch.load(a.parent,map_location='cpu',weights_only=False)
    successor=torch.load(a.candidate,map_location='cpu',weights_only=False)
    if (ancestor.get('architecture_round')!='FA1_R7_45_GEOMETRIC_VALUE_READ_V1' or
            successor.get('architecture_round')!=ROUND or
            successor.get('promoted') is not False or
            successor['source_checkpoint_sha256']!=digest(a.parent)):
        raise ValueError('unrelated or incorrectly promoted checkpoint')
    old_weights=ancestor['model']
    new_weights=successor['model']
    if not old_weights.keys()<=new_weights.keys():
        raise AssertionError('inherited model tensors missing')
    for name,value in old_weights.items():
        if not same(value,new_weights[name]):
            raise AssertionError('inherited tensor changed: '+name)
    old_opt=ancestor['optimizer']
    new_opt=successor['optimizer']
    if (len(old_opt['groups'])!=26 or len(new_opt['groups'])!=27 or
            not same(old_opt['groups'],new_opt['groups'][:26]) or
            not same(old_opt['legacy'],new_opt['legacy']) or
            len(new_opt['adamw']['param_groups'])!=len(old_opt['adamw']['param_groups'])+1):
        raise AssertionError('inherited optimizer topology changed')
    for old_group,new_group in zip(old_opt['adamw']['param_groups'],
                                   new_opt['adamw']['param_groups'][:25]):
        if not same(old_group,new_group):
            raise AssertionError('an inherited AdamW parameter group changed')
    for old_id,state in old_opt['adamw']['state'].items():
        if not same(state,new_opt['adamw']['state'].get(old_id)):
            raise AssertionError('an inherited AdamW moment changed')
    result={'round':ROUND,'parent_sha256':digest(a.parent),
            'candidate_sha256':digest(a.candidate),
            'inherited_model_tensors_bit_identical':len(old_weights),
            'new_model_state_tensors':len(new_weights)-len(old_weights),
            'inherited_optimizer_groups_bit_identical':26,
            'inherited_adamw_states_bit_identical':len(old_opt['adamw']['state']),
            'inherited_legacy_optimizer_state_identical':True,
            'one_optimizer_groups':27,
            'candidate_promoted':False,'independent_test_opened':False}
    out=Path(a.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('R747 INTEGRITY',json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--parent',required=True)
    p.add_argument('--candidate',required=True)
    p.add_argument('--output',required=True)
    run(p.parse_args())
