#!/usr/bin/env python3
"""Verify all pre-Diamond weights/moments survive R7.29 continuation exactly."""
import argparse
import json
from pathlib import Path

import torch

from fa1_r727_optimizer import _deep_equal
from fa1_r727_checkpoint import digest,load_unified
from fa1_r729_contextual_geometry import PARAMETER,ROUND


def adam_by_name(blob):
    optimizer=blob['optimizer']
    result={}
    # First legacy accumulator group is separate; the AdamW indices are local
    # to its own backend and include every historical non-legacy group.
    for group,names in zip(optimizer['adamw']['param_groups'],optimizer['groups'][1:]):
        for idx,name in zip(group['params'],names['names']):
            result[name]=optimizer['adamw']['state'].get(idx)
    return result


def run(a):
    p=torch.load(a.parent,map_location='cpu',weights_only=False)
    c=torch.load(a.candidate,map_location='cpu',weights_only=False)
    if c.get('architecture_round')!=ROUND or c.get('promoted') is not False:
        raise AssertionError('not a protected R7.29 candidate')
    before=p['model'];after=c['model']
    unchanged=[k for k in before if not k.startswith('diamond.')]
    for name in unchanged:
        if not torch.equal(before[name],after[name]):
            raise AssertionError('inherited model drift: '+name)
    if set(after)-set(before)!={PARAMETER}:
        raise AssertionError('unexpected new model tensor')
    po,co=p['optimizer'],c['optimizer']
    if po['groups']!=co['groups'][:-1] or co['groups'][-1]['names']!=[PARAMETER]:
        raise AssertionError('inherited optimizer group/order changed')
    if not _deep_equal(po['legacy'],co['legacy']):
        raise AssertionError('legacy accumulator drift')
    original=adam_by_name(p);candidate=adam_by_name(c)
    if candidate.get(PARAMETER) is None:
        raise AssertionError('new contextual read gate has no saved Adam moments')
    checked=0
    for name,moment in original.items():
        if not name.startswith('diamond.'):
            if not _deep_equal(moment,candidate[name]):
                raise AssertionError('inherited Adam moment drift: '+name)
            checked+=1
    model,opt,_,_,meta=load_unified(a.candidate,a.root,a.threads)
    if len(opt.param_groups)!=14 or PARAMETER not in dict(model.named_parameters()):
        raise AssertionError('R7.29 checkpoint did not round-trip')
    report={'format':'FA1_R7_29_FULL_CHECKPOINT_INTEGRITY_V1',
            'parent_sha256':digest(a.parent),'candidate_sha256':digest(a.candidate),
            'non_diamond_weight_tensors_verified':len(unchanged),
            'pre_diamond_adam_state_entries_verified':checked,
            'legacy_accumulator_bitwise_preserved':True,
            'old_optimizer_group_membership_preserved':True,
            'new_contextual_gate_optimizer_state_saved':True,
            'new_registered_parameters':[PARAMETER],
            'single_optimizer_groups':len(opt.param_groups),
            'full_checkpoint_reload':True,
            'accepted_parent_untouched':True,'candidate_promoted':False}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--parent',required=True)
    p.add_argument('--candidate',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--threads',type=int,default=2)
    run(p.parse_args())
