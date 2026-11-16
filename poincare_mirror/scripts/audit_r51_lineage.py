#!/usr/bin/env python3
"""Audit R51: same trained FA2 model, original frozen trunk and Adam moments."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from poincare_frontier import Config, MirrorLM, is_attached_geometry_parameter


def audit(before, after):
    issues = []
    # Newer source can serialize an explicit dataclass default absent from an
    # older checkpoint. Only *effective* architectural changes are forbidden.
    same_cfg = Config(**before['config']) == Config(**after['config'])
    if not same_cfg:
        issues.append('Architecture or learned hyperparameters changed')
    if before['data'] != after['data']:
        issues.append('Corpus manifest changed')
    if after['step'] <= before['step']:
        issues.append('Step did not advance')
    if before['model'].keys() != after['model'].keys():
        issues.append('Model parameter keys changed')
    observed = {entry.get('kind') for entry in after.get('growth_history',[])}
    if ('geometry_only_warmup' not in observed or
            not observed.intersection({'r51_train_only_address_specificity',
                                       'r52_content_byte_address_specificity'})):
        issues.append('R51/R52 TRAIN-only or geometry-only provenance missing')
    names = list(dict(MirrorLM(Config(**before['config'])).named_parameters()))
    old_groups, new_groups = before['optimizer']['param_groups'], after['optimizer']['param_groups']
    if len(old_groups)!=1 or len(new_groups)!=1:
        issues.append('Unified optimizer group was split/replaced')
        return {'lineage_pass':False, 'issues':issues}
    ids_before, ids_after = old_groups[0]['params'], new_groups[0]['params']
    if len(ids_before)!=len(names) or len(ids_after)!=len(names):
        issues.append('Adam parameter mapping length changed')
        return {'lineage_pass':False,'issues':issues}
    os0, os1 = before['optimizer']['state'], after['optimizer']['state']
    if len(os0)!=len(os1):
        issues.append('Adam state count changed')
    frozen=frozen_moments=changed=advanced=retained=0
    for name, id0, id1 in zip(names,ids_before,ids_after):
        if name not in before['model'] or name not in after['model']:
            issues.append('Missing learned tensor: '+name)
            continue
        if id0 not in os0 or id1 not in os1:
            issues.append('Missing Adam state: '+name)
            continue
        retained+=1
        attached=is_attached_geometry_parameter(name)
        a,b=before['model'][name],after['model'][name]
        s0,s1=os0[id0],os1[id1]
        if not attached:
            frozen+=1
            if not torch.equal(a,b): issues.append('Inherited backbone changed: '+name)
            if any(not torch.equal(s0[k],s1[k]) for k in ('step','exp_avg','exp_avg_sq')):
                issues.append('Frozen backbone Adam changed: '+name)
            else:
                frozen_moments+=1
        else:
            changed += int(not torch.equal(a,b))
            if s1['step'] < s0['step']:
                issues.append('Geometry Adam step rewound: '+name)
            elif s1['step'] > s0['step']:
                advanced+=1
        if not torch.isfinite(b).all() or any(not torch.isfinite(s1[k]).all()
                                               for k in ('exp_avg','exp_avg_sq')):
            issues.append('Nonfinite trained parameter or moment: '+name)
    if retained != len(os0): issues.append('Not all inherited Adam moments retained')
    if not changed or not advanced: issues.append('No geometric learning observed')
    return {'lineage_pass':not issues,'issues':issues,
            'parent_step':before['step'],'candidate_step':after['step'],
            'inherited_adam_states':len(os0),'retained_adam_states':retained,
            'frozen_backbone_parameters':frozen,'bitwise_frozen_adam_states':frozen_moments,
            'geometric_parameters_changed':changed,'geometric_adam_steps_advanced':advanced,
            'same_model_architecture':same_cfg,
            'same_corpus_manifest':before['data']==after['data']}


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.before.resolve()==args.after.resolve():p.error('Parent must differ from candidate')
    a=torch.load(args.before,map_location='cpu',weights_only=False)
    b=torch.load(args.after,map_location='cpu',weights_only=False)
    report=audit(a,b)
    report['parent_sha256']=sha(args.before)
    report['candidate_sha256']=sha(args.after)
    encoded=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(encoded,encoding='utf-8')
    print(encoded,end='')
    if not report['lineage_pass']:raise SystemExit(1)


if __name__=='__main__':
    main()
