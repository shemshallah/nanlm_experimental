#!/usr/bin/env python3
"""Read-only FA2 R44 audit: frozen trunk weights/moments, one optimizer lineage."""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poincare_frontier import Config, MirrorLM, is_attached_geometry_parameter
from audit_r40_lineage import sha256


def audit(before, after, requires_aux):
    errors = []
    cfg0, cfg1 = Config(**before['config']), Config(**after['config'])
    if cfg0.reciprocal_content_trust or replace(cfg0, reciprocal_content_trust=True) != cfg1:
        errors.append('Unexpected architecture or trust expansion')
    if before['data'] != after['data']:
        errors.append('Corpus changed')
    if after['step'] <= before['step']:
        errors.append('No training update')
    kinds = [r.get('kind') for r in after.get('growth_history', [])]
    if 'geometry_only_warmup' not in kinds:
        errors.append('Geometry-only training history missing')
    has_aux = 'causal_geometric_residual_correction_supervision' in kinds
    if bool(requires_aux) != has_aux:
        errors.append('Residual-objective history mismatch')
    names0 = list(dict(MirrorLM(cfg0).named_parameters()))
    names1 = list(dict(MirrorLM(cfg1).named_parameters()))
    if set(names1)-set(names0) != {'reciprocal.context_trust.weight'}:
        errors.append('New unexpected parameter or missing trust row')
    if set(names0)-set(names1):
        errors.append('Old parameter removed')
    if set(after['model'])-set(before['model']) != {'reciprocal.context_trust.weight'}:
        errors.append('New unexpected model tensor')
    groups0,groups1=before['optimizer']['param_groups'],after['optimizer']['param_groups']
    if len(groups0)!=1 or len(groups1)!=1:
        return {'lineage_pass':False,'issues':errors+['Optimizer group mismatch']}
    ids0,ids1=groups0[0]['params'],groups1[0]['params']
    if len(ids0)!=len(names0) or len(ids1)!=len(names1):
        return {'lineage_pass':False,'issues':errors+['Parameter mapping mismatch']}
    old_states,new_states=before['optimizer']['state'],after['optimizer']['state']
    old_ids,new_ids=dict(zip(names0,ids0)),dict(zip(names1,ids1))
    retained=frozen_params=frozen_states=changed_geometry=advanced_geo=0
    for name in names0:
        prior=before['model'][name]
        current=after['model'][name]
        geometry=is_attached_geometry_parameter(name)
        if not geometry:
            frozen_params+=1
            if not torch.equal(prior,current):
                errors.append('Inherited trunk weights changed: '+name)
        elif not torch.equal(prior,current):
            changed_geometry+=1
        prior_id,new_id=old_ids[name],new_ids[name]
        if prior_id not in old_states:
            continue
        if new_id not in new_states:
            errors.append('Inherited Adam state lost: '+name)
            continue
        retained+=1
        old,new=old_states[prior_id],new_states[new_id]
        if not geometry:
            frozen_states+=1
            if any(not torch.equal(old[key],new[key]) for key in ('step','exp_avg','exp_avg_sq')):
                errors.append('Frozen trunk Adam state changed: '+name)
        else:
            if float(new['step']) < float(old['step']):
                errors.append('Geometry Adam step reset: '+name)
            elif float(new['step']) > float(old['step']):
                advanced_geo+=1
            if any(not torch.isfinite(new[key]).all() for key in ('exp_avg','exp_avg_sq')):
                errors.append('Nonfinite geometry Adam moment: '+name)
    if retained != len(old_states):
        errors.append('Some inherited Adam states dropped')
    if changed_geometry == 0 or advanced_geo == 0:
        errors.append('No geometric learning occurred')
    trust_id=new_ids.get('reciprocal.context_trust.weight')
    if trust_id not in new_states:
        errors.append('New trust state missing')
    return {'lineage_pass':not errors,'issues':errors,
            'parent_step':before['step'],'candidate_step':after['step'],
            'inherited_adam_states':len(old_states),
            'inherited_adam_states_retained':retained,
            'frozen_trunk_parameters_unchanged':frozen_params,
            'frozen_trunk_adam_states_bitwise_unchanged':frozen_states,
            'changed_geometric_parameters':changed_geometry,
            'advanced_geometric_adam_states':advanced_geo,
            'new_trust_state_present':trust_id in new_states,
            'residual_auxiliary_enabled':bool(requires_aux),
            'split_manifest_sha256':before['data'].get('split_manifest_sha256')}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--before',type=Path,required=True)
    ap.add_argument('--after',type=Path,required=True)
    ap.add_argument('--requires-aux',action='store_true')
    ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    if args.before.resolve()==args.after.resolve():
        ap.error('Cannot audit a candidate in place of its protected parent')
    result=audit(torch.load(args.before,map_location='cpu',weights_only=False),
                 torch.load(args.after,map_location='cpu',weights_only=False),args.requires_aux)
    result['parent_sha256']=sha256(args.before)
    result['candidate_sha256']=sha256(args.after)
    encoded=json.dumps(result,indent=2)+'\n'
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(encoded)
    if not result['lineage_pass']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
