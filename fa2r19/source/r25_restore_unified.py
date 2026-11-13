#!/usr/bin/env python3
"""Restore FA2 R25 from ONE candidate + original R19 immutable release.
No R24 candidate, separate branch, or reset optimizer is required.
"""
import argparse,hashlib,json,pathlib,torch
from r25_unified_cpu import initialize_compatible
from r25_episodic_cache import AdaptiveNGramMixer

def restore(root,checkpoint=None,threads=2):
    root=pathlib.Path(root).resolve();torch.set_num_threads(threads)
    p=root/(checkpoint or 'reports/fa2_r25_unified_candidate.pth')
    data=torch.load(p,map_location='cpu',weights_only=False)
    if data.get('format')!='FA2_R25_UNIFIED_CANDIDATE_V1' or data.get('accepted_model_promoted'):
        raise ValueError('not an unpromoted FA2 R25 candidate')
    unified,optimizer,comp,r6,meta=initialize_compatible(root)
    if data['parent']['parent_sha256']!=meta['parent_sha256']:
        raise ValueError('wrong accepted ancestor')
    if data['parent']['r21_sha256']!=meta['r21_sha256'] or data['parent']['r22_sha256']!=meta['r22_sha256']:
        raise ValueError('wrong R21/R22 optimizer initializer')
    unified.add_module('r25',AdaptiveNGramMixer())
    optimizer.add_param_group({'params':list(unified.r25.parameters()),'lr':.004,'weight_decay':1e-4})
    unified.load_state_dict(data['selected']['model'],strict=True)
    optimizer.load_state_dict(data['selected']['optimizer'])
    if len(optimizer.param_groups)!=10 or len(optimizer.state)!=130:
        raise ValueError('optimizer group or state count changed')
    if len(data['selected']['carry'])!=12:raise ValueError('missing train continuation states')
    if data['selected'].get('r25_step')!=120:raise ValueError('unexpected selected continuation')
    return unified,optimizer,comp,r6,meta,data

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.')
    args=p.parse_args();model,opt,comp,r6,meta,s=restore(args.root)
    print(json.dumps({'status':'PASS','r25_step':s['selected']['r25_step'],
      'inherited_step':s['selected']['step'],'optimizer_groups':len(opt.param_groups),
      'optimizer_states':len(opt.state),'streams':len(s['selected']['carry']),
      'inherited_ancestor_sha256':meta['parent_sha256'],
      'r25_parameter_tensors':len(list(model.r25.parameters()))}))
