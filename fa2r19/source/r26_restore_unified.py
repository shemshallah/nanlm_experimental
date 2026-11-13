#!/usr/bin/env python3
"""Cold restore ONE R26 model from immutable PAL R19 and single R26 release ZIP.
There is no separate R24 or R25 model binary requirement for recovery.
"""
import argparse,hashlib,json,pathlib,torch
from r25_unified_cpu import initialize_compatible
from r25_episodic_cache import AdaptiveNGramMixer
from r26_span_memory import AdaptiveSpanMixer

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(4<<20),b''):h.update(part)
    return h.hexdigest()

def restore(root,checkpoint=None,threads=2):
    root=pathlib.Path(root).resolve()
    torch.set_num_threads(threads)
    p=root/(checkpoint or 'reports/fa2_r26_unified_candidate.pth')
    data=torch.load(p,map_location='cpu',weights_only=False)
    if data.get('format')!='FA2_R26_UNIFIED_CANDIDATE_V1' or data.get('accepted_model_promoted'):
        raise ValueError('wrong R26 checkpoint format or promotion claim')
    unified,opt,comp,r6,meta=initialize_compatible(root)
    if meta['parent_sha256']!=data['r18_parent_sha256']:
        raise ValueError('R18 parent SHA changed')
    for a,b in (('parent_sha256','parent_sha256'),('r21_sha256','r21_sha256'),('r22_sha256','r22_sha256')):
        if data['parent'][a]!=meta[b]:raise ValueError('historical initializer changed: '+a)
    unified.add_module('r25',AdaptiveNGramMixer())
    opt.add_param_group({'params':list(unified.r25.parameters()),'lr':.004,'weight_decay':1e-4})
    unified.add_module('r26',AdaptiveSpanMixer())
    opt.add_param_group({'params':list(unified.r26.parameters()),'lr':.012,'weight_decay':1e-4})
    unified.load_state_dict(data['selected']['model'],strict=True)
    opt.load_state_dict(data['selected']['optimizer'])
    assert len(opt.param_groups)==11, 'optimizer groups not retained'
    assert len(opt.state)>=130, 'inherited moments missing'
    assert len(data['selected']['carry'])==12, 'document states missing'
    assert data['selected']['r25_step']==120, 'not direct continuation of R25'
    return unified,opt,comp,r6,meta,data

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.')
    a=p.parse_args();m,o,comp,r6,meta,data=restore(a.root)
    print(json.dumps({'status':'PASS','r26_step':data['selected']['r26_step'],
        'r25_step':data['selected']['r25_step'],'r18_sha256':meta['parent_sha256'],
        'optimizer_groups':len(o.param_groups),'optimizer_states':len(o.state),
        'document_streams':len(data['selected']['carry']),
        'new_trainable_tensors':len(list(m.r26.parameters()))}))
