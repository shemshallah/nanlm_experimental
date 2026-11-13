#!/usr/bin/env python3
"""Cold restore one unified FA2 R27 checkpoint from original PAL R19 assets.

No R24/R25/R26 candidate binary is necessary after the R27 release overlay.
The accepted R18 ancestor in R19 remains immutable.
"""
import argparse,hashlib,json,pathlib,torch
from r25_unified_cpu import initialize_compatible
from r25_episodic_cache import AdaptiveNGramMixer
from r26_span_memory import AdaptiveSpanMixer
from r27_trust import CounterfactualTrustGate

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for part in iter(lambda:f.read(4<<20),b''):h.update(part)
 return h.hexdigest()

def restore(root,checkpoint='reports/fa2_r27_unified_candidate.pth',threads=2):
 root=pathlib.Path(root).resolve();torch.set_num_threads(threads)
 data=torch.load(root/checkpoint,map_location='cpu',weights_only=False)
 if data.get('format')!='FA2_R27_UNIFIED_CANDIDATE_V1' or data.get('accepted_model_promoted'):
  raise ValueError('wrong R27 checkpoint format or promotion status')
 model,opt,comp,r6,meta=initialize_compatible(root)
 if meta['parent_sha256']!=data['r18_parent_sha256']:
  raise ValueError('R18 original ancestor changed')
 for k in ('parent_sha256','r21_sha256','r22_sha256'):
  if data['parent'][k]!=meta[k]:raise ValueError('legacy initializer changed: '+k)
 model.add_module('r25',AdaptiveNGramMixer())
 opt.add_param_group({'params':list(model.r25.parameters()),'lr':.004,'weight_decay':1e-4})
 model.add_module('r26',AdaptiveSpanMixer())
 opt.add_param_group({'params':list(model.r26.parameters()),'lr':.012,'weight_decay':1e-4})
 model.add_module('r27',CounterfactualTrustGate())
 opt.add_param_group({'params':list(model.r27.parameters()),'lr':.004,'weight_decay':1e-4})
 model.load_state_dict(data['selected']['model'],strict=True)
 opt.load_state_dict(data['selected']['optimizer'])
 assert len(opt.param_groups)==12 and len(opt.state)==144,(len(opt.param_groups),len(opt.state))
 assert len(data['selected']['carry'])==12
 assert data['selected']['r25_step']==120
 assert data['selected']['r26_step']==90
 assert data['selected']['r27_step']>0
 return model,opt,comp,r6,meta,data

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.')
 args=p.parse_args();model,opt,comp,r6,meta,data=restore(args.root)
 print('R27_COLD_RESTORE_PASS',json.dumps({'r18_sha256':meta['parent_sha256'],
  'r25_step':data['selected']['r25_step'],'r26_step':data['selected']['r26_step'],
  'r27_step':data['selected']['r27_step'],'optimizer_groups':len(opt.param_groups),
  'optimizer_states':len(opt.state),'document_streams':len(data['selected']['carry'])}))