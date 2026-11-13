#!/usr/bin/env python3
"""Convert selected R27 adapter into one complete, fully recoverable checkpoint.

Preserves inherited R18-R26 tensors, Adam moments, document-stream snapshots.
"""
import copy,hashlib,json,pathlib,os,torch
from r26_restore_unified import restore,sha
from r27_trust import CounterfactualTrustGate

def exact_equal(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and a.shape==b.shape and torch.equal(a,b)
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(exact_equal(v,b[k]) for k,v in a.items())
 if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(exact_equal(x,y) for x,y in zip(a,b))
 return a==b

def main(root):
 root=pathlib.Path(root).resolve();torch.set_num_threads(2)
 u,opt,comp,r6,meta,old=restore(root)
 overlay=torch.load(root/'reports/r27_local_overlay.pt',map_location='cpu',weights_only=False)
 parent=root/'reports/fa2_r26_unified_candidate.pth'
 if overlay['parent_r26_sha256']!=sha(parent):raise ValueError('R26 checkpoint changed')
 if overlay['selected_step']!=40:raise ValueError('unexpected R27 step')
 base=old['selected']
 old_params={k:v.clone() for k,v in u.state_dict().items()}
 old_opt=copy.deepcopy(opt.state_dict())
 old_carry=copy.deepcopy(base['carry'])
 u.add_module('r27',CounterfactualTrustGate());u.r27.load_state_dict(overlay['r27_state'])
 opt.add_param_group({'params':list(u.r27.parameters()),'lr':.004,'weight_decay':1e-4})
 state=opt.state_dict()
 assert state['param_groups'][-1]['params']==overlay['r27_group']['params']
 state['param_groups'][-1]=overlay['r27_group']
 for k,v in overlay['r27_opt'].items():
  if k in state['state']:raise ValueError('R27 state ID collides with R26')
  state['state'][k]=v
 opt.load_state_dict(state)
 for k,v in old_params.items():
  if not torch.equal(u.state_dict()[k],v):raise ValueError('inherited R26 tensor changed: '+k)
 now=opt.state_dict()
 if now['param_groups'][:11]!=old_opt['param_groups']:
  raise ValueError('inherited optimizer settings changed')
 for k,v in old_opt['state'].items():
  if not exact_equal(v,now['state'][k]):raise ValueError('inherited Adam state changed: '+str(k))
 if not exact_equal(base['carry'],old_carry):raise ValueError('document stream changed')
 if len(now['param_groups'])!=12 or len(now['state'])!=144:raise ValueError('expected 12 groups / 144 Adam states')
 selected=dict(base)
 selected.update(model=u.state_dict(),optimizer=now,r27_step=overlay['selected_step'])
 dev=json.loads((root/'reports/r27_realbook_local_pilot.json').read_text())
 reserve=json.loads((root/'reports/r27_reserve_long_132k.json').read_text())
 checkpoint={
  'format':'FA2_R27_UNIFIED_CANDIDATE_V1',
  'accepted_model_promoted':False,
  'parent':old['parent'],
  'r18_parent_sha256':old['r18_parent_sha256'],
  'parent_r26_sha256':sha(parent),
  'selected':selected,
  'selection_dev':dev['selected'],
  'reserve_dev':reserve['results'],
  'original_R26_immutable':True,
  'original_Adam_states_preserved':len(old_opt['state']),
  'original_tensors_preserved':len(old_params),
  'original_stream_count':len(old_carry)
 }
 out=root/'reports/fa2_r27_unified_candidate.pth';temp=out.with_suffix('.temp.pth')
 torch.save(checkpoint,temp);os.replace(temp,out)
 report={'format':'FA2_R27_FULL_INHERITANCE_V1','r27_step':overlay['selected_step'],
  'inherited_model_tensors_exact':len(old_params),
  'inherited_adam_states_exact':len(old_opt['state']),
  'inherited_document_streams_exact':len(old_carry),
  'optimizer_groups':len(now['param_groups']),'optimizer_states':len(now['state']),
  'r18_parent_sha256':old['r18_parent_sha256'],
  'r26_parent_sha256':sha(parent),'r27_checkpoint_sha256':sha(out),
  'selected_gain_bpb':dev['selected']['gain_bpb'],
  'reserve_gain_bpb':reserve['results']['gain_bpb'],
  'accepted_model_promoted':False}
 (root/'reports/r27_inheritance_audit.json').write_text(json.dumps(report,indent=2)+'\n')
 print('R27_FULL_CHECKPOINT_VERIFIED',json.dumps(report),flush=True)

if __name__=='__main__':
 main('.')