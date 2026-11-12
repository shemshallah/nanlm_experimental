#!/usr/bin/env python3
"""R7.8 deterministic migration, temporal sensitivity, continuity and gradient checks."""
import json,pathlib,torch,numpy as np
from r77_semantic import restore as restore77
from r78_ordered import restore as restore78
from r7_core import Corpus
from r74_stream import OUTER_CONTEXT,OLDER,BLOCK
from train_r7 import batch_data

def run(root):
 torch.set_num_threads(2)
 root=pathlib.Path(root)
 old,o0,base,comp,d0=restore77(root,root/'checkpoint/pal_active.pt',torch.device('cpu'))
 new,o1,_,_,d1=restore78(root,root/'checkpoint/pal_active.pt','cpu')
 result={};
 assert len(o0)==len(o1)==5
 for a,b in zip(o0,o1):
  assert len(a.state)==len(b.state)
  for a_group,b_group in zip(a.param_groups,b.param_groups):
   for p,q in zip(a_group["params"],b_group["params"]):
    if p not in a.state:continue
    assert p in a.state and q in b.state
    for key,v in a.state[p].items():
     x=b.state[q][key]
     assert (torch.equal(v,x) if isinstance(v,torch.Tensor) else v==x),key
 result['all_existing_adam_moments_preserved']='PASS'
 original={k:v for k,v in new.state_dict().items() if not k.startswith('r74.outer.order_')}
 assert all(torch.equal(original[k],v) for k,v in old.state_dict().items())
 result['all_old_weights_bit_identical']='PASS'
 assert len(o1[3].param_groups)==len(o0[3].param_groups)+1
 assert set(id(p) for g in o1[3].param_groups for p in g['params'])==set(id(p) for p in new.r74.outer.parameters())|{id(new.r74.outer_gain)}
 result['one_live_mirror_memory_and_same_optimizer']='PASS'
 c=Corpus(root,'test');pos=np.array([8192,9216,16384])
 _,ctx,pcs,lens,y,_,_=batch_data(base,c,comp,pos)
 ex=torch.as_tensor(c.data[pos].copy());long=torch.as_tensor(c.long[pos].copy())
 hist=torch.as_tensor(c.get(pos,OUTER_CONTEXT)[0][:,:OLDER].copy())
 old.eval();new.eval()
 with torch.no_grad():
  so=old.initial(hist);sn=new.initial(hist)
  assert torch.equal(so,sn), float((so-sn).abs().max())
  _,ol,_=old(ex,long,ctx,pcs,lens,so,hist,y)
  _,nl,_=new(ex,long,ctx,pcs,lens,sn,hist,y)
  assert torch.equal(ol,nl),(ol,nl)
 result['zero_gate_migration_identical_states_and_nll']='PASS'
 swap=hist.reshape(-1,OLDER//BLOCK,BLOCK).flip(-1).reshape(-1,OLDER)
 with torch.no_grad():
  # Before training: new writer is exactly disabled, even on a permutation.
  zero_delta=float((new.initial(hist)-new.initial(swap)).abs().amax())
  assert zero_delta<1e-5,zero_delta
  new.r74.outer.order_gain.fill_(0.2)
  x=new.initial(hist);z=new.initial(swap)
  delta=float((x-z).abs().amax())
  assert delta>max(1e-8,zero_delta*2),(delta,zero_delta)
  new.r74.outer.order_gain.zero_()
 result['within_64_word_order_now_distinguishable']='PASS'
 new.train();state=new.initial(hist)
 _,nll,_=new(ex,long,ctx,pcs,lens,state,hist,y)
 for o in o1:o.zero_grad(set_to_none=True)
 nll.mean().backward()
 g=new.r74.outer.order_gain.grad
 assert g is not None and torch.isfinite(g).all()
 result['order_gate_has_causal_loss_gradient']='PASS'
 return {'checks':result,'passed':len(result),'permutation_delta_zero':zero_delta,'permutation_delta_gain_0p2':delta,'gate_grad':float(g),'step':d0['joint_steps']}
if __name__=='__main__':
 print(json.dumps(run('.'),indent=2))
