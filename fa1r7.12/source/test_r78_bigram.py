#!/usr/bin/env python3
"""Non-destructive causal event writer compatibility and sensitivity tests.

Only needs the accepted R7.7 checkpoint; unpromoted R7.8 weights are not needed.
"""
import json,pathlib,torch,numpy as np
from r78_ordered import restore as r78_restore
from r78_bigram import upgrade_pair
from r74_stream import OLDER,OUTER_CONTEXT,BLOCK
from r7_core import Corpus
from train_r7 import batch_data

def run(root='.'):
 torch.set_num_threads(2)
 root=pathlib.Path(root)
 model,opts,base,comp,prior=r78_restore(root,root/'checkpoint/pal_active.pt')
 old_state={k:v.detach().clone() for k,v in model.state_dict().items()}
 test=Corpus(root,'test');positions=np.array([8192,20000])
 _,ctx,pc,lens,y,_,_=batch_data(base,test,comp,positions)
 h=torch.as_tensor(test.get(positions,OUTER_CONTEXT)[0][:,:OLDER].copy())
 ex=torch.as_tensor(test.data[positions].copy());lo=torch.as_tensor(test.long[positions].copy())
 model.eval()
 with torch.no_grad():
  before=model.initial(h)
  _,before_loss,_=model(ex,lo,ctx,pc,lens,before,h,y)
 upgrade_pair(model,opts)
 checks={}
 assert len(opts)==5 and len(opts[3].param_groups)==3
 assert all(torch.equal(old_state[k],v) for k,v in model.state_dict().items() if k in old_state)
 checks['old_parameters_identical']='PASS'
 with torch.no_grad():
  after=model.initial(h)
  _,after_loss,_=model(ex,lo,ctx,pc,lens,after,h,y)
  assert torch.equal(before,after) and torch.equal(before_loss,after_loss)
 checks['zero_gate_exact_migration']='PASS'
 assert all(o.state for o in opts)
 assert id(model.r74.outer.pair_gain) in {id(p) for g in opts[3].param_groups for p in g['params']}
 checks['five_optimizer_slots_original_states_live']='PASS'
 with torch.no_grad():
  reversed_h=h.reshape(-1,OLDER//BLOCK,BLOCK).flip(-1).reshape(-1,OLDER)
  model.r74.outer.order_gain.zero_()
  zero_before=model.initial(h);zero_reverse=model.initial(reversed_h)
  noise=float((zero_before-zero_reverse).abs().max())
  model.r74.outer.pair_gain.fill_(.2)
  x=model.initial(h);z=model.initial(reversed_h)
  delta=float((x-z).abs().max())
  assert delta>max(1e-6,2*noise),(noise,delta)
  model.r74.outer.pair_gain.zero_()
 checks['causal_adjacent_order_distinguished']='PASS'
 model.train();state=model.initial(h)
 _,loss,_=model(ex,lo,ctx,pc,lens,state,h,y)
 for opt in opts:opt.zero_grad(set_to_none=True)
 loss.mean().backward()
 assert model.r74.outer.pair_gain.grad is not None
 assert bool(torch.isfinite(model.r74.outer.pair_gain.grad))
 checks['pair_event_gate_backprop']='PASS'
 return {'checks':checks,'passed':len(checks),'permutation_difference_disabled':noise,
         'permutation_difference_enabled':delta,'source_step':prior['joint_steps']}
if __name__=='__main__':print(json.dumps(run('.'),indent=2))
