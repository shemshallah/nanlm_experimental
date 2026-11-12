#!/usr/bin/env python3
"""R7.7 non-destructive structural, optimizer, causality and ablation checks."""
import argparse,json,pathlib
import torch,numpy as np
from r77_semantic import restore,PALR77
from r76_pair_training import PALR76
from r74_stream import OLDER,OUTER_CONTEXT
from r7_core import Corpus
from train_r7 import batch_data

def run(root):
 root=pathlib.Path(root);torch.set_num_threads(2)
 m,opts,base,comp,prior=restore(root,root/'checkpoint/pal_active.pt',torch.device('cpu'))
 checks={};m.eval()
 assert isinstance(m,PALR77) and len(opts)==5
 assert len(opts[4].param_groups)==3 and all(o.state for o in opts)
 assert all(prior['joint_steps']>=14000 for _ in [0]);checks['one_checkpoint_five_optimizer_state_histories']='PASS'
 names=dict(m.retrieval.named_parameters())
 target_by_group=[{id(p) for p in group['params']} for group in opts[4].param_groups]
 assert len(set.union(*target_by_group))==len(names)
 assert target_by_group[0]=={id(p) for n,p in names.items() if not n.startswith(('pair.','semantic.'))}
 assert target_by_group[1]=={id(p) for n,p in names.items() if n.startswith('pair.')}
 assert target_by_group[2]=={id(p) for n,p in names.items() if n.startswith('semantic.')}
 checks['optimizer_param_groups_bound_to_live_model']='PASS'
 assert all(len(group['params']) for group in opts[4].param_groups)
 assert all(id(p) in {id(v) for v in m.parameters()} for group in opts[4].param_groups for p in group['params'])
 checks['single_shared_trainable_network']='PASS'
 fake=torch.ones((2,OLDER),dtype=torch.long)
 fake[:,50:53]=torch.tensor([3,7,9]);fake[:,60:63]=torch.tensor([3,7,8])
 query=torch.tensor([[3,7],[3,7]]);p,e,_=m.pair_lookup(fake,query,torch.tensor([9,6]))
 assert e.all() and 0<float(p[0].detach())<1 and float(p[1].detach())==0
 checks['ordered_pair_lookup_preserved']='PASS'
 fake=torch.ones((2,OLDER),dtype=torch.long)
 fake[:,100:102]=torch.tensor([7,9]);query=torch.tensor([[4,7],[4,7]])
 p,e,_,ids,w=m.retrieval.semantic.lookup(fake,query,m.trunk.E,torch.tensor([9,9]))
 assert not e.any() and float(p.sum().detach())==0
 fake[:,100]=8
 p,e,_,ids,w=m.retrieval.semantic.lookup(fake,query,m.trunk.E,torch.tensor([9,9]))
 assert e.all() and float(p.min().detach())>0 and not bool((ids[e]==7).any())
 checks['soft_retrieval_novel_key_nonexact_value']='PASS'
 mutated=fake.clone();mutated[:,101]=10
 p2,e2,_,ids2,w2=m.retrieval.semantic.lookup(mutated,query,m.trunk.E,torch.tensor([9,9]))
 assert e2.all() and float(p2.sum().detach())==0
 checks['values_are_only_historical_following_words']='PASS'
 c=Corpus(root,'test');pos=np.array([2048,4096,8192])
 _,ctx,pc,lens,y,_,_=batch_data(base,c,comp,pos)
 a=torch.as_tensor(c.data[pos].copy());lo=torch.as_tensor(c.long[pos].copy())
 history=torch.as_tensor(c.get(pos,OUTER_CONTEXT)[0][:,:OLDER].copy());state=m.initial(history)
 with torch.no_grad():
  now,nll,metrics=m(a,lo,ctx,pc,lens,state,history,y)
  now2,off,_=m(a,lo,ctx,pc,lens,state,history,y,semantic_off=True)
  oldnow,oldnll,_=PALR76.forward(m,a,lo,ctx,pc,lens,state,history,y)
  assert torch.equal(now,now2) and torch.equal(now,oldnow)
  assert torch.equal(off,oldnll)
  checks['semantic_off_matches_identical_original_r76_path']='PASS'
  modified=a.clone();modified[:,3:]=1234
  _,altered,_=m(modified,lo,ctx,pc,lens,state,history,y)
  assert torch.equal(nll,altered)
  checks['target_and_future_not_injected_into_now']='PASS'
  _,none,_=m(a,lo,ctx,pc,lens,state,history,y,retrieval_off=True)
  assert torch.equal(none,m.trunk.nll(now,y))
  checks['all_retrieval_ablation_exactly_matches_trunk']='PASS'
  _,no_pair,_=m(a,lo,ctx,pc,lens,state,history,y,pair_off=True)
  assert torch.isfinite(no_pair).all() and torch.isfinite(nll).all()
  checks['pair_ablation_preserves_soft_path']='PASS'
  zero=history.clone();zero[:]=1
  _,empty,_=m(a,lo,ctx,pc,lens,state,zero,y)
  _,empty_base,_=m(a,lo,ctx,pc,lens,state,zero,y,retrieval_off=True)
  assert torch.equal(empty,empty_base)
  checks['no_older_memory_no_retrieval_prediction']='PASS'
 m.train()
 _,loss,_=m(a,lo,ctx,pc,lens,state,history,y)
 for op in opts:op.zero_grad(set_to_none=True)
 loss.mean().backward()
 assert m.trunk.P.grad is not None and float(m.trunk.P.grad.abs().sum())>0
 assert m.retrieval.pair.strength.grad is not None
 assert m.retrieval.semantic.cap_logit.grad is not None
 assert torch.isfinite(m.retrieval.semantic.cap_logit.grad)
 checks['trunk_pair_and_new_soft_head_receive_gradients']='PASS'
 return {'step':prior['joint_steps'],'format':prior['format'],'checks':checks,'passed':len(checks)}

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',default='.');a=ap.parse_args();print(json.dumps(run(a.root),indent=2))
