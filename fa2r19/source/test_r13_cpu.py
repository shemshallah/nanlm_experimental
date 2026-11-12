#!/usr/bin/env python3
"""Compare R13 cached fast path against complete R12 forward and gradients."""
import argparse,copy,json,time,torch
from r12_experiment import load_for_arm,run_window
from r13_cpu import run_window_fast

def run(args):
 torch.set_num_threads(args.threads)
 saved,r6,comp,model,bridge,adapter,opt,rng=load_for_arm(args.root,args.checkpoint,True)
 # Fixed synthetic inputs ensure reproducible test with exact pretrained model.
 gen=torch.Generator().manual_seed(11)
 B,T,W=args.batch,args.tokens,args.chunks
 ex=torch.randint(1,100,(B,W,7),generator=gen)
 lc=torch.zeros(B,W,72)
 targets=torch.randint(0,256,(B,T),generator=gen)
 tags=torch.arange(T)[None,:].expand(B,-1)*W//T
 tags=tags.to(torch.long)
 # Include padding and multiple distinct variable chunk boundaries.
 if B>1:targets[1,-7:]=-100
 h=model.core.hidden(ex.reshape(-1,7),lc.reshape(-1,72)).reshape(B,W,-1)
 errors={};times={}
 for name,runfun in [('original',run_window),('cached',run_window_fast)]:
  model.zero_grad(set_to_none=True);bridge.zero_grad(set_to_none=True);adapter.zero_grad(set_to_none=True)
  t0=time.perf_counter()
  for i in range(args.rounds):
   model.zero_grad(set_to_none=True);bridge.zero_grad(set_to_none=True);adapter.zero_grad(set_to_none=True)
   # Rebuild h to avoid repeated graph usage while benchmarking end-to-end.
   hh=model.core.hidden(ex.reshape(-1,7),lc.reshape(-1,72)).reshape(B,W,-1)
   loss,n=runfun(model,bridge,adapter,hh,targets,tags,training=True)
   loss.backward()
  times[name]=(time.perf_counter()-t0)/args.rounds
  errors[name]={'loss':float(loss.detach()),'attn_gain_grad':float(adapter.attn_gain.grad),
     'ffn_gain_grad':float(adapter.ffn_gain.grad),
     'query_grad':adapter.query.weight.grad.detach().clone(),
     'key_grad':adapter.key.weight.grad.detach().clone(),
     'value_grad':adapter.value.weight.grad.detach().clone(),
     'head_grad':model.memory.subword_out.weight.grad.detach().clone(),
     'mirror_p_grad':model.core.P.grad.detach().clone() if model.core.P.grad is not None else None}
 diff={}
 for key in ('loss','attn_gain_grad','ffn_gain_grad','query_grad','key_grad','value_grad','head_grad','mirror_p_grad'):
  x,y=errors['original'][key],errors['cached'][key]
  diff[key]=float((x-y).abs().max()) if isinstance(x,torch.Tensor) else abs(x-y)
 # Strict numerical tolerance, since reduction order differs by design.
 assert diff['loss']<1e-5,diff
 assert all(diff[k]<1e-4 for k in diff if k!='loss'),diff
 result={'passed':True,'batch':B,'tokens':T,'chunks':W,'threads':args.threads,'rounds':args.rounds,
         'loss_absolute_delta':diff['loss'],'max_gradient_absolute_delta':max(v for k,v in diff.items() if k!='loss'),
         'delta_by_field':diff,'seconds_per_step':times,'relative_speedup':times['original']/times['cached']}
 print(json.dumps(result,indent=2))
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.')
 p.add_argument('--checkpoint',default='checkpoint/r12_active.pal12')
 p.add_argument('--batch',type=int,default=2);p.add_argument('--tokens',type=int,default=72)
 p.add_argument('--chunks',type=int,default=12);p.add_argument('--threads',type=int,default=2)
 p.add_argument('--rounds',type=int,default=2)
 run(p.parse_args())
