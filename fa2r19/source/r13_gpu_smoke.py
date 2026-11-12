#!/usr/bin/env python3
"""Optional GPU device/optimizer smoke test; never overwrites PAL checkpoint.

This is NOT a performance benchmark. Full migration of all historical
regression/evaluation paths is required before actual GPU promotion.
"""
import argparse,hashlib,json,torch
from r12_experiment import load_for_arm
from r13_cpu import run_window_fast

def main(a):
 if not torch.cuda.is_available():
  print(json.dumps({'status':'SKIPPED','reason':'No CUDA device exposed by PyTorch','torch':torch.__version__}))
  return
 source_hash=hashlib.sha256(open(a.checkpoint,'rb').read()).hexdigest()
 saved,r6,comp,model,br,ad,opt,rng=load_for_arm(a.root,a.checkpoint,True)
 device=torch.device('cuda')
 model.to(device);br.to(device);ad.to(device)
 for state in opt.state.values():
  for key,val in list(state.items()):
   if torch.is_tensor(val) and key!='step':state[key]=val.to(device)
 B,T,W=2,80,12
 torch.manual_seed(1337)
 ex=torch.randint(1,100,(B,W,7),device=device)
 lc=torch.zeros(B,W,72,device=device)
 y=torch.randint(0,256,(B,T),device=device)
 tag=(torch.arange(T,device=device)[None,:].expand(B,-1)*W//T).long()
 h=model.core.hidden(ex.reshape(-1,7),lc.reshape(-1,72)).reshape(B,W,-1)
 loss,n=run_window_fast(model,br,ad,h,y,tag,training=True)
 opt.zero_grad(set_to_none=True);loss.backward()
 torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(br.parameters())+list(ad.parameters()),1.)
 opt.step();torch.cuda.synchronize()
 result={'status':'PASS','device':torch.cuda.get_device_name(0),'loss':float(loss.detach()),
         'finite_loss':bool(torch.isfinite(loss).item()),'grad_present':bool(ad.ffn[0].weight.grad is not None),
         'original_checkpoint_unmodified':source_hash==hashlib.sha256(open(a.checkpoint,'rb').read()).hexdigest(),
         'note':'One synthetic GPU step only, no GPU timing or corpus evaluation'}
 print(json.dumps(result,indent=2))
 assert result['finite_loss'] and result['grad_present'] and result['original_checkpoint_unmodified']
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--checkpoint',default='checkpoint/r12_active.pal12')
 main(p.parse_args())