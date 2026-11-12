#!/usr/bin/env python3
"""Matched, non-promotable FFN geometry-vs-Euclidean routing experiment.

Both arms continue the exact R15 weights, same optimizer moments, same RNG
training picks and 4 original optimizer groups; new route parameters have
identical shapes, initial values and zero-gated outputs. The only difference
is the routing distance. Parent checkpoint is never modified.
"""
import argparse,hashlib,json,math,pathlib,time
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
import r14_slim_base as pal
from r13_corpus_cache import read_books_cached
from r11_raw import windows,batch
from r16_content_attention_ablation import forward_token_losses,slice_metadata,category,CATEGORIES

class RoutingFFN(nn.Module):
 def __init__(self,kind,dim=192,heads=4):
  super().__init__();assert kind in ('hyperbolic','euclidean')
  self.kind=kind
  self.centers=nn.Parameter(torch.randn(heads,dim)*.12)
  self.values=nn.Parameter(torch.randn(heads,dim)*.10)
  self.alpha=nn.Parameter(torch.zeros(()))
 def forward(self,projection):
  # Routing takes place INSIDE FFN, between inherited first and second layers.
  q=torch.tanh(projection/4)/math.sqrt(projection.shape[-1])
  p=torch.tanh(self.centers/4)/math.sqrt(projection.shape[-1])
  sq=(q[:,None,:]-p[None,:,:]).square().sum(-1)
  if self.kind=='euclidean':logits=-sq
  else:
   # Poincare ball d=arcosh(1+2||x-y||^2/[(1-||x||^2)(1-||y||^2)])
   den=(1-q.square().sum(-1,keepdim=True)).clamp(min=1.e-5)*(1-p.square().sum(-1))[None,:].clamp(min=1.e-5)
   dist=torch.acosh(1+2*sq/den)
   logits=-dist.square()/4
  mix=logits.softmax(-1)@self.values
  return F.silu(projection)*(1+self.alpha*mix)


def metrics(model,bridge,adapter,router,books,picks,root,sp):
 model.eval();bridge.eval();adapter.eval();router.eval()
 totals={c:{'bytes':0.,'bits':0.} for c in CATEGORIES};totalbits=0.;nbytes=0
 with torch.no_grad():
  for j in range(0,len(picks),2):
   selected=picks[j:j+2];e,lc,t,tag,bcount=batch(books,selected,24)
   B,W=e.shape[:2];h=model.core.hidden(e.reshape(-1,7),lc.reshape(-1,72)).reshape(B,W,-1)
   nll,_=forward_token_losses(model,bridge,adapter,h,t,tag,router=router)
   for i,(bi,start) in enumerate(selected):
    weights,numbytes=slice_metadata(books[bi],(bi,start),24,sp,root)
    nbytes+=numbytes;totalbits+=float(nll[i,:len(weights)].sum())/math.log(2)
    for k,cs in enumerate(weights):
     bs=sum(cs.values());l=float(nll[i,k])/math.log(2)
     for c,wt in cs.items():totals[c]['bytes']+=wt;totals[c]['bits']+=l*wt/bs
 def cat_bpb(cats):
  n=sum(totals[c]['bytes'] for c in cats);bits=sum(totals[c]['bits'] for c in cats)
  return bits/n if n else None
 content=['lowercase','uppercase','digits','nonascii']
 return {'bpb':totalbits/nbytes,'letters_bpb':cat_bpb(['lowercase','uppercase']),
         'content_bpb':cat_bpb(content),'nonspace_bpb':cat_bpb([c for c in CATEGORIES if c!='space']),
         'structure_bpb':cat_bpb(['space','punctuation','newline','other_whitespace']),
         'bytes':nbytes,'alpha':float(router.alpha),'checksums':{'sum_category_bits':sum(v['bits'] for v in totals.values()),'total_bits':totalbits}}


def main(args):
 torch.set_num_threads(args.threads);root=pathlib.Path(args.root).resolve()
 ckpt=root/'checkpoint/r12_active.pal12';orig=pal.sha(ckpt)
 assert orig=='dc61500ec3abd424409f136ffff517242134af77435e66f135f4363cb7ec450f'
 # External data load is read-only, and same selection used for both arms.
 saved0,r6,comp,_,_,_,_,_=pal.load_for_arm(root,ckpt,True)
 docs,_=read_books_cached(root/'nanrouter',comp,r6.vocab,0)
 allp=windows(docs,24,8000,160)
 train=allp['train'];dev=allp['dev'][:96]
 stage_ticks=[0,args.steps//2,args.steps]
 # Shared fixed training selection sampled from parent R15 RNG-state clone.
 rg=np.random.default_rng();rg.bit_generator.state=saved0['stream_rng']
 batches=[[train[int(k)] for k in rg.integers(0,len(train),size=2)] for _ in range(args.steps)]
 report={'parent_sha256':orig,'parent_r12_updates':saved0['r12_updates'],'steps_per_arm':args.steps,
         'arms':{},'development_selection_hash':hashlib.sha256(json.dumps(dev).encode()).hexdigest(),
         'training_batches_hash':hashlib.sha256(json.dumps(batches).encode()).hexdigest(),
         'checkpoint_modified':False,'claim':'Matched research arms; candidate weights are deliberately not promoted or released',
         'controls':'Same inherited weights, inherited optimizer moments, identical training batch sequence, identical gate and prototype parameter counts. Routing distance only changes.'}
 for kind in ('euclidean','hyperbolic'):
  saved,r6,comp,model,bridge,adapter,opt,rng=pal.load_for_arm(root,ckpt,True)
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(9164);router=RoutingFFN(kind)
  opt.add_param_group({'params':list(router.parameters()),'lr':args.router_lr,'weight_decay':.00001})
  arm={'parameters':sum(p.numel() for p in router.parameters()),'measurements':[]}
  for step in range(args.steps+1):
   if step in stage_ticks:
    m=metrics(model,bridge,adapter,router,docs['dev'],dev,root/'nanrouter',comp.sp)
    arm['measurements'].append({'steps':step,'dev':m})
    print(json.dumps({'phase':'GATE','arm':kind,'step':step,'dev':m}),flush=True)
   if step==args.steps:break
   model.train();bridge.train();adapter.train();router.train()
   e,lc,t,tag,_=batch(docs['train'],batches[step],24)
   B,W=e.shape[:2];h=model.core.hidden(e.reshape(-1,7),lc.reshape(-1,72)).reshape(B,W,-1)
   nll,_=forward_token_losses(model,bridge,adapter,h,t,tag,router=router,return_tensors=True)
   loss=nll.sum()/(t!=-100).sum()
   if not torch.isfinite(loss):raise ValueError('nonfinite')
   opt.zero_grad(set_to_none=True);loss.backward()
   torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(bridge.parameters())+list(adapter.parameters())+list(router.parameters()),1.)
   opt.step()
  report['arms'][kind]=arm
  del model,bridge,adapter,opt,router
 report['parent_unchanged']=pal.sha(ckpt)==orig
 report['comparison']={name:{'euclidean':report['arms']['euclidean']['measurements'][-1]['dev'][name],
   'hyperbolic':report['arms']['hyperbolic']['measurements'][-1]['dev'][name]} for name in ('bpb','letters_bpb','content_bpb','nonspace_bpb','structure_bpb')}
 out=root/'reports/r16_matched_router.json';out.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'phase':'DONE','parent_unchanged':report['parent_unchanged'],'comparison':report['comparison']},indent=2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--threads',type=int,default=2);p.add_argument('--steps',type=int,default=100);p.add_argument('--router-lr',type=float,default=.004)
 main(p.parse_args())
