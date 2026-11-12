#!/usr/bin/env python3
"""R16 frozen-inherited-weights matched routing probe, content-weighted objective.
Both Euclidean and Poincare routers live BETWEEN FFN layers. They have identical
parameter counts, batch schedules and zero-gated initialization; checkpoint and
all historical optimizer moments are left completely unchanged. EXPERIMENTAL.
"""
import argparse,collections,hashlib,json,math,pathlib,time
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
import r14_slim_base as pal
from r13_corpus_cache import read_books_cached
from r11_raw import windows,batch
from r16_content_attention_ablation import slice_metadata,CATEGORIES

class Router(nn.Module):
 def __init__(self,geometry,dim=192,experts=4):
  super().__init__();self.geometry=geometry
  self.centers=nn.Parameter(torch.randn(experts,dim)*.12)
  self.values=nn.Parameter(torch.randn(experts,dim)*.1)
  self.alpha=nn.Parameter(torch.zeros(()))
 def forward(self,x):
  # Inside the existing FFN. The residual path and original decoder are shared.
  q=torch.tanh(x/2);q=q/(q.norm(dim=-1,keepdim=True).clamp(min=1.e-4))*.8
  p=torch.tanh(self.centers/2);p=p/(p.norm(dim=-1,keepdim=True).clamp(min=1.e-4))*.8
  sq=(q[:,None,:]-p).square().sum(-1)
  if self.geometry:
   dist=torch.acosh(1+2*sq/((1-q.square().sum(-1,keepdim=True))*(1-p.square().sum(-1))))
   logits=-dist.square()/4.
  else:logits=-sq
  weights=logits.softmax(-1)
  return F.silu(x)*(1+self.alpha*(weights@self.values))

@torch.no_grad()
def base_forward(model,bridge,adapter,h,targets,tags):
 B,T=targets.shape;prev=torch.full((B,),model.memory.start_piece,dtype=torch.long)
 state=h[:,0];last=tags[:,0];retain=bridge.retain_logit.sigmoid();kh=[];vh=[];ls=[];xs=[]
 for t in range(T):
  valid=targets[:,t]!=-100;ix=tags[:,t].clamp(0,h.shape[1]-1)
  next_h=h.gather(1,ix[:,None,None].expand(-1,1,h.shape[-1]))[:,0]
  changed=(ix!=last)&valid
  state=torch.where(changed[:,None],retain*state+(1-retain)*bridge.context_gain*next_h,state)
  state=model.memory.subword_cell(model.memory.piece_emb(prev),state)
  out=state
  if kh:
   keys=torch.stack(kh[-adapter.capacity:],dim=1);values=torch.stack(vh[-adapter.capacity:],dim=1)
   scores=(adapter.query(state).unsqueeze(1)*keys).sum(-1)/math.sqrt(state.shape[-1])
   k=min(adapter.topk,scores.shape[1]);ids=scores.topk(k,dim=1).indices
   selected=values.gather(1,ids[:,:,None].expand(-1,-1,state.shape[-1]))
   recalled=(selected*F.softmax(scores.gather(1,ids),dim=-1)[:,:,None]).sum(1)
   out=out+adapter.attn_gain*adapter.attn_out(recalled)
  x=adapter.ffn[0](state)
  out=out+adapter.ffn_gain*adapter.ffn[2](F.silu(x))
  xs.append(x);ls.append(model.memory.subword_out(out))
  kh.append(adapter.key(state));vh.append(adapter.value(state))
  prev=torch.where(valid,targets[:,t],torch.zeros_like(prev));last=torch.where(valid,ix,last)
 return torch.stack(xs,dim=1),torch.stack(ls,dim=1)

@torch.no_grad()
def build_cache(model,bridge,adapter,books,picks,root,sp):
 xs=[];ls=[];ys=[];cat_w=[];counts=collections.Counter()
 for ix in range(0,len(picks),2):
  chosen=picks[ix:ix+2];e,lc,t,tag,nbytes=batch(books,chosen,24)
  B,W=e.shape[:2];h=model.core.hidden(e.reshape(-1,7),lc.reshape(-1,72)).reshape(B,W,-1)
  x,logits=base_forward(model,bridge,adapter,h,t,tag)
  for i,(bi,start) in enumerate(chosen):
   category_weights,nraw=slice_metadata(books[bi],(bi,start),24,sp,root)
   N=len(category_weights);assert nraw==sum(sum(q.values()) for q in category_weights)
   xs.append(x[i,:N].clone());ls.append(logits[i,:N].clone());ys.append(t[i,:N].clone())
   cat_w.extend(category_weights)
   for row in category_weights:counts.update(row)
 return {'x':torch.cat(xs),'logits':torch.cat(ls),'targets':torch.cat(ys),'weights':cat_w,'category_bytes':dict(counts)}


def evaluate_cache(data,router,W):
 with torch.no_grad():
  base=data['logits'];x=data['x'];labels=data['targets'];N=len(x)
  loss=[]
  for pos in range(0,N,192):
   X=x[pos:pos+192];delta=router(X)-F.silu(X)
   v=base[pos:pos+192]+delta@W
   loss.append(F.cross_entropy(v,labels[pos:pos+192],reduction='none').cpu())
  nll=torch.cat(loss).numpy();total=collections.Counter();size=collections.Counter()
  for i,d in enumerate(data['weights']):
   k=sum(d.values());lv=nll[i]/math.log(2)
   for c,weight in d.items():total[c]+=lv*weight/k;size[c]+=weight
  def bp(cats):
   bytes_=sum(size[c] for c in cats);return float(sum(total[c] for c in cats)/bytes_) if bytes_ else None
  return {'bpb':bp(CATEGORIES),'letters_bpb':bp(['lowercase','uppercase']),
   'content_bpb':bp(['lowercase','uppercase','digits','nonascii']),
   'nonspace_bpb':bp([c for c in CATEGORIES if c!='space']),
   'structure_bpb':bp(['space','punctuation','newline','other_whitespace']),
   'space_bpb':bp(['space']),'bytes':float(sum(size.values())),'alpha':float(router.alpha.detach())}


def main(args):
 torch.set_num_threads(args.threads)
 root=pathlib.Path(args.root).resolve();ckpt=root/'checkpoint/r12_active.pal12';parent=pal.sha(ckpt)
 assert parent=='dc61500ec3abd424409f136ffff517242134af77435e66f135f4363cb7ec450f'
 saved,r6,comp,model,bridge,adapter,opt,rng=pal.load_for_arm(root,ckpt,True)
 assert saved['r12_updates']==10756 and len(opt.param_groups)==4
 model.eval();bridge.eval();adapter.eval()
 books,_=read_books_cached(root/'nanrouter',comp,r6.vocab,0)
 picks=windows(books,24,8000,160)
 train_picks=picks['train'][:args.train_windows];dev_picks=picks['dev'][:96]
 train=build_cache(model,bridge,adapter,books['train'],train_picks,root/'nanrouter',comp.sp)
 dev=build_cache(model,bridge,adapter,books['dev'],dev_picks,root/'nanrouter',comp.sp)
 # Linear readout formula exact because two inherited layers are affine:
 # out = parent_out + gain * FFN2(router(x)-SiLU(x)) then shared output matrix.
 W=adapter.ffn_gain.detach()*(adapter.ffn[2].weight.detach().T@model.memory.subword_out.weight.detach().T)
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(123456);g=Router(False)
 baseline=evaluate_cache(dev,g,W)
 print(json.dumps({'phase':'BASE','parent':parent,'dev':baseline,'train_samples':len(train['targets'])}),flush=True)
 report={'parent_sha256':parent,'parent_updates':saved['r12_updates'],'frozen_inherited_weights':True,
    'all_inherited_optimizer_moments_preserved':True,'train_windows':len(train_picks),'dev_windows':len(dev_picks),
    'objective':f'cross-entropy with content_fraction*{args.content_blend} + uniform*{1-args.content_blend} token weighting',
    'experiment':'Four-prototype Poincare ball metric vs Euclidean squared-distance routing, inserted inside the inherited FFN at identical parameter count, parent-identical zero gate.',
    'baseline':baseline,'arms':{},'promoted':False}
 frac=torch.tensor([sum(w.get(c,0.) for c in ('lowercase','uppercase','digits','nonascii'))/sum(w.values()) for w in train['weights']],dtype=torch.float32)
 N=len(train['targets']);assert N>500
 rg=np.random.default_rng(93882);schedule=rg.integers(0,N,size=(args.steps,args.batch_size))
 report['train_schedule_sha256']=hashlib.sha256(schedule.tobytes()).hexdigest()
 for is_geom,name in [(False,'euclidean'),(True,'poincare')]:
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(123456);router=Router(is_geom)
  assert sum(z.numel() for z in router.parameters())==1537
  optim=torch.optim.AdamW(router.parameters(),lr=args.lr,weight_decay=.00001)
  measurements=[{'step':0,'dev':evaluate_cache(dev,router,W)}];start=time.monotonic()
  for step,idx in enumerate(schedule,1):
   X=train['x'][idx];target=train['targets'][idx];weight=args.content_blend*frac[idx]+(1.-args.content_blend)
   logits=train['logits'][idx]+(router(X)-F.silu(X))@W
   ce=F.cross_entropy(logits,target,reduction='none');loss=(ce*weight).sum()/weight.sum().clamp(min=1.)
   optim.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(router.parameters(),1.);optim.step()
   if step in (args.steps//2,args.steps):
    m=evaluate_cache(dev,router,W);measurements.append({'step':step,'dev':m,'train_loss':float(loss.detach())})
    print(json.dumps({'phase':'EVAL','arm':name,'step':step,'dev':m,'seconds':round(time.monotonic()-start,1)}),flush=True)
  report['arms'][name]={'extra_parameters':sum(z.numel() for z in router.parameters()),'measurements':measurements}
 report['parent_sha256_unchanged']=pal.sha(ckpt)==parent
 out=root/'reports/r16_router_frozen_probe.json';out.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'phase':'DONE','results':{k:v['measurements'][-1]['dev'] for k,v in report['arms'].items()},'parent_unchanged':report['parent_sha256_unchanged']},indent=2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--threads',type=int,default=2);p.add_argument('--steps',type=int,default=100);p.add_argument('--batch-size',type=int,default=96);p.add_argument('--train-windows',type=int,default=96);p.add_argument('--lr',type=float,default=.02);p.add_argument('--content-blend',type=float,default=1.0)
 main(p.parse_args())
