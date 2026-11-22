#!/usr/bin/env python3
"""Experimental CPU word-conditional signed-cloud training on TRAIN-only
v164 ngram counts. Preserves inherited v166/167/170 checkpoints and optimizer.

Open-world caveat: count aggregates have no per-book provenance; deterministic
context-key holdout is NOT book-disjoint, and the reused DEV comparison is not
an untouched independent test. Never auto-promote this head.
"""
import json,math,hashlib,struct,re,time,sys,os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3]
VOC=12000;DIM=32;K=64

def load_parent():
 data=(ROOT/'checkpoints/v166_lexical_birth.bin').read_bytes();pos=0
 assert data[:8]==b'N166BRT1';pos+=8
 fp=struct.unpack_from('<3Q',data,pos);pos+=24
 nv,d=struct.unpack_from('<2I',data,pos);pos+=8
 assert nv==VOC and d==DIM
 plus=np.frombuffer(data,offset=pos,count=128*128*8,dtype='<f4').copy().reshape(128,128,8);pos+=128*128*8*4
 minus=np.frombuffer(data,offset=pos,count=128*128*8,dtype='<f4').copy().reshape(128,128,8);pos+=128*128*8*4
 low=np.frombuffer(data,offset=pos,count=128*8,dtype='<f4').copy().reshape(128,8);pos+=128*8*4
 high=np.frombuffer(data,offset=pos,count=128*8,dtype='<f4').copy().reshape(128,8);pos+=128*8*4
 lex_gain,cloud_gain=struct.unpack_from('<2f',data,pos);pos+=8
 names=[];b=[];ctx=[];out=[]
 for i in range(nv):
  n=data[pos];pos+=1;names.append(data[pos:pos+n].decode('ascii'));pos+=n
  b.append(struct.unpack_from('<f',data,pos)[0]);pos+=4
  ctx.append(np.frombuffer(data,offset=pos,count=DIM,dtype='<f4').copy());pos+=4*DIM
  out.append(np.frombuffer(data,offset=pos,count=DIM,dtype='<f4').copy());pos+=4*DIM
 assert pos==len(data)
 # The signed S+/S- 128+128 cloud is inherited, not reinitialized.
 spatial=low+high+.5*((plus-minus).mean(axis=1))
 return names,np.asarray(ctx),np.asarray(out),np.asarray(b),spatial,float(lex_gain),fp

def hash166(word):
 h=2166136261
 for x in word.encode('ascii'):h=((h^x)*16777619)&0xffffffff
 return h&127

class CloudNCE(torch.nn.Module):
 def __init__(self,contexts,outputs,bias,spatial,gain,words):
  super().__init__();self.ctx=torch.nn.Embedding.from_pretrained(torch.tensor(contexts).float(),freeze=False)
  self.out=torch.nn.Embedding.from_pretrained(torch.tensor(outputs).float(),freeze=False)
  self.bias=torch.nn.Parameter(torch.tensor(bias).float())
  self.cloud_projection=torch.nn.Linear(8,DIM,bias=False)
  torch.nn.init.zeros_(self.cloud_projection.weight)
  self.register_buffer('geo',torch.tensor(np.array([spatial[hash166(w)] for w in words])).float())
  self.gain=torch.nn.Parameter(torch.tensor(gain).float())
 def query(self,indices):
  mask=(indices>=0).float();idx=indices.clamp(min=0)
  order_weights=torch.tensor([.2,.3,.5],device=indices.device)
  weights=mask*order_weights;weights=weights/(weights.sum(1,keepdim=True)+1e-10)
  v=(self.ctx(idx)*weights.unsqueeze(-1)).sum(dim=1)
  z=(self.geo[idx]*weights.unsqueeze(-1)).sum(dim=1)
  return F.normalize(v+self.cloud_projection(z),p=2,dim=1,eps=1e-8)
 def logits(self,query,target):
  # target may [batch] or [batch,negatives]
  emb=self.out(target);q=query if target.ndim==1 else query[:,None,:]
  return (q*emb).sum(-1)*self.gain.clamp(0.,25.)+self.bias[target]

def word_ids(context,mapping):
 ws=context.split();c=[-1,-1,-1]
 for i,w in enumerate(ws[-3:]):c[3-len(ws[-3:])+i]=mapping.get(w,-1)
 return c

def load_examples(mapping):
 examples=[];freq=np.zeros(VOC,dtype='float64');dropped=0
 with (ROOT/'checkpoints/v164_native_distribution.tsv').open() as f:
  assert next(f).startswith('N164DIST1')
  for line in f:
   order,context,word,n,total=line.rstrip('\n').split('\t');target=mapping.get(word)
   if target is None:continue
   ids=word_ids(context,mapping)
   if all(i<0 for i in ids):continue
   n=int(n)
   # Sample count with square-root tempering to avoid common function words
   # monopolizing every update; still learn exact word likelihood ranking.
   weight=math.sqrt(n)
   digest=hashlib.sha256(('v172-ngram-context:'+order+':'+context).encode()).digest()
   val=digest[0]<13
   examples.append((ids,target,weight,val))
   freq[target]+=n
 X=np.array([a for a,b,w,v in examples],dtype='int32');Y=np.array([b for a,b,w,v in examples],dtype='int64');W=np.array([w for a,b,w,v in examples],dtype='float64');V=np.array([v for a,b,w,v in examples],dtype='bool')
 return X,Y,W,V,freq

def eval_full(model,indices,targets,batch=256):
 model.eval();loss=0.;hit1=hit5=total=0
 with torch.no_grad():
  out=model.out.weight.T;bias=model.bias;gain=model.gain.clamp(0,25)
  for start in range(0,len(targets),batch):
   ids=torch.tensor(indices[start:start+batch],dtype=torch.long); y=torch.tensor(targets[start:start+batch],dtype=torch.long)
   q=model.query(ids);scores=gain*q@out+bias
   loss+=F.cross_entropy(scores,y,reduction='sum').item();top=scores.topk(5,dim=1).indices
   hit1+=(top[:,0]==y).sum().item();hit5+=(top==y[:,None]).any(1).sum().item();total+=len(y)
 return {'n':total,'nll_nats':loss/total,'top1':hit1,'top5':hit5}

def dev_cohort(mapping):
 sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'parent/scripts')]
 from train_gate163 import winread
 wins=list(winread(ROOT/'external_book_gate/heldout_dev160.bin'))
 rows=[json.loads(s) for s in (ROOT/'results/v169_portfolio_dev.jsonl').read_text().splitlines()]
 x=[];y=[]
 for r in rows:
  h=wins[int(r['book'])][0][:r['pos']]
  previous=[m.group().lower().decode('ascii') for m in re.finditer(rb'[A-Za-z]+',h)]
  x.append(word_ids(' '.join(previous[-3:]),mapping));y.append(mapping.get(r['truth'],-1))
 return np.array(x,dtype='int32'),np.array(y,dtype='int64'),rows

def main():
 torch.set_num_threads(2);torch.manual_seed(172);np.random.seed(172)
 names,C,O,B,spatial,gain,fp=load_parent();vocab={w:i for i,w in enumerate(names)}
 X,Y,W,V,freq=load_examples(vocab)
 tr=np.where(~V)[0];va=np.where(V)[0];rng=np.random.default_rng(172)
 print('DATA',len(X),'train',len(tr),'val_context_rows',len(va),'vocab',len(names),'train_only',flush=True)
 model=CloudNCE(C,O,B,spatial,gain,names)
 baseline=eval_full(model,X[va[:1024]],Y[va[:1024]])
 D,Dy,dev=dev_cohort(vocab);known=np.where(Dy>=0)[0]
 dev_init=eval_full(model,D[known],Dy[known]);print('PRETRAINED',baseline,'DEV diagnostic',dev_init,'DEV vocab coverage',len(known),flush=True)
 freq=(freq+1)**.75;freq/=freq.sum();noise=torch.tensor(freq,dtype=torch.float32)
 opt=torch.optim.AdamW([{'params':[model.ctx.weight,model.out.weight],'lr':.0004},{'params':[model.bias,model.cloud_projection.weight],'lr':.001},{'params':[model.gain],'lr':.00004}],weight_decay=.002)
 # Cached random sample schedule, no use of DEV labels in optimization.
 train_prob=W[tr]/W[tr].sum();draw=rng.choice(tr,size=(480,256),p=train_prob).astype('int64')
 negatives=torch.multinomial(noise,480*256*K,replacement=True).reshape(480,256,K)
 log_noise=torch.log(noise+1e-30)
 eval_snapshot=[];best=None;started=time.perf_counter()
 for step in range(480):
  ids=torch.tensor(X[draw[step]],dtype=torch.long);target=torch.tensor(Y[draw[step]],dtype=torch.long)
  neg=negatives[step]
  q=model.query(ids)
  pos_s=model.logits(q,target)-math.log(K)-log_noise[target]
  neg_s=model.logits(q,neg)-math.log(K)-log_noise[neg]
  loss=F.softplus(-pos_s).mean()+F.softplus(neg_s).sum(dim=1).mean()
  loss+=.0001*model.cloud_projection.weight.square().sum()
  opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
  if (step+1)%80==0:
   m=eval_full(model,X[va[:1024]],Y[va[:1024]])
   eval_snapshot.append({'step':step+1,'val':m,'train_noise_loss':loss.item()})
   print('STEP',step+1,'loss',round(loss.item(),5),'val',m,'elapsed',round(time.perf_counter()-started,1),flush=True)
   if best is None or m['nll_nats']<best['nll']:
    best={'step':step+1,'nll':m['nll_nats'],'state':{k:v.detach().clone() for k,v in model.state_dict().items()},'optimizer':opt.state_dict()}
 model.load_state_dict(best['state']);after=eval_full(model,D[known],Dy[known]);valbest=eval_full(model,X[va[:1024]],Y[va[:1024]])
 print('SELECTED STEP',best['step'],'DEV',after,'val',valbest,flush=True)
 # Retain COMPLETE optimizer state, not just the neural weights.
 torch.save({'architecture':'NaNCM v172 signed lexical conditional NCE','step':best['step'],'state_dict':best['state'],'optimizer_state_dict':best['optimizer'],'word_order':names,'fingerprints':fp,'trained_on':'TRAIN-derived v164 ngram contexts only','selection':'5% SHA-keyed context holdout; NOT book-disjoint'},ROOT/'checkpoints/v172_conditional_signed_cloud.pt')
 report={'name':'NaNCM v172 conditional signed-cloud NCE pilot','data':{'rows':int(len(X)),'train_rows':int(len(tr)),'validation_rows':int(len(va)),'word_vocab':len(names),'dev_evaluated':len(known),'dev_total':len(dev)},'parent_fingerprints':list(fp),'steps':480,'selected_step':best['step'],'pretrained_validation':baseline,'pretrained_dev':dev_init,'selected_validation':valbest,'selected_dev':after,'history':eval_snapshot,'promotion':'RESEARCH_ONLY unless full native strict word comparison and fresh book certification','train_data':'TRAIN-derived 500-book conditional counts, filtered to v166 vocab','cloud_geometry':'inherited 128+128 signed S+/S- row difference plus lower/upper geometry, trainable 8x32 projection, inherited lexical 32D weights','leakage_note':'validation is 5% context-key holdout NOT book-disjoint; DEV is historically exposed'}
 (ROOT/'results/v172_conditional_cloud_train.json').write_text(json.dumps(report,indent=2));print('COMPLETE',json.dumps({k:v for k,v in report.items() if k not in ('history','parent_fingerprints')}),flush=True)
if __name__=='__main__':main()