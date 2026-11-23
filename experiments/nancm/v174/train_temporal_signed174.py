#!/usr/bin/env python3
"""NaNCM v174: causal dual-polarity temporal NOW state, exact full-vocab word CE.

Train only on reconstructed TRAIN-book word histories already retained in v173.
New parameters use a book-disjoint holdout within the TRAIN archive. Inherited
v173 teacher weights, parent optimizer and 128+128 signed point cloud are frozen.
This is NOT independent book-disjoint validation of the entire inherited system;
the old teacher's pretraining aggregates may include the heldout TRAIN books.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, re, sys, time, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/nancm/v172'))
from train_conditional_cloud172 import load_parent,CloudNCE,hash166,dev_cohort
WORD=re.compile(rb'[A-Za-z]+')
D=12

def load_sign_geometry(names):
 # In the inherited binary layout, + and - are distinct 128x128x8 surfaces.
 import struct
 data=(ROOT/'checkpoints/v166_lexical_birth.bin').read_bytes();p=8+24+8
 plus=np.frombuffer(data,dtype='<f4',offset=p,count=128*128*8).reshape(128,128,8);p+=128*128*8*4
 minus=np.frombuffer(data,dtype='<f4',offset=p,count=128*128*8).reshape(128,128,8);p+=128*128*8*4
 low=np.frombuffer(data,dtype='<f4',offset=p,count=128*8).reshape(128,8);p+=128*8*4
 high=np.frombuffer(data,dtype='<f4',offset=p,count=128*8).reshape(128,8)
 idx=np.array([hash166(w) for w in names]);pos=low+plus.mean(axis=1);neg=high+minus.mean(axis=1)
 return torch.tensor(np.asarray(pos[idx],dtype='float32')),torch.tensor(np.asarray(neg[idx],dtype='float32'))

def tokenize(history):
 return [x.decode('ascii').lower() for x in WORD.findall(history.encode('latin1'))]

def make_dataset(names):
 vocab={w:i for i,w in enumerate(names)}
 meta=[json.loads(s) for s in (ROOT/'results/v172_train_context_positions.jsonl').read_text().splitlines()]
 assert len(meta)>1000
 books=sorted({x['book'] for x in meta},key=lambda b:hashlib.sha256(('v174-bookholdout:'+b).encode()).digest())
 holdout=set(books[-20:])
 observed={}
 for row in meta:
  b=row['book']; words=tokenize(row['history'])
  # All words inside a TRAIN passage are valid causal supervised targets.
  # The row's target is TRAIN-derived, too. Deduplicate overlapping windows.
  words=words+[row['truth'].lower()]
  for t in range(1,len(words)):
   target=vocab.get(words[t]);
   if target is None:continue
   ix=[vocab.get(w,-1) for w in words[max(0,t-D):t]]
   ix=([-1]*(D-len(ix)))+ix
   if sum(v>=0 for v in ix)==0:continue
   k=(b,tuple(ix),target)
   observed[k]=1
 train,valid=[],[]
 for (b,ix,y) in observed:
  (valid if b in holdout else train).append((ix,y,b))
 def arr(rows):
  return np.asarray([r[0] for r in rows],dtype=np.int32),np.asarray([r[1] for r in rows],dtype=np.int64),[r[2] for r in rows]
 return arr(train),arr(valid),books,holdout

class DualPolarTemporal(nn.Module):
 def __init__(self,teacher:CloudNCE,names,gpos,gneg,dim=64):
  super().__init__()
  self.register_buffer('embed',teacher.ctx.weight.detach().clone())
  self.register_buffer('out',teacher.out.weight.detach().clone())
  self.register_buffer('bias',teacher.bias.detach().clone())
  self.register_buffer('geo',teacher.geo.detach().clone())
  self.register_buffer('projection',teacher.cloud_projection.weight.detach().clone())
  self.register_buffer('gain',teacher.gain.detach().clone())
  self.register_buffer('gpos',gpos.detach().clone());self.register_buffer('gneg',gneg.detach().clone())
  self.gru_pos=nn.GRU(48,dim,batch_first=True)
  self.gru_neg=nn.GRU(48,dim,batch_first=True)
  self.signed_norm=nn.LayerNorm(3*dim)
  self.to_lex=nn.Sequential(nn.Linear(dim*3,96),nn.GELU(),nn.Linear(96,32))
  self.trust=nn.Parameter(torch.tensor(-3.0))
  self.signed_scale=nn.Parameter(torch.tensor(0.0))
 def forward(self,hist,return_components=False):
  mask=(hist>=0);h=hist.clamp(min=0).long();b,n=h.shape
  w=mask.float();weights=torch.tensor([.2,.3,.5],device=h.device)[None,:]*w[:,-3:]
  weights=weights/(weights.sum(1,keepdim=True)+1e-10)
  vec=(self.embed[h[:,-3:]]*weights.unsqueeze(-1)).sum(dim=1)
  geo=(self.geo[h[:,-3:]]*weights.unsqueeze(-1)).sum(dim=1)
  q=F.normalize(vec+F.linear(geo,self.projection),dim=-1,eps=1e-8)
  baseline=(self.gain.clamp(0,25)*q@self.out.T)+self.bias
  # Positive and negative 128-point rails remain separate and interact only
  # after their causal GRU states have reached the current word boundary.
  base=self.embed[h]*mask.unsqueeze(-1)
  p=torch.cat([base,self.gpos[h]*mask.unsqueeze(-1),self.gneg[h]*mask.unsqueeze(-1)],dim=2)
  neg=torch.cat([base,self.gneg[h]*mask.unsqueeze(-1),self.gpos[h]*mask.unsqueeze(-1)],dim=2)
  pos_state,_=self.gru_pos(p)
  neg_state,_=self.gru_neg(neg)
  # Last word is the current causal horizon. Padding is left, not right.
  hp=pos_state[:,-1,:];hn=neg_state[:,-1,:]
  signed=torch.cat([hp,hn,hp*hn],dim=-1)
  res=F.normalize(self.to_lex(self.signed_norm(signed)),dim=-1,eps=1e-8)
  alpha=F.softplus(self.trust)*2
  logits=baseline+alpha*(res@self.out.T)*self.gain.clamp(0,25)
  if return_components:return logits,baseline,alpha
  return logits

def measure(model,X,Y,batch=128):
 model.eval();n=0;loss=0.;hit1=hit5=0.;old_loss=old1=old5=0.
 with torch.no_grad():
  for i in range(0,len(Y),batch):
   x=torch.as_tensor(X[i:i+batch].astype('int64'));y=torch.as_tensor(Y[i:i+batch]);z,ref,_=model(x,True)
   for probs,base in ((z,False),(ref,True)):
    l=F.cross_entropy(probs,y,reduction='sum').item();top=probs.topk(5,dim=-1).indices
    h1=int((top[:,0]==y).sum());h5=int((top==y[:,None]).any(1).sum())
    if base:old_loss+=l;old1+=h1;old5+=h5
    else:loss+=l;hit1+=h1;hit5+=h5
   n+=len(y)
 return dict(n=n,nll=loss/n,top1=int(hit1),top5=int(hit5),baseline_nll=old_loss/n,baseline_top1=int(old1),baseline_top5=int(old5))

def diagnostic_dev(model,vocab):
 sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'parent/scripts')]
 from train_gate163 import winread
 wins=list(winread(ROOT/'external_book_gate/heldout_dev160.bin'))
 rows=[json.loads(s) for s in (ROOT/'results/v169_portfolio_dev.jsonl').read_text().splitlines()]
 X=[];Y=[]
 for row in rows:
  h=wins[int(row['book'])][0][:row['pos']].decode('latin1')
  ws=tokenize(h);ids=[vocab.get(w,-1) for w in ws[-D:]];X.append(([-1]*(D-len(ids)))+ids);Y.append(vocab.get(row['truth'],-1))
 X=np.asarray(X,dtype=np.int32);Y=np.asarray(Y,dtype=np.int64)
 ok=Y>=0
 return measure(model,X[ok],Y[ok]),dict(known=int(ok.sum()),total=len(Y))

def main():
 pa=argparse.ArgumentParser();pa.add_argument('--steps',type=int,default=600);pa.add_argument('--batch',type=int,default=80);pa.add_argument('--log-every',type=int,default=50);pa.add_argument('--resume',action='store_true');args=pa.parse_args()
 torch.set_num_threads(int(os.getenv('V174_THREADS','2')));torch.manual_seed(174);np.random.seed(174)
 names,C,O,B,spatial,gain,fp=load_parent()
 ck=torch.load(ROOT/'checkpoints/v173_exact_ce_word.pt',weights_only=False,map_location='cpu')
 assert ck['word_order']==names and tuple(ck['fingerprints'])==tuple(fp)
 teacher=CloudNCE(C,O,B,spatial,gain,names);teacher.load_state_dict(ck['state_dict']);teacher.eval()
 for p in teacher.parameters():p.requires_grad_(False)
 gp,gn=load_sign_geometry(names);model=DualPolarTemporal(teacher,names,gp,gn)
 tr,va,books,holdout=make_dataset(names)
 print('DATA',{'train':len(tr[1]),'val':len(va[1]),'TRAIN_books':len(books),'heldout_books':len(holdout),'orig_teacher_step':ck['step']},flush=True)
 # Validation hyperparameters selected only from separately heldout TRAIN books.
 rng=np.random.default_rng(174);order=rng.integers(len(tr[1]),size=(args.steps,args.batch));model.train()
 optimizer=torch.optim.AdamW([{'params':list(model.gru_pos.parameters())+list(model.gru_neg.parameters()),'lr':0.0006},
     {'params':list(model.signed_norm.parameters())+list(model.to_lex.parameters()),'lr':0.0008},
     {'params':[model.trust,model.signed_scale],'lr':0.0003}],weight_decay=.01)
 valX,valY=va[:2];best={'val_loss':1e10};logs=[];start=time.time();parent_step=0
 if args.resume:
  prev=torch.load(ROOT/'checkpoints/v174_temporal_signed_now.pt',weights_only=False,map_location='cpu')
  assert prev['word_order']==names and tuple(prev['fingerprints'])==tuple(fp)
  model.load_state_dict(prev['model_state_dict'])
  optimizer.load_state_dict(prev['optimizer_state_dict'])
  parent_step=int(prev['training_step'])
  # The selected snapshot and all old AdamW moments remain intact.
  print('RESUMED',parent_step,'ADAM_GROUPS',len(optimizer.param_groups),'STATES',len(optimizer.state),flush=True)
 init=measure(model,valX,valY);best={'val_loss':init['nll'],'step':parent_step,'state':copy.deepcopy(model.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),'val':init}
 print('INIT_VAL',init,flush=True)
 for step in range(args.steps):
  ix=order[step];x=torch.from_numpy(tr[0][ix]).long();y=torch.from_numpy(tr[1][ix]).long()
  model.train();z=model(x);loss=F.cross_entropy(z,y)
  optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
  if (step+1)%args.log_every==0 or step+1==args.steps:
   met=measure(model,valX,valY)
   row={'step':parent_step+step+1,'train_loss':float(loss.detach()),'val':met,'s':round(time.time()-start,2)};logs.append(row)
   print('ITER',json.dumps(row),flush=True)
   if met['nll'] < best['val_loss']:
    best={'val_loss':met['nll'],'step':parent_step+step+1,'state':copy.deepcopy(model.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),'val':met}
 model.load_state_dict(best['state'])
 diag,coverage=diagnostic_dev(model,{w:i for i,w in enumerate(names)})
 print('BEST',best['step'],best['val'],'DEV_REUSED',diag,'COVERAGE',coverage,flush=True)
 path=ROOT/'checkpoints/v174_temporal_signed_now.pt'
 torch.save({'model_state_dict':best['state'],'optimizer_state_dict':best['optimizer'],
  'parent_checkpoint':str('checkpoints/v173_exact_ce_word.pt'),'parent_checkpoint_step':ck['step'],
  'parent_optimizer_state_dict':ck['optimizer_state_dict'],
  'fingerprints':fp,'word_order':names,'training_step':best['step'],
  'architecture':'v174 dual signed causal temporal GRUs, full exact 12k CE, 128D temporal NOW'},path)
 rep={'architecture':'v174 dual signed causal temporal GRUs 64+64, coupled states and frozen inherited teacher',
  'source':'TRAIN-only archived book-prefix contexts (overlap-deduplicated, causal)',
  'train_contexts':len(tr[1]),'val_contexts':len(va[1]),'train_books':len(books)-len(holdout),'heldout_train_books':len(holdout),
  'holdout_not_pristine':True,'checkpoint_step':best['step'],'optimizer_preserved':True,
  'parent_step':ck['step'],'initial':init,'selected_validation':best['val'],'reused_dev':diag,'reused_dev_coverage':coverage,
  'history':logs,'timing_s':round(time.time()-start,2),'resumed_from':parent_step,'promotion':'NOT_AUTO_PROMOTED'}
 (ROOT/'results/v174_temporal_signed_train.json').write_text(json.dumps(rep,indent=2))
 print('SAVED',path,round(path.stat().st_size/1e6,2),'MB',flush=True)
if __name__=='__main__':main()