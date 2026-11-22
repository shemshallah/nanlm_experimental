#!/usr/bin/env python3
"""Proof-of-concept long-range signed-cloud associative lexical birth.

Uses the full TRAIN-only v164 conditional index and the inherited trained v166
128+128 signed geometry + 32D lexical key. No DEV labels in search or index;
truth is compared only after nearest neighbors return predictions.
"""
import argparse,collections,hashlib,json,re,sys,time,os
from pathlib import Path
import numpy as np,torch
R=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(R/'experiments/nancm/v166'),str(R/'scripts'),str(R/'parent/scripts')]
from eval_birth166 import collect, evaluate
from train_birth166 import fnv
from train_gate163 import winread
from test_native_speller156 import getwords

def load_keys(path):
 ctx=[];ks=[];candidate=[];key=None;group=[]
 def flush():
  if key is not None:
   k,w=key;ks.append(k);ctx.append(w);candidate.append(group[:])
 with path.open() as f:
  print('INDEX_HEADER',f.readline().strip(),flush=True)
  for line in f:
   k,w,t,c,n=line.strip().split('\t');kv=(int(k),tuple(w.split()))
   if key!=kv:flush();key=kv;group=[]
   group.append((t,float(c)/max(1.,float(n))))
  flush()
 return ctx,ks,candidate

def main():
 t0=time.monotonic();torch.set_num_threads(2)
 pack=torch.load(R/'checkpoints/v166_lexical_birth.pt',map_location='cpu',weights_only=False)
 emb=pack['model']['ctx.weight'].detach().numpy().astype(np.float32)
 vocabulary=pack['vocab'];vid={w:i+1 for i,w in enumerate(vocabulary)}
 cloud=pack['model'];low=cloud['lower'].detach().numpy();upper=cloud['upper'].detach().numpy();plus=cloud['plus'].detach().numpy();minus=cloud['minus'].detach().numpy()
 # Lower word b -> activation over upper concept points: true S+ and S- energy.
 field=(plus*np.tanh(low[None,:,:]+upper[:,None,:])-minus*np.tanh(low[None,:,:]-upper[:,None,:])+low[None,:,:]*upper[:,None,:]/np.sqrt(8)).sum(-1).T
 # Oriented signed field projected onto its leading 16 eigenchannels.
 u,s,vt=np.linalg.svd(field,full_matrices=False)
 signed=field@vt[:16,:].T
 # Normalize per point; train-valued geometry is preserved, no new weights.
 signed=signed.astype(np.float32)
 signed/=(np.linalg.norm(signed,axis=1,keepdims=True)+1e-8)
 train,ks,targets=load_keys(R/'checkpoints/v164_native_distribution.tsv');n=len(train)
 print('INDEX',n,'keys',sum(len(x) for x in targets),'rows',time.monotonic()-t0,flush=True)
 def encode(words):
  ww=words[-3:];base=np.zeros((32,),np.float32);c=np.zeros((16,),np.float32);rec=[.2,.3,.5][3-len(ww):]
  for w,r in zip(ww,rec):
   base+=r*emb[vid.get(w,0)];c+=r*signed[fnv(w)]
  den=sum(rec);base/=den;c/=den
  return base,c
 X=np.empty((n,32),np.float32);Y=np.empty((n,16),np.float32)
 for i,w in enumerate(train):X[i],Y[i]=encode(w)
 def norm(z):return z/(np.linalg.norm(z,axis=-1,keepdims=True)+1e-9)
 X=norm(X);Y=norm(Y)
 dev=collect('dev',24,20)
 assert len(dev)==480
 Q=np.empty((480,32),np.float32);V=np.empty((480,16),np.float32)
 for i,row in enumerate(dev):Q[i],V[i]=encode(row['past'])
 Q=norm(Q);V=norm(V)
 parent=[json.loads(line) for line in (R/'results/v164_dev_rows.jsonl').open()]
 assert all(a['word']==b['truth'] for a,b in zip(parent,dev))
 P=[r['predictions']['v164_b0.6'][:32] for r in parent]
 # Preexisting v166 raw proposal from the exact same checkpoint and positions.
 _,_,predictions=evaluate(R/'checkpoints/v166_lexical_birth.pt','dev',24,20)
 G=predictions['normal']
 truth=[r['truth'] for r in dev]
 print('BASE','parent32',sum(w in p for w,p in zip(truth,P)),'lex32',sum(w in g[:32] for w,g in zip(truth,G)),flush=True)
 res={'train_keys':n,'train_rows':sum(len(x) for x in targets),'dev_positions':480,'dev_books':24,'architecture':'native trained S+/S- eigenfield 16D and learned lexical context 32D nearest neighbors','retrieval_k':32,'arms':{}}
 for alpha in [0.,.33,.66,1.]:
  # lex and signed channels independently normalized, then combined without retraining.
  out=[];closest=[];K=32
  for start in range(0,480,24):
   xx=(1-alpha)*(Q[start:start+24]@X.T)+alpha*(V[start:start+24]@Y.T)
   ii=np.argpartition(xx,-K,axis=1)[:,-K:]
   val=np.take_along_axis(xx,ii,axis=1)
   order=np.argsort(-val,axis=1)
   ii=np.take_along_axis(ii,order,axis=1);val=np.take_along_axis(val,order,axis=1)
   for ids,similarity in zip(ii,val):
    counts=collections.defaultdict(float)
    for rank,(key,sim) in enumerate(zip(ids,similarity)):
     factor=(1./(1.+rank/3.))*max(.01,float(sim))**3*([0,1.,1.2,1.5][ks[key]])
     for word,prob in targets[key]:counts[word]+=factor*prob
    pred=sorted(counts,key=lambda w:(-counts[w],w))[:96]
    out.append(pred)
    closest.append(float(similarity[0]))
  hits={k:sum(w in a[:k] for w,a in zip(truth,out)) for k in (1,5,16,32,64,96)}
  union32=sum(w in set(p)|set(g[:32])|set(a[:32]) for w,p,g,a in zip(truth,P,G,out))
  parent_plus=sum(w in set(p)|set(a[:32]) for w,p,a in zip(truth,P,out))
  res['arms'][str(alpha)]={'hits':hits,'union32_lex32_parent32':union32,'union32_parent32':parent_plus,'average_neighbor_cos':float(np.mean(closest))}
  print('ARM',alpha,json.dumps(res['arms'][str(alpha)]),'secs',round(time.monotonic()-t0,1),flush=True)
  (R/f'results/v168_signed_ann_a{str(alpha).replace(".","_")}_dev.jsonl').write_text(''.join(json.dumps({'truth':w,'candidates':a[:64]})+'\n' for w,a in zip(truth,out)))
 path=R/'results/v168_signed_ann_dev.json';res['seconds']=time.monotonic()-t0;path.write_text(json.dumps(res,indent=2));print('FINAL',json.dumps(res),flush=True)
if __name__=='__main__':main()