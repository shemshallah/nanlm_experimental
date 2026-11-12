#!/usr/bin/env python3
"""One shared PAL7 rank-64 decoder + PAL9 causal 2048-word CountSketch adapter.
No future tokens, reverse prediction branch, or extra next-token decoder.
"""
import argparse,re,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from infer_unified_now import load,hidden,topk,RGX
D=96;N=72

def context_mix(x):
 x&=0xffffffff;x^=x>>16;x=(x*0x7feb352d)&0xffffffff
 x^=x>>15;x=(x*0x846ca68b)&0xffffffff;x^=x>>16
 return x

def context_features(history):
 # Causal 2047 previous observed *whole word tokens*: lags 1..3 are native PAL7.
 feat=np.zeros(N,dtype=np.int8)
 for band in range(9):
  lo=4<<band;hi=(8<<band)-1
  if len(history)<lo:continue
  counts=[0]*8
  for tok in history[-hi:-lo+1] if len(history)>=hi else history[:-lo+1]:
   x=context_mix(int(tok)^((0x9e3779b9*(band+1))&0xffffffff))
   counts[x&7]+=-1 if x&8 else 1
  norm=16.0/(hi-lo+1)**.5
  for j,c in enumerate(counts):feat[band*8+j]=np.int8(max(-127,min(127,int(np.sign(c*norm)*np.floor(abs(c*norm)+.5)))))
 return feat

def read_long(path,updates,vocab_size):
 raw=path.read_bytes();h=np.frombuffer(raw,dtype='<i4',count=6)
 if list(h)!=[0x50414c39,vocab_size,D,64,N,updates]:raise ValueError(f'PAL9 checkpoint mismatch: {h}')
 a=np.frombuffer(raw,dtype='<f4',offset=24)
 if len(a)!=2*D*N:raise ValueError('PAL9 tensor size mismatch')
 return a[:D*N].reshape(D,N)

def hidden_long(m,long_weights,hist,ablate=False):
 past=([1,1,1]+list(hist))[-3:]
 base=hidden(m,past)
 if ablate:return base
 z=np.arctanh(np.clip(base,-.999999,.999999))
 cv=context_features(hist).astype(np.float32)/32.
 z+=long_weights@cv
 return np.tanh(z).astype(np.float32)

def generate(m,long_weights,prompt,n=70,seed=13,temp=.85,k=40,ablate=False):
 rng=np.random.default_rng(seed)
 prompt_tokens=RGX.findall(prompt.lower())
 hist=[m['lookup'].get(w,0) for w in prompt_tokens]
 out=[];recent=[]
 for _ in range(n):
  h=hidden_long(m,long_weights,hist,ablate)
  c=[(i,s) for i,s in topk(m,h,max(60,k)+3) if i>1][:k]
  logits=np.array([s for i,s in c],dtype=np.float64)/temp
  for j,(i,_) in enumerate(c):
   if i in recent[-16:] and i>6:logits[j]-=.7
  logits-=max(logits);probs=np.exp(logits);probs/=probs.sum()
  y=c[int(rng.choice(len(c),p=probs))][0]
  out.append(m['vocab'][y]);recent.append(y);hist.append(y)
 return re.sub(r'\s+([.,!?;:])',r'\1',' '.join(out))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--context',type=Path,required=True);p.add_argument('--prompt',default='she looked at');p.add_argument('--prompt-file',type=Path);p.add_argument('--prompt-words',type=int,default=2048);p.add_argument('--n',type=int,default=75);p.add_argument('--seed',type=int,default=20261111);p.add_argument('--temp',type=float,default=.85);p.add_argument('--top-k',type=int,default=40);p.add_argument('--ablate',action='store_true')
 a=p.parse_args();m=load(a.weights,a.root);L=read_long(a.context,m['updates'],len(m['vocab']))
 prompt=a.prompt if not a.prompt_file else ' '.join(RGX.findall(a.prompt_file.read_text(encoding='utf-8',errors='replace'))[:a.prompt_words])
 print('checkpoint_steps',m['updates'],'prompt_tokens',len(RGX.findall(prompt)),'long_context',not a.ablate)
 print(generate(m,L,prompt,a.n,a.seed,a.temp,a.top_k,a.ablate))