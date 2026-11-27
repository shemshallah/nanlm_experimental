"""v70: finite-horizon path posterior with trainable conserved allocation."""
from collections import defaultdict
import math,re
import numpy as np
import torch
from torch import nn
from trajectory69 import Sparse

def words(root):
    return tuple(x.decode('ascii').lower() for x in re.findall(rb'[A-Za-z]+',root))

def normalize(d):
    z=math.fsum(d.values())
    return {k:v/z for k,v in d.items() if v>0} if z>0 else {}

def condition(paths,observed):
    """Consume only actually observed words; drop exhausted paths."""
    evidence=math.fsum(v for p,v in paths.items() if p[0]==observed)
    out=defaultdict(float)
    for p,v in paths.items():
        if p[0]==observed and len(p)>1:out[p[1:]]+=v
    return normalize(out),evidence

def combine(a,b,weight):
    if not a:return dict(b)
    if not b:return dict(a)
    out=defaultdict(float)
    for p,v in a.items():out[p]+=(1-weight)*v
    for p,v in b.items():out[p]+=weight*v
    return normalize(out)

def marginal(paths):
    if not paths:return None
    d=defaultdict(float)
    for p,v in paths.items():d[p[0]]+=v
    return Sparse(d)

class PathField:
    """Read-only reference path bank; per-document runtime owns the posterior."""
    def __init__(self,vocab,code,docs):
        self.vocab=tuple(vocab);self.ids={w:i for i,w in enumerate(vocab)}
        x=np.asarray(code,np.float32)
        self.code=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
        self.code[0]=0
        self.valid=np.array([bool(re.fullmatch('[a-z]{1,63}',w)) for w in vocab])
        chunks=[]
        for doc in docs:
            ids=np.array([self.ids.get(w,0) for w in doc],np.int32)
            if len(ids)<6:continue
            p=np.lib.stride_tricks.sliding_window_view(ids,6)
            chunks.append(p[(p[:,2]>0)&self.valid[p[:,3:]].all(1)].copy())
        p=np.concatenate(chunks);self.paths=p[np.argsort(p[:,2],kind='stable')]
        self.offset=np.r_[0,np.cumsum(np.bincount(self.paths[:,2],minlength=len(vocab)))]
        self.paths.setflags(write=False);self.code.setflags(write=False)
    def retrieve(self,h,wrong=False):
        if len(h)<3:return {}
        q=np.array([self.ids.get(w,0) for w in h[-3:]],np.int32)
        if wrong:q=q[[1,0,2]]
        if not q[-1]:return {}
        lo,hi=self.offset[q[-1]:q[-1]+2];p=self.paths[lo:hi]
        if not len(p):return {}
        # Exact last-word addresses, preserve exact two/three-word support.
        match=np.flatnonzero((p[:,1]==q[1])&(q[1]>0))
        bg=np.flatnonzero((p[:,1]!=q[1])|(q[1]==0))
        hist=[self.ids.get(w,0) for w in h[-16:]]
        topic=self.code[hist].mean(0);topic/=max(float(np.linalg.norm(topic)),1e-12)
        out=defaultdict(float)
        for ids,cap in ((match,1024),(bg,512)):
            if not len(ids):continue
            take=ids[np.linspace(0,len(ids)-1,min(cap,len(ids)),dtype=int)]
            a=p[take];eq=(a[:,:3]==q)&(q>0)
            exact=eq@np.array([.15,.35,.5])
            signed=sum(k*(self.code[a[:,j]]@self.code[q[j]]) for j,k in enumerate((.15,.35,.5)))
            future=self.code[a[:,3:]].mean(1)
            coherence=(future@topic)/np.maximum(np.linalg.norm(future,axis=1),1e-12)
            v=(exact**7+.04*np.maximum(signed-.25,0.)**3)*np.exp(np.clip(coherence,-1,1))
            v*=min(32.,len(ids)/len(take))
            for seq,w in zip(a[:,3:],v):
                if w>0:out[tuple(self.vocab[i] for i in seq)]+=float(w)
        return normalize(dict(sorted(out.items(),key=lambda kv:(-kv[1],kv[0]))[:384]))

class Session:
    def __init__(self,bank,wrong=False):
        self.bank=bank;self.wrong=wrong;self.reset()
    def reset(self):
        self.h=();self.paths={};self.carried={};self.direct={};self.evidence=0.
    def predict(self,root):
        h=words(root)
        assert h[:len(self.h)]==self.h
        if h==self.h and self.direct:return marginal(self.direct),marginal(self.carried),self.evidence
        # Warm up only through observed words; each iteration predicts the
        # next word after the current prefix, then observes it on the next step.
        start=len(self.h)
        if start==0:start=max(0,len(h)-24)
        for j in range(start,len(h)):
            self.carried,self.evidence=condition(self.paths,h[j])
            self.direct=self.bank.retrieve(h[:j+1],self.wrong)
            self.paths=combine(self.direct,self.carried,.5)
        self.h=h
        return marginal(self.direct),marginal(self.carried),self.evidence

def features(root,base,direct,carry,evidence):
    def peak(x):return max(x.p.values(),default=0.) if x else 0.
    def count(x):return math.log1p(len(x.p))/6 if x else 0.
    def agree(x):return math.fsum(v*base.complete(w) for w,v in x.p.items()) if x else 0.
    return np.array([1.,float(root[-1:]==b' '),float(root[-1:] in (b'\n',b'\r')),peak(direct),peak(carry),count(direct),count(carry),agree(direct),agree(carry),evidence],np.float64)

class Allocation(nn.Module):
    """Small new optimizer group; exact parent rollback at scale zero."""
    def __init__(self):
        super().__init__();self.theta=nn.Parameter(torch.zeros(2,10,dtype=torch.float64))
        with torch.no_grad():self.theta[:,0]=-2.
    def weights(self,f,available,scale=1.):
        z=f@self.theta.T
        z=z.masked_fill(~available,-torch.inf)
        a=torch.cat((torch.zeros_like(z[...,:1]),z),-1).softmax(-1)
        b=a[...,1:]*scale
        return torch.cat((1.-b.sum(-1,keepdim=True),b),-1)

def selftest():
    v=['?','a','b','c','d','e','f','g'];docs=[list('abcdeabcde'),list('abcfgeabcfge')]
    b=PathField(v,np.eye(len(v)),docs);s=Session(b)
    d,c,e=s.predict(b'a b c ');assert d.complete('d')>0 and d.complete('f')>0
    d,c,e=s.predict(b'a b c d ');assert c.complete('e')>0 and e>0
    d2,c2,e2=s.predict(b'a b c d ');assert d.p==d2.p and c.p==c2.p and e==e2
    s.reset();d3,c3,e3=s.predict(b'a b c d ');assert d.p==d3.p and c.p==c3.p
    a=Allocation();f=torch.ones(5,10,dtype=torch.float64);avail=torch.tensor([[True,True],[True,False],[False,True],[False,False],[True,True]])
    w=a.weights(f,avail);assert torch.allclose(w.sum(-1),torch.ones(5,dtype=torch.float64))
    assert torch.equal(a.weights(f,avail,0.)[:,0],torch.ones(5,dtype=torch.float64))
    assert (w[:,1:][~avail]==0).all()
    (-w[0,1].log()).backward();assert a.theta.grad.abs().sum()>0
    print('v70 path posterior, causal replay, masking, rollback and gradient preflight passed')
if __name__=='__main__':selftest()
