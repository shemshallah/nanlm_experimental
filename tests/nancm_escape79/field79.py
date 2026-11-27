"""v79: hierarchical escape allocation over extended TRAIN-only context counts."""
import math
from functools import lru_cache
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from field76 import Suffix
from trajectory69 import Sparse
from field70 import words

N=10 # four semantic levels, six exact levels
class Evidence:
    def __init__(self,index,seqs,classes):
        self.ids=index.ids;self.vocab=index.words
        self.classes=np.asarray(classes,np.int32).copy()
        self.classes[1:]+=1;self.classes[0]=0
        self.exact=Suffix(seqs,6)
        self.semantic=Suffix(seqs,4,self.classes)
    @lru_cache(maxsize=384)
    def donor(self,semantic,key):
        idx=self.semantic if semantic else self.exact
        ids,c=idx.counts(key)
        if not len(ids):return None,np.zeros(6),0.
        keep=np.array([int(i)>0 and 0<len(self.vocab[int(i)])<=63 for i in ids])
        ids=ids[keep];c=c[keep].astype(np.float64)
        if not len(c):return None,np.zeros(6),0.
        n=float(c.sum());types=len(c)
        # Reserve probability for inherited and lower-order predictions.
        # The bounded discount distinguishes singleton and repeated evidence.
        discount=np.where(c==1,.75,np.where(c==2,1.,1.25))
        retained=np.maximum(c-discount,0.)/n
        mass=float(retained.sum())
        law=Sparse({self.vocab[int(i)]:float(v) for i,v in zip(ids,retained)})
        p=np.array(list(law.p.values()))
        feat=np.array([1.,math.log1p(n)/12.,len(key)/6.,math.log1p(types)/10.,float(p.max()),math.log1p(n/types)/5.])
        return law,feat,mass
    def query(self,prefix,wrong=False):
        h=[self.ids.get(w,0) for w in words(prefix)[-6:]]
        if wrong and len(h)>1:h=h[:-1][::-1]+h[-1:]
        a=np.asarray(h,np.int32);out=[]
        for semantic,order in ((True,4),(False,6)):
            seq=self.classes[a] if semantic else a
            for k in range(1,order+1):
                key=tuple(map(int,seq[-k:]))
                out.append(self.donor(semantic,key) if len(key)==k and all(key) else (None,np.zeros(6),0.))
        return out

class Escape(nn.Module):
    def __init__(self):
        super().__init__();self.theta=nn.Parameter(torch.zeros(N,6,dtype=torch.float64))
        with torch.no_grad():self.theta[:,0]=-1.5
    def forward(self,f,cap,scale=1.):
        return scale*cap*torch.sigmoid((f*self.theta).sum(-1))
    def allocate(self,f,cap,scale=1.):
        # Ordered evidence: high-order context can use only its retained mass.
        g=self(f,cap,scale)
        remaining=torch.cumprod((1-g).flip(-1),dim=-1).flip(-1)
        return torch.cat((remaining[...,:1],g[...,:-1]*remaining[...,1:],g[...,-1:]),-1)

def compact(base,laws,weights):
    # Flatten sparse branches, leaving the exact inherited escape distribution.
    from open_word import fuse
    if not any(weights[1:]):return base
    q=defaultdict(float)
    for (law,_,_),weight in zip(laws,weights[1:]):
        if weight:
            assert law is not None
            for w,p in law.p.items():q[w]+=float(weight)*p
    mass=math.fsum(q.values())
    assert abs(mass+weights[0]-1)<1e-10
    return fuse(base,Sparse(q),min(1.,mass))

def selftest():
    import types
    index=types.SimpleNamespace(words=['','a','b','c','d'],ids={'a':1,'b':2,'c':3,'d':4})
    e=Evidence(index,[np.array([1,2,3,1,2,3,1,2,4]),np.array([1,2,3,1,2,3])],[0,1,2,3,3])
    laws=e.query(b'a b ');m=Escape()
    f=torch.tensor(np.stack([x[1] for x in laws]));c=torch.tensor([x[2] for x in laws],dtype=torch.float64)
    w=m.allocate(f,c);assert abs(float(w.sum().detach())-1)<1e-12
    base=Sparse({'a':1,'b':1,'c':1,'d':1})
    q=compact(base,laws,w.detach().numpy())
    assert compact(base,laws,m.allocate(f,c,0.).detach().numpy()) is base
    for word in index.words[1:]:
        expected=float(w[0].detach())*base.complete(word)+sum(float(a.detach())*l.complete(word) for a,(l,_,_) in zip(w[1:],laws) if l)
        assert abs(q.complete(word)-expected)<1e-12
    assert q.complete('c')>base.complete('c')
    (-w[0].log()).backward();assert m.theta.grad.abs().sum()>0
    print('v79 ordered escape, complete mass, zero rollback and gradient preflight passed')
if __name__=='__main__':selftest()
