"""v76: collision-free suffix and semantic-class context hierarchy."""
from collections import defaultdict,Counter
from functools import lru_cache
import math
import numpy as np
import torch
from torch import nn
from field70 import words
from trajectory69 import Sparse

class Suffix:
    """Lexicographic index of reversed contexts. Exact integer keys, no hashes."""
    def __init__(self,seqs,order=8,classes=None):
        self.order=order;parts=[];positions=[];off=0
        for s in seqs:
            s=np.asarray(s,np.int32)
            if len(s)<2:continue
            p=np.r_[np.zeros(order,np.int32),s]
            parts.append(p);positions.append(np.arange(off+order+1,off+len(p),dtype=np.int32));off+=len(p)
        self.flat=np.concatenate(parts);pos=np.concatenate(positions)
        self.address=self.flat if classes is None else np.asarray(classes,dtype=np.int32)[self.flat]
        self.pos=pos[np.lexsort(tuple(self.address[pos-k] for k in range(order,0,-1)))]
        self.flat.setflags(write=False);self.address.setflags(write=False);self.pos.setflags(write=False)
    @lru_cache(maxsize=4096)
    def bounds(self,h):
        if not h:return 0,len(self.pos)
        lo,hi=self.bounds(h[1:]);k=len(h);v=h[0]
        if not v or lo==hi:return lo,lo
        a=self.address[self.pos[lo:hi]-k]
        l=int(np.searchsorted(a,v,'left'));r=int(np.searchsorted(a,v,'right'))
        return lo+l,lo+r
    @lru_cache(maxsize=4096)
    def counts(self,h):
        lo,hi=self.bounds(h)
        ids,c=np.unique(self.flat[self.pos[lo:hi]],return_counts=True)
        return ids,c
    def query(self,h):
        out=[]
        for k in range(1,min(len(h),self.order)+1):
            key=tuple(h[-k:])
            if not all(key):break
            ids,c=self.counts(key)
            if not len(ids):break
            out.append((k,ids,c))
        return out

def hierarchy(levels,vocab,discount=.65):
    # A lower context is retained only via unallocated escape mass.
    # Condition the sparse donor on the union of represented continuations.
    q={};support=0.;depth=0
    for k,ids,counts in levels:
        keep=np.array([int(i)>0 and 0<len(vocab[int(i)])<=63 for i in ids])
        ids=ids[keep];counts=np.asarray(counts[keep],np.float64)
        if not len(ids):continue
        n=float(counts.sum());mass=np.maximum(counts-discount,0)/n
        residual=1-float(mass.sum())
        q={i:v*residual for i,v in q.items()}
        for i,v in zip(ids,mass):q[int(i)]=q.get(int(i),0.)+float(v)
        support=n;depth=k
    if not q:return None,0.,0
    q={vocab[i]:v for i,v in q.items() if v>0}
    return Sparse(q),support,depth

class ContextField:
    def __init__(self,vocab,seqs,classes):
        self.vocab=tuple(vocab);self.ids={w:i for i,w in enumerate(vocab) if i}
        self.classes=np.asarray(classes,np.int32).copy()
        self.classes[1:]+=1;self.classes[0]=0
        self.exact=Suffix(seqs,8);self.semantic=Suffix(seqs,6,self.classes)
    def query(self,prefix,wrong=False):
        h=words(prefix)[-1024:];ids=np.array([self.ids.get(w,0) for w in h],np.int32)
        query=ids.copy()
        if wrong and len(query)>1:query[:-1]=query[:-1][::-1]
        e=hierarchy(self.exact.query(tuple(map(int,query[-8:]))),self.vocab)
        s=hierarchy(self.semantic.query(tuple(map(int,self.classes[query[-6:]]))),self.vocab)
        # Match only observed antecedents and their already observed successors.
        local=[];n=len(ids)
        for k in range(1,min(8,n-1)+1):
            q=query[-k:]
            if not q.all():break
            ix=np.arange(k,n)
            ok=np.ones(len(ix),bool)
            for j in range(k):ok&=ids[ix-k+j]==q[j]
            ix=ix[ok]
            if not len(ix):break
            vals,c=np.unique(ids[ix],return_counts=True)
            local.append((k,vals,c))
        l=hierarchy(local,self.vocab,discount=.35)
        # Recency distribution is separate from suffix matches; only seen words.
        rec=defaultdict(float)
        for i,w in enumerate(h):
            if w and len(w)<=63:rec[w]+=math.exp(-(n-1-i)/128.)
        t=(Sparse(rec),sum(rec.values()),1) if rec else (None,0.,0)
        return (e,s,l,t)

def feature(laws,a,b):
    feat=[];mask=[]
    for law,n,k in laws:
        mask.append(law is not None)
        if law is None:feat.append([0.]*8);continue
        p=np.array(list(law.p.values()))
        agree=math.fsum(v*.5*(a.complete(w)+b.complete(w)) for w,v in law.p.items())
        feat.append([1.,math.log1p(n)/10.,k/8.,math.log1p(len(p))/10.,float(p.max()),float(-(p*np.log(p)).sum()/10.),math.log1p(100*agree)/5.,float(n/(n+len(p)+8))])
    return np.asarray(feat,np.float64),np.asarray(mask,bool)

class Gate(nn.Module):
    def __init__(self):
        super().__init__();self.theta=nn.Parameter(torch.zeros(4,8,dtype=torch.float64))
        with torch.no_grad():self.theta[:,0]=-2.5
    def forward(self,f,mask,scale=1.):
        assert 0<=scale<=1
        z=(f*self.theta).sum(-1).masked_fill(~mask,-torch.inf)
        p=torch.cat((torch.zeros_like(z[...,:1]),z),-1).softmax(-1)
        g=p[...,1:]*scale
        return torch.cat((1-g.sum(-1,keepdim=True),g),-1)

def selftest():
    vocab=['','a','b','c','d','e','f'];docs=[np.array([1,2,3,4,1,2,3,5]),np.array([1,2,3,6,1,2,4])]
    c=ContextField(vocab,docs,[0,1,1,2,3,3,3])
    for query in [(1,),(1,2),(1,2,3),(2,3),(3,4,1),(6,1,2),(2,5),(0,1)]:
        for k,ids,counts in c.exact.query(query):
            q=query[-k:];expected=Counter(int(d[j]) for d in docs for j in range(k,len(d)) if tuple(d[j-k:j])==q)
            assert dict(zip(ids,counts))==expected,(q,ids,counts,expected)
    rng=np.random.default_rng(76)
    synthetic=[rng.integers(1,7,size=100) for _ in range(7)]
    idx=Suffix(synthetic,8)
    for _ in range(200):
        doc=synthetic[int(rng.integers(len(synthetic)))]
        j=int(rng.integers(8,len(doc)));k=int(rng.integers(1,9))
        key=tuple(map(int,doc[j-k:j]))
        expected=Counter(int(d[t]) for d in synthetic for t in range(k,len(d)) if tuple(d[t-k:t])==key)
        assert dict(zip(*idx.counts(key)))==expected
    assert dict(zip(*c.exact.counts((1,2,3))))=={4:1,5:1,6:1}
    assert dict(zip(*c.exact.counts((5,1))))=={} # no crossing books
    laws=c.query(b'a b c d a b c ')
    for law,n,k in laws:
        if law:
            assert abs(sum(law.p.values())-1)<1e-12
            for p in ('','a','b','c','d','e','f'):
                assert abs(law.mass(p)-law.complete(p)-sum(law.mass(p+x) for x in 'abcdefghijklmnopqrstuvwxyz'))<1e-12
    assert laws[2][0].complete('d')>0
    gate=Gate();f,mask=feature(laws,laws[0][0],laws[0][0])
    p=gate(torch.from_numpy(f),torch.from_numpy(mask))
    assert abs(float(p.sum().detach())-1)<1e-12
    assert torch.equal(gate(torch.from_numpy(f),torch.from_numpy(mask),0.),torch.tensor([1.,0.,0.,0.,0.],dtype=torch.float64))
    (-p[1].log()).backward();assert gate.theta.grad.abs().sum()>0
    print('v76 exact suffix counts, document boundary, causal local recurrence, normalized gate and rollback passed')
if __name__=='__main__':selftest()
