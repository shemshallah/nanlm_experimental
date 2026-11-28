"""v91 multiscale associative retrieval: a causal concept-to-word field.

Build directed word associations on TRAIN only. Retrieve evidence from
nonadjacent observed words using distance-dependent predictive likelihood.
No target is used when constructing the candidate distribution.
"""
from collections import defaultdict
import math
import numpy as np
import torch
from torch import nn
from field70 import words
from trajectory69 import Sparse

class Association:
    def __init__(self,seqs,vocab,prior,limit=192):
        self.vocab=tuple(vocab);self.ids={w:i for i,w in enumerate(vocab) if i}
        self.prior=np.asarray(prior,np.float64);self.limit=limit
        V=len(vocab);self.tables=[]
        # Distinct causal bands, not symmetric co-occurrence.
        # Aggregate collision-free source/target pairs; no cross-book edges.
        for offsets in ((1,2),(3,4,5,6),(7,9,12,16),(20,28,40,56)):
            chunks=[]
            for seq in seqs:
                a=np.asarray(seq,np.int64)
                for d in offsets:
                    if len(a)<=d:continue
                    x=a[:-d];y=a[d:];ok=(x>0)&(y>0)
                    chunks.append(x[ok]*V+y[ok])
            pairs=np.concatenate(chunks);del chunks
            keys,c=np.unique(pairs,return_counts=True);del pairs
            source=keys//V;target=keys%V
            off=np.r_[0,np.cumsum(np.bincount(source,minlength=V))]
            table={}
            for i in range(1,V):
                lo,hi=off[i:i+2]
                if lo==hi:continue
                t=target[lo:hi];n=c[lo:hi].astype(np.float64);total=float(n.sum())
                # Discount singleton evidence; rank by probability lift with
                # tempered prior so rare noise cannot dominate.
                p=np.maximum(n-.5,0.)/total
                score=p/np.maximum(self.prior[t],1e-12)**.35
                k=min(limit,len(t));ix=np.argpartition(-score,k-1)[:k]
                order=np.lexsort((t[ix],-score[ix]));ix=ix[order]
                table[i]=(t[ix].astype(np.int32),p[ix],total)
            self.tables.append(table)
    def query(self,prefix,wrong=False):
        h=[self.ids.get(w,0) for w in words(prefix)[-64:]]
        if wrong and len(h)>4:h=h[:-4][::-1]+h[-4:]
        scores=defaultdict(float);agreement=defaultdict(set);support=0.
        # Match training distance to distance from prediction boundary.
        # Suppress repeated source words so a repeated topic cannot monopolize mass.
        seen=set()
        for j,i in enumerate(reversed(h),1):
            if not i or i in seen:continue
            seen.add(i)
            band=0 if j<=2 else 1 if j<=6 else 2 if j<=16 else 3
            item=self.tables[band].get(i)
            if item is None:continue
            t,p,n=item;w=(n/(n+32.))*math.exp(-j/24.)
            support+=w
            for k,v in zip(t,p):
                k=int(k)
                if len(self.vocab[k])<=63:
                    scores[k]+=w*float(v)
                    agreement[k].add(i)
        if not scores:return None,np.zeros(6,np.float64)
        # Multiplicative agreement across distinct observed anchors.
        for k in scores:scores[k]*=1.+math.log1p(len(agreement[k])-1)
        top=sorted(scores,key=lambda k:(-scores[k]/max(self.prior[k],1e-12)**.2,k))[:384]
        law=Sparse({self.vocab[k]:scores[k]/max(self.prior[k],1e-12)**.2 for k in top})
        p=np.array(list(law.p.values()))
        f=np.array([1.,math.log1p(support)/4.,math.log1p(len(seen))/5.,p.max(),-float(p@np.log(p))/8.,max(len(agreement[k]) for k in top)/16.],np.float64)
        return law,f

class Gate91(nn.Module):
    def __init__(self):
        super().__init__();self.theta=nn.Parameter(torch.zeros(6,dtype=torch.float64))
        with torch.no_grad():self.theta[0]=-2.
    def forward(self,f,scale=1.):
        return scale*torch.sigmoid((f*self.theta).sum(-1))*(f[...,0]>0)

def selftest():
    from open_word import fuse
    vocab=['','a','b','c','d','e','f']
    seqs=[np.array([1,2,3,4,5,6]*50),np.array([1,2,3,4,5,6]*30)]
    m=Association(seqs,vocab,np.ones(7)/7,limit=6)
    law,f=m.query(b'a b c ');assert law is not None and abs(sum(law.p.values())-1)<1e-12
    assert law.complete('d')>law.complete('a')
    assert m.query(b'a b c d')[0].p==law.p # incomplete final word is not observed
    assert m.query(b'')[0] is None
    gate=Gate91();v=gate(torch.from_numpy(f));assert 0<v<1
    (-v.log()).backward();assert gate.theta.grad.abs().sum()>0
    assert float(gate(torch.from_numpy(f),0.))==0.
    assert fuse(law,law,0.) is law
    print('v91 directed multiscale associations, causal boundary, normalization and gate gradient passed',flush=True)
if __name__=='__main__':selftest()
