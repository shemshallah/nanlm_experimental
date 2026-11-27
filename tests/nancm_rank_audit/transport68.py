"""v68: causal signed transport, exact sparse-to-open probability bridge."""
import math,re
from collections import defaultdict
import numpy as np

class Sparse:
    """A complete word law. Prefix mass is a sum, never a subtraction."""
    def __init__(self, scores):
        z=math.fsum(scores.values())
        assert z>0 and all(v>0 and re.fullmatch('[a-z]{1,63}',w) for w,v in scores.items())
        self.p={w:float(v/z) for w,v in scores.items()}
        self.prefix=defaultdict(float)
        for w,v in self.p.items():
            for i in range(len(w)+1):self.prefix[w[:i]]+=v
    def mass(self,p):return self.prefix.get(p,0.)
    def complete(self,w):return self.p.get(w,0.)

class Transport:
    """Past-only transition graph in inherited signed coordinates.
    A successor can vote only if it has already been observed at the event root.
    Exact context edges and signed geometrical edges share the same target law.
    """
    def __init__(self,words,code):
        x=np.asarray(code,dtype=np.float64)
        assert x.ndim==2 and len(x)==len(words)
        self.ids={w:i for i,w in enumerate(words)}
        self.code=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    def propose(self,root,parent,other,raw0,mode='full'):
        h=[w.decode('ascii').lower() for w in re.findall(rb'[A-Za-z]+',root)][-1024:]
        if len(h)<5:return None,0.,{}
        # Same word multiset and final two words, wrong temporal addresses.
        if mode=='wrong':h=h[:-2][::-1]+h[-2:]
        ids=np.array([self.ids.get(w,0) for w in h],dtype=int)
        a=np.arange(2,len(h));valid=np.array([bool(re.fullmatch('[a-z]{1,63}',h[j])) for j in a])
        # Query uses previous two observed words, never the next target.
        s=np.zeros(len(a));exact=np.zeros(len(a));negative=np.zeros(len(a))
        for d,k in ((1,0.7),(2,0.3)):
            eq=np.array([h[j-d]==h[-d] for j in a])
            exact+=k*eq
            c=self.code[ids[a-d]]@self.code[ids[-d]]
            c=np.where((ids[a-d]>0)&(ids[-d]>0),np.clip(c,-1,1),0.)
            s+=k*np.maximum(c,0.);negative+=k*np.maximum(-c,0.)
        geom=np.maximum(0.,s-negative)
        if mode=='exact':geom=np.zeros_like(geom)
        # Reinforce matched edges, admit near-addresses only at substantial agreement.
        strength=(exact**2+0.5*np.maximum(geom-.35,0.)**2)*valid
        strength*=np.exp(-(len(h)-1-a)/512.)
        best=np.argsort(-strength,kind='stable')[:48]
        votes=defaultdict(float)
        for j in best:
            if strength[j]>1e-12:votes[h[a[j]]]+=float(strength[j])
        if not votes:return None,0.,{}
        # Reciprocal support from the native recurrent lexical, contrastive and
        # raw-byte root laws. Bounded multipliers retain new/OOV candidates.
        scores={}
        for w,v in votes.items():
            b=ord(w[0]);p=parent.complete(w);q=other.complete(w)
            corroboration=(1.+min(4.,math.sqrt(max(0.,p*q))*500.))
            first=1.+min(2.,float(raw0[b]+raw0[b-32])*8.)
            scores[w]=v*corroboration*first
        # Reliability decays when address evidence is weak; calibration sets
        # the maximum transfer share. Identical zero-gain rollback is possible.
        confidence=min(1.,float(strength[best].sum())/2.)
        return Sparse(scores),confidence,dict(candidates=len(scores),exact_edges=int(np.count_nonzero(exact>=.999)),max_signed=float(geom.max()))

def selftest():
    s=Sparse({'a':2.,'ab':3.,'b':5.})
    for p in ('','a','ab','b','z'):
        assert abs(s.mass(p)-s.complete(p)-sum(s.mass(p+c) for c in 'abcdefghijklmnopqrstuvwxyz'))<1e-14
    assert s.mass('')==1. and s.complete('ab')==.3
    words=['?','a','b','c','d','e'];code=np.eye(len(words))
    t=Transport(words,code);q=np.ones(256)/256
    base=Sparse({'a':1.,'b':1.,'c':1.,'d':1.,'e':1.})
    x,conf,audit=t.propose(b'a b c a b c a b ',base,base,q)
    assert x.complete('c')>0 and conf>0 and audit['exact_edges']>=2
    assert t.propose(b'a b ',base,base,q)[0] is None
    # Same root always yields same law, no mutable history or target argument.
    y,_,_=t.propose(b'a b c a b c a b ',base,base,q)
    assert x.p==y.p
    print('v68 sparse conservation and causal signed-transport preflight passed')

if __name__=='__main__':selftest()
