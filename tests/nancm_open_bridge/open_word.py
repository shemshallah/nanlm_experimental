"""Lift categorical known-word + unknown law to exact open vocabulary."""
import bisect, math
from collections import Counter, defaultdict
import numpy as np
A='abcdefghijklmnopqrstuvwxyz'
class Spelling:
    def __init__(self,docs):
        self.counts=defaultdict(Counter)
        for doc in docs:
            for w in doc:
                h=''
                for c in w+' ':
                    self.counts[h[-2:]][c]+=1; h+=c
        self.cache={}
    def next(self,p):
        key=p[-2:];z=self.cache.get(key)
        if z is None:
            alphabet=A+' ' if p else A
            c=self.counts.get(key,{});den=sum(c.get(x,0)+.25 for x in alphabet)
            z={x:(c.get(x,0)+.25)/den for x in alphabet};self.cache[key]=z
        return z
    def mass(self,p):
        q=1.
        for i,c in enumerate(p):q*=self.next(p[:i])[c]
        return q
    def complete(self,w):return self.mass(w)*self.next(w)[' '] if w else 0.
class Outside:
    """Condition a static spelling law on NOT being a known word."""
    def __init__(self,words,base):
        self.words=tuple(words);self.known=set(words);self.base=base
        assert self.words==tuple(sorted(self.known)) and all(w and all(c in A for c in w) for w in words)
        children=defaultdict(set)
        for w in words:
            for i,c in enumerate(w):children[w[:i]].add(c)
            children[w]
        self.survival={}
        for p,cc in sorted(children.items(),key=lambda kv:-len(kv[0])):
            q=base.next(p)
            self.survival[p]=math.fsum(v*(self.survival[p+c] if c in cc else 1.) for c,v in q.items() if c!=' ' or p not in self.known)
        self.z=self.survival.get('',1.);assert 0<self.z<=1+1e-12
    def mass(self,p):return self.base.mass(p)*self.survival.get(p,1.)/self.z
    def complete(self,w):return 0. if w in self.known else self.base.complete(w)/self.z
class Lift:
    def __init__(self,outside,p):
        self.outside=outside;self.words=outside.words;p=np.asarray(p,np.float64)
        assert p.shape==(len(self.words)+1,) and np.isfinite(p).all() and p.min()>=0 and abs(math.fsum(p)-1)<1e-8
        p=p/math.fsum(p);self.unknown=float(p[0]);self.p=p[1:].copy();self.cum=np.r_[0.,np.cumsum(self.p)]
    def complete(self,w):
        i=bisect.bisect_left(self.words,w)
        return float(self.p[i]) if i<len(self.words) and self.words[i]==w else self.unknown*self.outside.complete(w)
    def mass(self,p):
        if not p:return 1.
        a=bisect.bisect_left(self.words,p);b=bisect.bisect_left(self.words,p+'{');m=float(self.cum[b]-self.cum[a])
        if m<1e-10:m=math.fsum(self.p[a:b])
        return m+self.unknown*self.outside.mass(p)
class Blend:
    def __init__(self,parent,child,g):self.parent=parent;self.child=child;self.g=float(g)
    def mass(self,p):return (1-self.g)*self.parent.mass(p)+self.g*self.child.mass(p)
    def complete(self,w):return (1-self.g)*self.parent.complete(w)+self.g*self.child.complete(w)
def fuse(parent,child,g):
    assert 0<=g<=1
    return parent if g==0 else child if g==1 else Blend(parent,child,g)
