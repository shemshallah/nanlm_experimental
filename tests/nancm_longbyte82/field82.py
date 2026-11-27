"""v82 finite-state long-context byte law, exact case-folded word marginal."""
from functools import lru_cache
import math,itertools
import numpy as np
import folded_event as FE
from field76 import Suffix

class LongByte:
    def __init__(self,docs,order=12):
        self.order=order
        seq=[np.frombuffer(d,dtype=np.uint8).astype(np.int32)+1 for d in docs if len(d)>1]
        self.idx=Suffix(seq,order)
        counts=np.zeros(256,np.float64)
        for d in seq:counts+=np.bincount(d-1,minlength=256)
        self.prior=(counts+.5)/(counts.sum()+128.)
    @lru_cache(maxsize=65536)
    def canonical(self,h):
        h=h[-self.order:]
        for i in range(len(h)):
            lo,hi=self.idx.bounds(tuple(b+1 for b in h[i:]))
            if lo<hi:return h[i:]
        return b''
    @lru_cache(maxsize=32768)
    def prob(self,h):
        h=self.canonical(h);q=self.prior.copy()
        for k in range(1,len(h)+1):
            ids,c=self.idx.counts(tuple(b+1 for b in h[-k:]))
            if not len(c):continue
            n=float(c.sum());discount=np.where(c==1,.75,np.where(c==2,1.,1.25))
            mass=np.maximum(c-discount,0.)/n
            q*=1-float(mass.sum());q[ids-1]+=mass
        assert q.min()>0 and abs(q.sum()-1)<1e-10
        q.setflags(write=False)
        return q
    def support(self,h):
        s=self.canonical(h[-self.order:])
        if not s:return 0,0
        lo,hi=self.idx.bounds(tuple(b+1 for b in s))
        return len(s),hi-lo

class LongEvent(FE.Event):
    def __init__(self,word,model,root,surface,weight,cap=64):
        self.model=model
        # The base class's order limit is irrelevant: canonical() provides
        # the actual finite-state address.
        super().__init__(word,model.prob,root,surface,weight,order=3,cap=cap)
    def suffix(self,p):return self.model.canonical(bytes(p[-self.model.order:]))

class MixedEvent:
    """Mix raw donors at event level, preserving the shared lexical law."""
    def __init__(self,word,raw,long,root,surface,weight,gain):
        assert 0<gain<=1
        self.a=FE.Event(word,raw,root,surface,weight)
        self.b=LongEvent(word,long,root,surface,weight)
        self.gain=float(gain);self.cap=self.a.cap;self.word=word;self.weight=weight
    def done(self,p):return self.a.done(p)
    def mass(self,p):return (1-self.gain)*self.a.mass(p)+self.gain*self.b.mass(p)
    def literal(self,p):
        a,b=self.a.literal(p);c,d=self.b.literal(p)
        assert abs(b-d)<1e-15
        return (1-self.gain)*a+self.gain*c,b
    def complete(self,p):return (1-self.gain)*self.a.complete(p)+self.gain*self.b.complete(p)
    def prefix_mass(self,p):return (1-self.gain)*self.a.prefix_mass(p)+self.gain*self.b.prefix_mass(p)
    def next(self,p):
        a=(1-self.gain)*self.a.mass(p);b=self.gain*self.b.mass(p)
        return (a*self.a.next(p)+b*self.b.next(p))/(a+b)
    rank=FE.Event.rank
    tokens=FE.Event.tokens

def event82(word,raw,long,root,surface,weight,gain):
    return FE.Event(word,raw,root,surface,weight) if gain==0 else MixedEvent(word,raw,long,root,surface,weight,gain)

def selftest():
    from trajectory69 import Sparse
    from collections import Counter
    docs=[b' The cat sat. The Cat ran. the cat ate.',b'A cat ran. The dog sat.']
    m=LongByte(docs,12);r=np.random.default_rng(82)
    for _ in range(150):
        d=docs[int(r.integers(len(docs)))];j=int(r.integers(len(d)));h=d[max(0,j-12):j]
        for k in range(1,len(h)+1):
            key=h[-k:];actual=dict(zip(*m.idx.counts(tuple(b+1 for b in key))))
            expected=Counter(d0[i]+1 for d0 in docs for i in range(k,len(d0)) if d0[i-k:i]==key)
            assert actual==expected
        c=bytes([int(r.integers(256))])
        assert m.canonical(h+c)==m.canonical(m.canonical(h)+c)
    law=Sparse({'cat':.5,'dog':.3,'a':.2});surface=FE.Surface(docs)
    raw=lambda h:np.ones(256)/256
    old=FE.Event(law,raw,b'The ',surface,.3)
    zero=event82(law,raw,m,b'The ',surface,.3,0.)
    e=event82(law,raw,m,b'The ',surface,.3,.75)
    for p in (b'',b'c',b'Ca',b'cat',b'Do'):
        assert np.array_equal(old.next(p),zero.next(p))
        q=e.next(p);expected=np.array([e.mass(p+bytes([i]))/e.mass(p) for i in range(256)])
        assert np.max(np.abs(q-expected))<1e-12 and abs(q.sum()-1)<1e-12
    for p in (b'a',b'cat',b'dog'):
        variants=[bytes(v) for v in itertools.product(*[(b,b-32) for b in p])]
        expected=math.fsum(e.mass(v+bytes([i])) for v in variants for i in FE.STOP)
        assert abs(e.complete(p)-expected)<1e-12
    assert m.support(b'The cat ')[0]>3
    print('v82 exact suffix counts, canonical transition, mixed next-byte parity, case enumeration and zero rollback passed')
if __name__=='__main__':selftest()
