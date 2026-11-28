"""v90 conditional allocation over a mixture of exact raw byte event laws."""
import math
import numpy as np
import folded_event as FE
from field81 import Hurdle
from field82 import LongEvent,event82

class ConditionalMixed:
    def __init__(self,word,raw,long,root,surface,weight,gain,conditional):
        self.word=word;self.weight=weight;self.cap=64
        self.r0=float(raw(root[-3:])[FE.IS_LETTER].sum())
        self.r1=float(long.prob(long.canonical(root[-long.order:]))[FE.IS_LETTER].sum())
        self.r=(1-gain)*self.r0+gain*self.r1
        self.m=weight+(1-weight)*self.r;self.original=weight/self.m
        assert 0<=conditional<1 and self.r>0
        self.a=self.m*(1-conditional)/self.r;self.b=self.m*conditional
        self.identity=conditional==self.original
        self.factor=self.a+self.b
        self.inner=event82(word,raw,long,root,surface,self.b/self.factor,gain)
        # Nonword probability is conserved exactly at each event boundary.
        self.stop=(1-weight)*((1-gain)*raw(root[-3:])+gain*long.prob(long.canonical(root[-long.order:])))
    def done(self,p):return self.inner.done(p)
    def literal(self,p):return self.inner.literal(p)
    def mass(self,p):
        if not p:return 1.
        if not FE.IS_LETTER[p[0]]:return float(self.stop[p[0]]) if len(p)==1 else 0.
        return self.factor*self.inner.mass(p)
    def next(self,p):
        if p:return self.inner.next(p)
        q=self.factor*self.inner.next(p)
        q[FE.STOP]=self.stop[FE.STOP]
        assert q.min()>0 and abs(q.sum()-1)<1e-8
        return q
    def prefix_mass(self,p):return self.factor*self.inner.prefix_mass(p) if p else 1.
    def complete(self,p):return self.factor*self.inner.complete(p)
    rank=FE.Event.rank
    tokens=FE.Event.tokens

def event90(word,raw,long,root,surface,weight,gain,conditional):
    if gain==0:return Hurdle(word,raw,root,surface,weight,conditional)
    return ConditionalMixed(word,raw,long,root,surface,weight,gain,conditional)

def selftest():
    from trajectory69 import Sparse
    from field82 import LongByte
    import itertools
    docs=[b'The cat ran. The Cat sat.',b'The dog ran. A cat sat.']
    surface=FE.Surface(docs);long=LongByte(docs,12);law=Sparse({'a':.2,'cat':.5,'dog':.3})
    raw=lambda h:np.arange(1,257,dtype=np.float64)/32896.
    for gain in (0.,.25,.5,1.):
        weight=.3;root=b'The '
        q=(1-gain)*raw(root[-3:])+gain*long.prob(long.canonical(root))
        r=float(q[FE.IS_LETTER].sum());m=weight+(1-weight)*r;s=weight/m
        for cond in (s,.2,.7,.95):
            e=event90(law,raw,long,root,surface,weight,gain,cond)
            for p in (b'',b'c',b'Ca',b'cat',b'Do'):
                q1=e.next(p);q2=np.array([e.mass(p+bytes([i]))/e.mass(p) for i in range(256)])
                assert np.max(np.abs(q1-q2))<1e-11
            assert np.max(np.abs(e.next(b'')[FE.STOP]-(1-weight)*q[FE.STOP]))<1e-14
            for p in (b'a',b'cat',b'dog'):
                variants=[bytes(v) for v in itertools.product(*[(b,b-32) for b in p])]
                expected=math.fsum(e.mass(v+bytes([i])) for v in variants for i in FE.STOP)
                assert abs(e.complete(p)-expected)<1e-12
            if cond==s:
                old=event82(law,raw,long,root,surface,weight,gain)
                for p in (b'',b'c',b'Ca',b'dog'):
                    assert np.max(np.abs(old.next(p)-e.next(p)))<1e-12
        assert event90(law,raw,long,root,surface,weight,0.,.7).__class__ is Hurdle
    print('v90 conditional mixed event normalization, punctuation conservation, case marginal and zero identity passed')
if __name__=='__main__':selftest()
