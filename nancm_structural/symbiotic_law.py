"""Finite likelihood transport and one raw-event prefix law. No corpus fitting."""
import math, heapq
import numpy as np
ALPHA=tuple(range(97,123))
CAP=64
SPLIT=frozenset(b' etaoinr')

def transport(base, candidates, log_lift):
    return Transport(base,candidates,log_lift)

class Transport:
    """Tilt finite candidate masses; scale all unlisted outcomes equally."""
    def __init__(self,base,candidates,log_lift):
        self.base=base;self.delta={};self.candidates=tuple(candidates)
        assert len(set(self.candidates))==len(self.candidates)
        for w,l in zip(self.candidates,log_lift,strict=True):
            assert w and w.isascii() and w.isalpha() and w.islower()
            self.delta[w]=base.complete(w)*math.expm1(float(np.clip(l,-4,4)))
        self.z=1.+math.fsum(self.delta.values());assert self.z>0
        self.prefix={}
        for w,p in self.delta.items():
            for i in range(len(w)+1):self.prefix[w[:i]]=self.prefix.get(w[:i],0.)+p
    def complete(self,w):return max(0.,(self.base.complete(w)+self.delta.get(w,0.))/self.z)
    def mass(self,p):return max(0.,(self.base.mass(p)+self.prefix.get(p,0.))/self.z)

class Event:
    """Canonical lowercase+space event and unrestricted raw event mixture.
    Both components use a stop rule: first non-lowercase byte, or cap.
    Prefix mass and complete-event probability are available directly.
    """
    def __init__(self,word,raw,weight,cap=CAP):
        assert 0<=weight<1 and cap>=2
        self.word=word;self.raw=raw;self.weight=weight;self.cap=cap;self._rawmass={b'':1.}
    def done(self,p):return len(p)>=self.cap or bool(p and p[-1] not in ALPHA)
    def rawmass(self,p):
        if p not in self._rawmass:
            assert not self.done(p[:-1])
            q=self.raw(p[:-1]);self._rawmass[p]=self.rawmass(p[:-1])*float(q[p[-1]])
        return self._rawmass[p]
    def canonmass(self,p):
        if not p:return 1.
        if p[-1]==32 and len(p)>1 and all(b in ALPHA for b in p[:-1]):return self.word.complete(p[:-1].decode('ascii'))
        if all(b in ALPHA for b in p):return self.word.mass(p.decode('ascii'))
        return 0.
    def mass(self,p):
        if not p:return 1.
        assert len(p)<=self.cap and not self.done(p[:-1])
        return (1-self.weight)*self.rawmass(p)+self.weight*self.canonmass(p)
    def next(self,p):
        assert not self.done(p)
        raw=np.asarray(self.raw(p),np.float64);assert raw.shape==(256,) and raw.min()>0 and abs(raw.sum()-1)<1e-8
        a=(1-self.weight)*self.rawmass(p);b=self.weight*self.canonmass(p)
        q=a*raw
        if b:
            s=p.decode('ascii');den=self.word.mass(s)
            for c in ALPHA:q[c]+=b*self.word.mass(s+chr(c))/den
            if s:q[32]+=b*self.word.complete(s)/den
        q/=a+b;assert abs(q.sum()-1)<1e-7 and q.min()>0
        return q
    def tokens(self,p):
        """Prefix-complete adaptive leaves; stop token at event boundary."""
        q=self.next(p);out={}
        for i,v in enumerate(q):
            a=bytes([i]);h=p+a
            if i in SPLIT and not self.done(h):
                for j,w in enumerate(self.next(h)):out[a+bytes([j])]=float(v*w)
            else:out[a]=float(v)
        assert abs(math.fsum(out.values())-1)<1e-7
        return out
    def rank_words(self,k=5,budget=2000):
        """Rank actual canonical word events (lowercase, then space).
        Prefix masses are rigorous upper bounds; noncanonical events excluded.
        """
        heap=[(-1.,b'')];best=[];seen=0
        while heap and seen<budget:
            if len(best)>=k and -heap[0][0]<best[0][0]:break
            _,p=heapq.heappop(heap);seen+=1
            if p:
                w=self.mass(p+b' ');item=(w,p)
                if len(best)<k:heapq.heappush(best,item)
                elif item>best[0]:heapq.heapreplace(best,item)
            if len(p)<self.cap-1:
                for b in ALPHA:
                    h=p+bytes([b]);m=self.mass(h)
                    if m>0 and (len(best)<k or m>=best[0][0]):heapq.heappush(heap,(-m,h))
        cert=not heap or (len(best)>=k and -heap[0][0]<best[0][0])
        return [p.decode('ascii') for _,p in sorted(best,reverse=True)],cert,seen

def preflight():
    class Law:
        d={'a':.2,'an':.3,'ant':.5}
        def mass(self,p):return math.fsum(v for w,v in self.d.items() if w.startswith(p))
        def complete(self,w):return self.d.get(w,0.)
    base=Law();raw=lambda p:np.full(256,1/256)
    law=Transport(base,['a','ant'],[1.3,-.7]);assert abs(sum(law.complete(w) for w in base.d)-1)<1e-14
    zero=Transport(base,['a'],[0]);assert all(zero.complete(w)==base.complete(w) for w in base.d)
    e=Event(law,raw,.6,cap=8);err=0.
    for p in (b'',b'a',b'an',b'ant',b'z',b'zz'):
        q=e.next(p);err=max(err,float(np.max(np.abs(q-np.array([e.mass(p+bytes([b]))/e.mass(p) for b in range(256)])))))
        t=e.tokens(p)
        for b in range(256):assert abs(sum(v for k,v in t.items() if k[0]==b)-q[b])<1e-12
    for w in base.d:
        p=b'';v=1.
        for b in w.encode()+b' ':v*=e.next(p)[b];p+=bytes([b])
        assert abs(v-e.mass(p))<1e-14
    rank,cert,n=e.rank_words(3);assert cert and rank==sorted(base.d,key=lambda w:-e.mass(w.encode()+b' '))
    assert err<1e-12
    print(dict(max_prefix_error=err,rank=rank,certified=cert,expanded=n,all_256_supported=True,zero_transport_exact=True))
if __name__=='__main__':preflight()
