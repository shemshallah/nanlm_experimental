"""Exact case-folded ASCII word marginal of a full-support raw byte event law.
Raw donor must depend on no more than the declared suffix order.
Event ends at first nonletter byte or at cap; empty and capped events retained.
"""
import collections as C
import functools
import heapq
import math
import numpy as np
LOWER=tuple(range(97,123)); LETTERS=tuple(range(65,91))+LOWER
IS_LETTER=np.zeros(256,bool);IS_LETTER[list(LETTERS)]=True
STOP=np.flatnonzero(~IS_LETTER)

def folded(p):return p.lower().decode('ascii')

class Surface:
    """Smoothed surface law fitted only on donor documents."""
    def __init__(self,docs):
        self.cases=np.ones((2,26,2),np.float64)*.5
        self.ends=np.ones((2,256),np.float64)*.01;self.ends[:,IS_LETTER]=0
        for doc in docs:
            n=0
            for b in doc:
                if IS_LETTER[b]:
                    self.cases[int(n>0),(b|32)-97,int(b<97)]+=1;n+=1
                else:
                    if n:self.ends[int(n>1),b]+=1
                    n=0
        self.cases/=self.cases.sum(-1,keepdims=True);self.ends/=self.ends.sum(-1,keepdims=True)
    def letter(self,p,c):return self.cases[int(bool(p)),c-97]
    def delimiter(self,p):return self.ends[int(len(p)>1)]

class Event:
    def __init__(self,word,raw,root,surface,weight,order=3,cap=64):
        assert 0<=weight<1 and 0<=order<=8 and cap>=2
        self.word=word;self.raw=raw;self.root=bytes(root);self.surface=surface;self.weight=float(weight);self.order=order;self.cap=cap
        self._states={b'':{self.suffix(self.root):1.}}
        self._raw={};self._literal={b'':(1.,1.)}
    def suffix(self,p):return p[-self.order:] if self.order else b''
    def done(self,p):return len(p)>=self.cap or bool(p and not IS_LETTER[p[-1]])
    def donor(self,suffix):
        if suffix not in self._raw:
            q=np.asarray(self.raw(suffix),np.float64);assert q.shape==(256,) and q.min()>0 and abs(q.sum()-1)<1e-9
            if len(self._raw)<8192:self._raw[suffix]=q
            return q
        return self._raw[suffix]
    def states(self,p):
        assert len(p)<=self.cap and all(c in LOWER for c in p)
        if p in self._states:return self._states[p]
        out=C.defaultdict(float);c=p[-1]
        for s,m in self.states(p[:-1]).items():
            q=self.donor(s)
            for b in (c,c-32):out[self.suffix(s+bytes([b]))]+=m*q[b]
        self._states[p]=dict(out);return self._states[p]
    def prefix_mass(self,p):
        assert len(p)<=self.cap
        return (1-self.weight)*math.fsum(self.states(p).values())+self.weight*self.word.mass(p.decode('ascii'))
    def complete(self,p):
        if not p or len(p)>=self.cap:return 0.
        raw=math.fsum(m*float(self.donor(s)[STOP].sum()) for s,m in self.states(p).items())
        return (1-self.weight)*raw+self.weight*self.word.complete(p.decode('ascii'))
    def literal(self,p):
        if p in self._literal:return self._literal[p]
        assert not self.done(p[:-1]);a,b=self.literal(p[:-1]);c=p[-1]
        a*=self.donor(self.suffix(self.root+p[:-1]))[c]
        if b:
            s=folded(p[:-1]);den=self.word.mass(s)
            if den<=0:b=0.
            elif IS_LETTER[c]:
                k=c|32;b*=self.word.mass(s+chr(k))/den*self.surface.letter(p[:-1],k)[int(c<97)]
            elif p[:-1]:b*=self.word.complete(s)/den*self.surface.delimiter(p[:-1])[c]
            else:b=0.
        v=(float(a),float(b));self._literal[p]=v;return v
    def mass(self,p):
        a,b=self.literal(p);return (1-self.weight)*a+self.weight*b
    def next(self,p):
        assert not self.done(p)
        a,b=self.literal(p);a*=1-self.weight;b*=self.weight
        q=a*self.donor(self.suffix(self.root+p)).copy()
        if b:
            s=folded(p);den=self.word.mass(s);assert den>0
            for c in LOWER:
                m=b*self.word.mass(s+chr(c))/den;case=self.surface.letter(p,c)
                q[c]+=m*case[0];q[c-32]+=m*case[1]
            if p:q+=b*self.word.complete(s)/den*self.surface.delimiter(p)
        q/=(a+b);assert abs(q.sum()-1)<1e-8 and q.min()>0
        return q
    def tokens(self,p,split=frozenset(b' etaoinr')):
        q=self.next(p);out={}
        for c,m in enumerate(q):
            t=bytes([c]);h=p+t
            if c in split and not self.done(h):
                for d,v in enumerate(self.next(h)):out[t+bytes([d])]=float(m*v)
            else:out[t]=float(m)
        assert abs(math.fsum(out.values())-1)<1e-8;return out
    def rank(self,k=5,budget=2000):
        heap=[(-1.,b'')];best=[];seen=0
        while heap and seen<budget:
            if len(best)>=k and -heap[0][0]<best[0][0]:break
            _,p=heapq.heappop(heap);seen+=1
            if p:
                item=(self.complete(p),p)
                if len(best)<k:heapq.heappush(best,item)
                elif item>best[0]:heapq.heapreplace(best,item)
            if len(p)<self.cap-1:
                for b in LOWER:
                    h=p+bytes([b]);m=self.prefix_mass(h)
                    if m>0 and (len(best)<k or m>=best[0][0]):heapq.heappush(heap,(-m,h))
        cert=not heap or (len(best)>=k and -heap[0][0]<best[0][0])
        return [p.decode('ascii') for _,p in sorted(best,reverse=True)],cert,seen
