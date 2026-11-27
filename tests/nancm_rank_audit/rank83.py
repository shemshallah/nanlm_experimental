"""Exact completion certification for composed NaNCM probability laws.

No target enters the search. Seed words provide only lower bounds; the
priority queue certifies against every unvisited open-vocabulary prefix.
"""
import bisect, heapq, math
from collections import defaultdict
import numpy as np

def up(x):
    # Conservative rounding envelope for positive sums/products.
    return math.nextafter(float(x),math.inf)+1e-14*max(1.,abs(float(x)))

def best_indices(p,k):
    k=min(k,len(p))
    if not k:return []
    return np.argpartition(-p,k-1)[:k]

class Bound:
    def __init__(self,law):
        self.law=law;self.cache={};self.seeds=set();self.kind='generic'
        if all(hasattr(law,k) for k in ('outside','words','p','unknown')):
            self.kind='lift'
            self.seeds.update(law.words[i] for i in best_indices(law.p,8))
        elif all(hasattr(law,k) for k in ('parent','child','g')):
            self.kind='blend';self.a=Bound(law.parent);self.b=Bound(law.child)
            self.seeds=self.a.seeds|self.b.seeds
        elif all(hasattr(law,k) for k in ('base','f','z','delta')):
            self.kind='tilt';self.a=Bound(law.base);self.maxima=defaultdict(float)
            self.seeds=set(self.a.seeds)
            for w,f in law.f.items():
                p=law.complete(w)
                for i in range(len(w)+1):
                    h=w[:i];self.maxima[h]=max(self.maxima[h],p)
            self.seeds.update(sorted(law.f,key=lambda w:(-law.complete(w),w))[:8])
        elif isinstance(getattr(law,'p',None),dict):
            self.kind='sparse'
            self.maxima=defaultdict(float)
            for w,p in law.p.items():
                for i in range(len(w)+1):
                    h=w[:i];self.maxima[h]=max(self.maxima[h],p)
            self.seeds.update(sorted(law.p,key=lambda w:(-law.p[w],w))[:8])
    def __call__(self,h):
        if h in self.cache:return self.cache[h]
        l=self.law
        if self.kind=='lift':
            a=bisect.bisect_left(l.words,h);b=bisect.bisect_left(l.words,h+'{')
            known=float(np.max(l.p[a:b])) if a<b else 0.
            v=max(known,l.unknown*l.outside.mass(h))
        elif self.kind=='blend':
            v=(1-l.g)*self.a(h)+l.g*self.b(h)
        elif self.kind=='tilt':
            # Unmodified completions cannot exceed the unmodified bound / Z.
            # Modified completions have an independently exact prefix maximum.
            v=max(self.a(h)/l.z,self.maxima.get(h,0.))
        elif self.kind=='sparse':v=self.maxima.get(h,0.)
        else:v=l.mass(h)
        v=up(v)
        if len(self.cache)<65536:self.cache[h]=v
        return v

def raw_mass(event,p):
    return math.fsum(event.states(p).values())

class EventBound:
    def __init__(self,event):
        self.e=event;self.word=Bound(event.word)
        self.seeds=self.word.seeds
    def __call__(self,p):
        e=self.e;lex=self.word(p.decode('ascii'))
        if all(hasattr(e,k) for k in ('a','b','gain')):
            raw=(1-e.gain)*raw_mass(e.a,p)+e.gain*raw_mass(e.b,p)
            v=(1-e.weight)*raw+e.weight*lex
        elif all(hasattr(e,k) for k in ('identity','original','r','m')):
            v=((1-e.weight)*raw_mass(e,p)+e.weight*lex if e.identity
               else e.a*raw_mass(e,p)+e.b*lex)
        elif hasattr(e,'states'):
            v=(1-e.weight)*raw_mass(e,p)+e.weight*lex
        else:v=e.prefix_mass(p)
        # Both are bounds; minimum remains a bound. Avoid extra calls to
        # expensive composite prefix sums when the analytical bound suffices.
        return up(v)

def rank(event,k=5,budget=1200,seeds=()):
    assert k>0 and budget>0
    bound=EventBound(event);best=[];scored=set();nodes=0
    def score(p):
        if not p or p in scored:return
        scored.add(p)
        item=(event.complete(p),p)
        if len(best)<k:heapq.heappush(best,item)
        elif item>best[0]:heapq.heapreplace(best,item)
    for w in sorted(bound.seeds|set(seeds)):
        if isinstance(w,str) and 0<len(w)<event.cap and w.isascii() and w.isalpha() and w.islower():score(w.encode('ascii'))
    queue=[(-up(1.),b'')]
    while queue and nodes<budget:
        if len(best)==k and -queue[0][0]<best[0][0]:break
        _,p=heapq.heappop(queue);nodes+=1;score(p)
        if len(p)>=event.cap-1:continue
        for c in range(97,123):
            h=p+bytes([c]);m=bound(h)
            if len(best)<k or m>=best[0][0]:heapq.heappush(queue,(-m,h))
    cert=not queue or (len(best)==k and -queue[0][0]<best[0][0])
    return [p.decode('ascii') for _,p in sorted(best,reverse=True)],bool(cert),nodes

def selftest():
    import itertools,time
    import folded_event as FE
    from open_word import Lift,Outside,Spelling,fuse
    from trajectory69 import Sparse
    from field77 import Tilt
    from field81 import Hurdle
    from field82 import LongByte,MixedEvent
    rng=np.random.default_rng(83);docs=[b'The cat ran. A dog sat. The Cat sat.',b'A dog ran. The cat ate.']
    vocab=['a','ab','an','and','cat','dog','ran','sat','the']
    outside=Outside(vocab,Spelling([['a','cat','the','dog'],['and','ran','sat']]))
    long=LongByte(docs,12);surface=FE.Surface(docs)
    raw=lambda h:np.arange(1,257,dtype=np.float64)/32896.
    oldnodes=newnodes=0;oldtime=newtime=0.;cases=0
    for i in range(12):
        p=rng.random(len(vocab)+1);p/=p.sum();a=Lift(outside,p)
        sparse=Sparse(dict(zip(vocab,rng.random(len(vocab)))))
        b=fuse(a,sparse,.3)
        law=Tilt(b,['a','cat','dog','z'],np.exp(rng.uniform(-2,2,4)))
        weight=float(rng.uniform(.15,.85))
        for kind in ('base','hurdle','mixed'):
            def event():
                if kind=='base':return FE.Event(law,raw,b'The ',surface,weight,cap=8)
                if kind=='hurdle':return Hurdle(law,raw,b'The ',surface,weight,.7,cap=8)
                return MixedEvent(law,raw,long,b'The ',surface,weight,.6)
            e=event();bd=EventBound(e)
            for w in vocab+['z','zz','cats','dogs']:
                word=w.encode()
                assert all(bd(word[:j])>=e.complete(word) for j in range(1,len(word)+1))
            t=time.perf_counter();old=event().rank(budget=2400);oldtime+=time.perf_counter()-t
            t=time.perf_counter();new=rank(event(),budget=2400);newtime+=time.perf_counter()-t
            assert old[1] and new[1] and old[0]==new[0],(kind,old,new)
            oldnodes+=old[2];newnodes+=new[2];cases+=1
    print('RANK83',dict(cases=cases,old_nodes=oldnodes,new_nodes=newnodes,old_seconds=oldtime,new_seconds=newtime))
if __name__=='__main__':selftest()
