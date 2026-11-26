"""Bayesian word-event/raw-byte mixture. Ends at first non-lowercase
ASCII byte or finite cap; canonical word branch uses ASCII space termination."""
import numpy as np
import math
def canonical(law,prefix,cap=64):
    if len(prefix)>=cap or any(not 97<=b<=122 for b in prefix):raise ValueError('event already ended')
    q=np.zeros(256,np.float64);s=prefix.decode('ascii');mass=law.mass(s)
    if mass<=0:return q
    for b in range(97,123):q[b]=law.mass(s+chr(b))/mass
    if s:q[32]=law.complete(s)/mass
    assert abs(q.sum()-1)<1e-8,(s,q.sum());q/=q.sum();return q
class Bridge:
    def __init__(self,law,weight,cap=64):
        assert 0<=weight<1 and cap>0
        self.law=law;self.weight=float(weight);self.raw_weight=1.-weight;self.prefix=b'';self.cap=cap;self.done=False
    def probability(self,raw):
        assert not self.done
        raw=np.asarray(raw,np.float64);assert raw.shape==(256,) and raw.min()>0 and np.isclose(raw.sum(),1)
        if self.weight==0:return raw
        c=canonical(self.law,self.prefix,self.cap);q=self.raw_weight*raw+self.weight*c
        assert abs(q.sum()-1)<1e-8 and q.min()>0
        return q
    def observe(self,b,raw):
        q=self.probability(raw)
        if self.weight:
            self.raw_weight=self.raw_weight*raw[b]/q[b]
            self.weight=self.weight*canonical(self.law,self.prefix,self.cap)[b]/q[b]
            den=self.weight+self.raw_weight;self.weight/=den;self.raw_weight/=den
        self.prefix+=bytes([b]);self.done=not 97<=b<=122 or len(self.prefix)>=self.cap
        return float(q[b])
def preflight():
    class Law:
        d={'a':.2,'an':.3,'ant':.5}
        def mass(self,p):return sum(v for w,v in self.d.items() if w.startswith(p))
        def complete(self,w):return self.d.get(w,0.)
    raw=np.ones(256)/256;law=Law();maxerr=0.
    for w,p in law.d.items():
        event=w.encode()+b' ';s=Bridge(law,.4);prob=math.prod(s.observe(b,raw) for b in event)
        target=.4*p+.6*(1/256)**len(event);maxerr=max(maxerr,abs(prob-target));assert abs(prob-target)<1e-12 and s.done
    s=Bridge(law,.4);assert s.observe(255,raw)==.6/256 and s.done and s.weight==0
    s=Bridge(law,0);assert np.array_equal(s.probability(raw),raw)
    s=Bridge(law,.4,cap=2);p=s.observe(97,raw)*s.observe(110,raw);assert s.done and abs(p-(.4*.8+.6/256**2))<1e-12
    print({'canonical_event_max_error':maxerr,'raw_escape_exact':True,'zero_weight_exact':True,'truncated_event_exact':True})
if __name__=='__main__':preflight()
