"""v85 compile a composite open-word law into known + escape + finite OOV correction."""
import bisect,math
from collections import defaultdict
import numpy as np

class Compiler:
    def __init__(self,outside):
        self.outside=outside;self.words=outside.words
        self.ids={w:i for i,w in enumerate(self.words)}
    def parts(self,law):
        if all(hasattr(law,k) for k in ('outside','words','p','unknown')):
            assert law.outside is self.outside
            return law.p.copy(),float(law.unknown),{}
        if all(hasattr(law,k) for k in ('parent','child','g')):
            a,u,x=self.parts(law.parent);b,v,y=self.parts(law.child)
            g=law.g;d=defaultdict(float)
            for w,p in x.items():d[w]+=(1-g)*p
            for w,p in y.items():d[w]+=g*p
            return (1-g)*a+g*b,(1-g)*u+g*v,dict(d)
        if all(hasattr(law,k) for k in ('base','f','z','delta')):
            p,u,x=self.parts(law.base)
            p/=law.z;u/=law.z;x={w:v/law.z for w,v in x.items()}
            for w,f in law.f.items():
                i=self.ids.get(w)
                if i is not None:p[i]*=f
                else:
                    old=u*self.outside.complete(w)+x.get(w,0.)
                    x[w]=x.get(w,0.)+old*(f-1.)
            return p,u,x
        if isinstance(getattr(law,'p',None),dict):
            a=np.zeros(len(self.words),np.float64);d={}
            for w,p in law.p.items():
                i=self.ids.get(w)
                if i is None:d[w]=p
                else:a[i]=p
            return a,0.,d
        raise TypeError(type(law))
    def __call__(self,law):
        return Flat(self,*self.parts(law))

class Flat:
    def __init__(self,compiler,p,u,x):
        self.compiler=compiler;self.words=compiler.words;self.outside=compiler.outside
        self.known=p;self.escape=u;self.x=x
        assert p.min()>=0 and u>=0
        assert abs(math.fsum(p)+u+math.fsum(x.values())-1)<1e-9
        self.d=defaultdict(list);self.cache={}
        for w,v in x.items():
            assert u*self.outside.complete(w)+v>=-1e-15
            for i in range(len(w)+1):self.d[w[:i]].append(v)
        self.d={w:math.fsum(v) for w,v in self.d.items()}
    def complete(self,w):
        i=self.compiler.ids.get(w)
        if i is not None:return float(self.known[i])
        return max(0.,math.fsum((self.escape*self.outside.complete(w),self.x.get(w,0.))))
    def mass(self,p):
        if not p:return 1.
        if p in self.cache:return self.cache[p]
        a=bisect.bisect_left(self.words,p);b=bisect.bisect_left(self.words,p+'{')
        q=max(0.,math.fsum((math.fsum(self.known[a:b]),self.escape*self.outside.mass(p),self.d.get(p,0.))))
        self.cache[p]=q;return q

def selftest():
    from open_word import Outside,Spelling,Lift,fuse
    from trajectory69 import Sparse
    from field77 import Tilt
    import folded_event as FE
    rng=np.random.default_rng(85)
    outside=Outside(['a','ab','and','cat','dog','the'],Spelling([['a','cat','dog','the','and']]))
    comp=Compiler(outside);raw=lambda h:np.arange(1,257,dtype=np.float64)/32896.
    surface=FE.Surface([b'The cat and a dog.'])
    error=0.;cases=0
    for i in range(48):
        p=rng.random(7);p/=p.sum();a=Lift(outside,p)
        b=Sparse({'a':.1,'cat':.4,'zoo':.5})
        t=Tilt(fuse(a,b,.35),['a','cat','zoo','zz'],np.exp(rng.uniform(-4,4,4)))
        law=fuse(Tilt(t,['and','zz'],np.exp(rng.uniform(-4,4,2))),a,.17)
        flat=comp(law)
        for w in ('','a','ab','and','cat','dog','the','z','zo','zoo','zz','zzz'):
            error=max(error,abs(flat.mass(w)-law.mass(w)),abs(flat.complete(w)-law.complete(w)))
            assert abs(flat.mass(w)-flat.complete(w)-math.fsum(flat.mass(w+c) for c in 'abcdefghijklmnopqrstuvwxyz'))<1e-11
        a0=FE.Event(law,raw,b'The ',surface,.65);b0=FE.Event(flat,raw,b'The ',surface,.65)
        for p in (b'',b'a',b'Ca',b'z',b'zo'):
            d=float(np.max(np.abs(a0.next(p)-b0.next(p))));error=max(error,d);assert d<1e-11
        aa=a0.rank();bb=b0.rank();assert aa[1] and bb[1] and aa[0]==bb[0]
        cases+=1
    assert error<1e-11
    print('FLAT85',dict(cases=cases,max_error=error),flush=True)
if __name__=='__main__':selftest()
