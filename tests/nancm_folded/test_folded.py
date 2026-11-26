import itertools,math,numpy as np
from folded_event import *
class Word:
    d={'a':.2,'an':.3,'ant':.5}
    def mass(self,p):return math.fsum(v for w,v in self.d.items() if w.startswith(p))
    def complete(self,w):return self.d.get(w,0.)
@functools.lru_cache(None)
def raw(s):
    seed=int.from_bytes(s,'little')%2**32
    q=np.random.default_rng(seed).random(256)+.001;q/=q.sum();return q
surface=Surface([b'An ANT! a an. a\n'])
e=Event(Word(),raw,b'root ',surface,.63,order=3,cap=5)
err=0.;marg=0.;n=0
for w in ['a','an','ant','zz','abcd']:
    sums=[]
    for variant in itertools.product(*[(b,b-32) for b in w.encode()]):
        p=bytes(variant)
        sums.append(sum(e.mass(p+bytes([d])) for d in STOP))
        q=e.next(p);err=max(err,float(np.max(np.abs(q-np.array([e.mass(p+bytes([d]))/e.mass(p) for d in range(256)])))))
        t=e.tokens(p);b=np.zeros(256)
        for tok,v in t.items():b[tok[0]]+=v
        marg=max(marg,float(np.max(np.abs(b-q))));n+=1
    assert abs(sum(sums)-e.complete(w.encode()))<1e-12
    p=w.encode()
    if len(p)<4:assert abs(e.prefix_mass(p)-sum(e.prefix_mass(p+bytes([d])) for d in LOWER)-e.complete(p))<1e-12
rank,cert,seen=e.rank(3);assert cert and rank==sorted(Word.d,key=lambda w:-e.complete(w.encode()))
assert abs(e.prefix_mass(b'a')-sum(e.mass(bytes([c])) for c in (97,65)))<1e-12
assert abs(sum(e.prefix_mass(bytes([c])) for c in LOWER)+sum(e.mass(bytes([c])) for c in STOP)-1)<1e-12
z=Event(Word(),raw,b'root ',surface,0.,order=3,cap=5)
for p in (b'',b'A',b'AnT'):
    assert np.allclose(z.next(p),raw((b'root '+p)[-3:]),atol=1e-15,rtol=0)
print(dict(surface_variants=n,conditional_error=err,token_marginal_error=marg,rank=rank,certified=cert,expanded=seen,escape=True,zero_raw_parity=True))
