import math,numpy as np
from open_word import Spelling,Outside,Lift,fuse,A
from folded_event import Event,Surface
base=Spelling([['an','ant','a','bat','apple','ante','z']*40]);words=['a','an','ant','ante','bat'];outside=Outside(words,base);p=np.array([.14,.2,.12,.2,.14,.2]);l=Lift(outside,p)
assert all(l.complete(w)==v for w,v in zip(words,p[1:]));assert all(outside.complete(w)==0 for w in words)
assert abs(outside.z-(1-sum(base.complete(w) for w in words)))<1e-13
assert l.complete('apple')>0 and l.complete('unknown')>0
assert fuse(l,base,0) is l
rng=np.random.default_rng(58);masserr=0.;checks=0
for prefix in ['','a','an','ant','ante','b','ba','bat','app','z']+[''.join(rng.choice(list(A),size=rng.integers(1,12))) for _ in range(128)]:
    for m in (outside,l,fuse(base,l,.5)):
        masserr=max(masserr,abs(m.mass(prefix)-math.fsum([m.complete(prefix)]+[m.mass(prefix+c) for c in A])));checks+=1
raw=lambda h:np.full(256,1/256);surface=Surface([b'An ANT! a an. z\n']);err=0.;marg=0.;fold=0.;events=0
for g in (0.,.25,.75,1.):
    m=fuse(base,l,g)
    for weight in (0.,.25,.75):
        e=Event(m,raw,b'root ',surface,weight,cap=4)
        for event in (b'A ',b'An!',b'AnT?',b'Ante',b'z?',b'xy?',b'!',b'\xff'):
            pref=b'';prod=1.
            for b in event:
                q=e.next(pref);prod*=q[b];assert q.min()>0
                t=e.tokens(pref);v=np.zeros(256)
                for k,z in t.items():v[k[0]]+=z
                marg=max(marg,float(np.max(abs(v-q))));pref+=bytes([b])
            assert e.done(pref);err=max(err,abs(prod-e.mass(event)));events+=1
        for w in (b'a',b'an',b'ant'):
            import itertools
            literal=sum(e.mass(bytes(v)+bytes([b])) for v in itertools.product(*[(c,c-32) for c in w]) for b in range(256) if not (65<=b<=90 or 97<=b<=122))
            fold=max(fold,abs(literal-e.complete(w)))
assert max(masserr,err,marg,fold)<1e-12
print(dict(prefix_checks=checks,mass_error=masserr,events=events,literal_error=err,token_marginal_error=marg,folded_error=fold,unknown_mass=p[0],outside_base_mass=outside.z,exact_known=True,zero_exact=True))
