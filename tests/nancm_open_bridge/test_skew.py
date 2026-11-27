"""Large skewed dictionary regression, exercising real Lift, not a toy copy."""
import bisect,itertools,math,numpy as np
from open_word import Spelling,Outside,Lift,A

def main():
    words=[''.join(x) for x in itertools.islice(itertools.product(A,repeat=4),49960)]
    # Zero OOV weight isolates the known-prefix numerical failure.
    p=np.full(len(words)+1,1e-9);p[0]=0.;p[1]=1-math.fsum(p[2:])
    m=Lift(Outside(words,Spelling([['a','b','cat','dog','z']])),p)
    c=np.r_[0.,np.cumsum(m.p)]
    def old(s):
        if not s:return 1.
        a=bisect.bisect_left(words,s);b=bisect.bisect_left(words,s+'{')
        z=float(c[b]-c[a])
        return math.fsum(m.p[a:b]) if z<1e-10 else z
    rng=np.random.default_rng(59);prefixes=['']+[words[i][:j] for i in rng.integers(1,len(words),size=512) for j in (1,2,3,4)]
    err=old_err=0.;old_bad=0
    for s in prefixes:
        z=m.mass(s);ref=m.complete(s)+math.fsum(m.mass(s+a) for a in A)
        assert z>0
        err=max(err,abs(ref/z-1))
        old_z=old(s);old_ref=m.complete(s)+math.fsum(old(s+a) for a in A)
        e=abs(old_ref/old_z-1);old_err=max(old_err,e);old_bad+=e>=1e-8
    assert err<1e-12 and old_bad>0
    print(dict(vocabulary=len(words),prefix_checks=len(prefixes),old_max_relative_error=old_err,old_failed=old_bad,new_max_relative_error=err))
if __name__=='__main__':main()
