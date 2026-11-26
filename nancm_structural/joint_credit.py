"""Task conflict diagnostics and symmetric per-group gradient surgery."""
import numpy as np
def merge(word, token, eps=1e-12):
    out={}; stats={"shared":0,"conflicts":0,"cosine_sum":0.,"projected":0}
    for k in sorted(set(word)|set(token)):
        if k not in word:out[k]=token[k].copy();continue
        if k not in token:out[k]=word[k].copy();continue
        a=word[k];b=token[k];aa=float(np.vdot(a,a));bb=float(np.vdot(b,b));ab=float(np.vdot(a,b))
        if aa<=eps:out[k]=b.copy();continue
        if bb<=eps:out[k]=a.copy();continue
        stats['shared']+=1;stats['cosine_sum']+=ab/np.sqrt(aa*bb)
        if ab<0:
            stats['conflicts']+=1
            u=a-(ab/bb)*b;v=b-(ab/aa)*a
            stats['projected']+=1
        else:u=a;v=b
        size=min(np.sqrt(aa),np.sqrt(bb));g=.5*size*(u/np.sqrt(aa)+v/np.sqrt(bb))
        out[k]=g.astype(a.dtype,copy=False)
    return out,stats
def preflight():
    rng=np.random.default_rng(37);worst=0.;conflicts=0
    for _ in range(1000):
        a=rng.normal(size=64);b=rng.normal(size=64)-.7*a
        g,s=merge({'p':a},{'p':b});g=g['p'];conflicts+=s['conflicts']
        worst=min(worst,float(a@g),float(b@g));assert np.isfinite(g).all()
    assert worst>=-1e-10 and conflicts>0
    g,s=merge({'p':np.array([1.,0.]),'w':np.ones(2)},{'p':np.array([-1.,0.]),'t':np.ones(3)})
    assert np.array_equal(g['p'],np.zeros(2)) and np.array_equal(g['w'],np.ones(2)) and np.array_equal(g['t'],np.ones(3))
    print({'random_pairs':1000,'conflicts':conflicts,'worst_directional_dot':worst,'opposed_exact_zero':True})
if __name__=='__main__':preflight()
