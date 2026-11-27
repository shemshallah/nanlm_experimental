import itertools,math,numpy as np
from switching import Switch

def main():
    pi=np.array([.2,.3,.5]);h=.17;lik=np.array([[.8,.2,.1],[.3,.6,.5],[.2,.1,.7],[.9,.2,0.]])
    f=Switch(pi,h);p=math.prod(f.observe(x) for x in lik)
    oracle=0.
    for path in itertools.product(range(3),repeat=len(lik)):
        x=pi[path[0]]*lik[0,path[0]]
        for t in range(1,len(path)):x*=((1-h)*(path[t]==path[t-1])+h*pi[path[t]])*lik[t,path[t]]
        oracle+=x
    assert abs(p-oracle)<1e-15
    f.reset();assert np.array_equal(f.posterior,pi)
    static=Switch(pi,1.);assert all(abs(static.observe(x)-pi@x)<1e-15 for x in lik)
    # Constructed changing regimes; no claim about language.
    pi=np.array([.2,.4,.4]);seq=np.array([[.5,.9,.1]]*64+[[.5,.1,.9]]*64+[[.5,.9,.1]]*64)
    f=Switch(pi,.05);fixed=-np.log2(seq@pi).mean();adaptive=-np.log2([f.observe(x) for x in seq]).mean()
    assert adaptive<fixed and f.posterior[1]>.9
    print(dict(enumeration_error=abs(p-oracle),paths=3**len(lik),static_exact=True,reset_exact=True,constructed_fixed_bits=float(fixed),constructed_adaptive_bits=float(adaptive),events=len(seq)))
if __name__=='__main__':main()
