import itertools,math,numpy as np
from word_bound import CompletionBound,rank
from folded_event import Event,Surface
from open_word import Lift,Outside,Spelling,fuse
def main():
    words=sorted(''.join(x) for n in (1,2,3) for x in itertools.product('abcdefgh',repeat=n))
    rng=np.random.default_rng(66);p=rng.lognormal(0,1,len(words));p/=p.sum()
    outside=Outside(words,Spelling([words]))
    a=Lift(outside,np.r_[.005,.995*p]);b=Lift(outside,np.r_[.01,.99*p[::-1]])
    raw=lambda h: np.r_[np.full(65,.6/204),np.full(26,.4/52),np.full(6,.6/204),np.full(26,.4/52),np.full(133,.6/204)]
    surface=Surface([b'a b c d e f g h ']);count=0
    for law in (a,b,fuse(a,b,.4)):
        bound=CompletionBound(law)
        for prefix in ['',*words]:
            assert bound(prefix)+1e-15>=max([law.complete(w) for w in words if w.startswith(prefix)] or [0.])
            # actual unknown completions, outside vocabulary.
            assert bound(prefix)+1e-15>=law.complete(prefix+'z')
            count+=1
        e=Event(law,raw,b'root ',surface,.98,cap=4)
        old=rank(e,budget=2000,tight=False);new=rank(e,budget=2000,tight=True)
        assert old[1] and new[1] and old[0]==new[0]
        assert not rank(e,budget=1)[1]
        print(dict(baseline_nodes=old[2],tight_nodes=new[2],rank=new[0],certified=True))
    print(dict(actual_lift_blend_prefix_checks=count,insufficient_budget_uncertified=True))
if __name__=='__main__':main()
