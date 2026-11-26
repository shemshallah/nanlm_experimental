import itertools,math,functools
import numpy as np
import torch
import folded_event as FE
from symbiotic_law import Transport
from event_credit import probabilities,objective,factors

class Word:
    d={'a':.2,'an':.3,'ant':.4,'ante':.1}
    def mass(self,p):return math.fsum(v for w,v in self.d.items() if w.startswith(p))
    def complete(self,w):return self.d.get(w,0.)

@functools.lru_cache(None)
def raw(s):
    rng=np.random.default_rng(int.from_bytes(s,'little')%2**32);q=rng.random(256)+.02
    return q/q.sum()

def test():
    torch.set_num_threads(2);torch.manual_seed(45)
    base=Word();words=['a','ant','ante'];prior=np.array([base.complete(w) for w in words]);surface=FE.Surface([b'An ANT! a an. a\n'])
    donor=FE.Event(base,raw,b'root ',surface,0.,cap=3)
    events=[b'A ',b'an!',b'AnT',b'zzz',b'xy?',b'!',b'\xff']
    fs=[factors(e,words,base,donor) for e in events]
    member,b,s,r,f,n=(torch.as_tensor(np.asarray(a),dtype=torch.float64) for a in zip(*fs))
    p=torch.as_tensor(np.tile(prior,(len(events),1)),dtype=torch.float64)
    lift=(torch.randn(len(events),len(words),dtype=torch.float64)*.7).requires_grad_()
    err=0.;fold_err=0.;token_err=0.
    for weight in (0.,.25,.63,.9):
        literal,folded=probabilities(lift,p,member,b,s,r,f,weight)
        for i,event in enumerate(events):
            law=Transport(base,words,lift[i].detach().numpy());e=FE.Event(law,raw,b'root ',surface,weight,cap=3)
            err=max(err,abs(float(literal[i].detach())-e.mass(event)))
            letters=event if FE.IS_LETTER[event[-1]] else event[:-1]
            if not letters: expected=sum(e.mass(bytes([x])) for x in FE.STOP)
            elif FE.IS_LETTER[event[-1]]:expected=e.prefix_mass(letters.lower())
            else:expected=e.complete(letters.lower())
            fold_err=max(fold_err,abs(float(folded[i].detach())-expected))
            q=e.next(b'');t=e.tokens(b'');m=np.zeros(256)
            for tok,v in t.items():m[tok[0]]+=v
            token_err=max(token_err,float(np.abs(m-q).max()))
        assert torch.allclose(objective(lift,p,member,b,s,r,f,n,weight),-literal.log().sum()/n.sum(),atol=1e-14,rtol=0)
    assert max(err,fold_err,token_err)<1e-12
    def loss():return objective(lift,p,member,b,s,r,f,n,.63)
    loss().backward();grad=lift.grad.clone();fd=0.
    for ix in itertools.product(range(len(events)),range(len(words))):
        with torch.no_grad():lift[ix]+=1e-5;hi=float(loss());lift[ix]-=2e-5;lo=float(loss());lift[ix]+=1e-5
        fd=max(fd,abs((hi-lo)/2e-5-float(grad[ix])))
    assert fd<1e-8 and grad.norm()>0
    assert torch.count_nonzero(grad[-2:])==0
    z=lift.detach().clone().requires_grad_();objective(z,p,member,b,s,r,f,n,0.).backward();assert torch.count_nonzero(z.grad)==0
    print(dict(events=len(events),weights=4,exact_literal_max_error=err,folded_max_error=fold_err,token_marginal_max_error=token_err,gradient_max_error=fd,gradient_norm=float(grad.norm()),zero_weight_gradient_exact=True,empty_gradient_exact=True))
if __name__=='__main__':test()
