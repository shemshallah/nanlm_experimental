"""Exact candidate/complement probability allocation with a shared signed readout."""
import math
import numpy as np
import torch
from torch import nn

def allocate(prior, valid, lifts, mass_shift, innovation, strength=1.):
    p=prior.double(); valid=valid.bool(); z=lifts.double()
    assert p.shape==valid.shape==z.shape and bool((p>=0).all())
    assert bool((p[~valid]==0).all()) and bool(valid.any(-1).all())
    m=p.sum(-1); assert bool(((m>0)&(m<1)).all())
    if strength==0:return p,torch.ones_like(m)
    logp=torch.where(valid,p.clamp_min(1e-300).log(),-torch.inf)
    old=(logp+z*strength).softmax(-1)
    novel=(z*strength).masked_fill(~valid,-torch.inf).softmax(-1)
    a=innovation.double()*strength
    q=old*(1-a[:,None])+novel*a[:,None]
    mass=(torch.logit(m)+mass_shift.double()*strength).sigmoid()
    return mass[:,None]*q,(1-mass)/(1-m)

class AllocationLaw:
    def __init__(self,base,words,q,complement):
        self.base=base;self.complement=float(complement);self.delta={};self.prefix={}
        assert len(set(words))==len(words)==len(q)
        assert self.complement>=0 and all(v>=0 for v in q)
        for w,v in zip(words,q):
            assert w and w.isascii() and w.isalpha() and w.islower()
            d=float(v)-self.complement*base.complete(w);self.delta[w]=d
            for i in range(len(w)+1):self.prefix[w[:i]]=self.prefix.get(w[:i],0.)+d
        assert abs(self.mass('')-1)<1e-10
    def complete(self,w):return max(0.,self.complement*self.base.complete(w)+self.delta.get(w,0.))
    def mass(self,p):return max(0.,self.complement*self.base.mass(p)+self.prefix.get(p,0.))

def event_probs(q,complement,prior,member,base,surface,raw_literal,raw_folded,weight):
    p,member,base,surface,raw_literal,raw_folded=(a.double() for a in (prior,member,base,surface,raw_literal,raw_folded))
    rest=base-(p*member).sum(-1);assert bool((rest>=-1e-12).all())
    lex=complement*rest.clamp_min(0)+(q*member).sum(-1)
    return (1-weight)*raw_literal+weight*surface*lex,(1-weight)*raw_folded+weight*lex

class Budget(nn.Module):
    """Predict candidate occupancy/innovation from the SAME signed score and base mass."""
    def __init__(self):
        super().__init__(); self.gain=nn.Parameter(torch.zeros(2,6))
        with torch.no_grad():self.gain[1,0]=-4.
    def forward(self,p,valid,lift):
        p=p.double();v=valid.bool();l=lift.double();m=p.sum(-1)
        n=v.sum(-1).double();avg=(l*v).sum(-1)/n
        mx=l.masked_fill(~v,-torch.inf).max(-1).values
        mean=(p*l).sum(-1)/m
        x=torch.stack((torch.ones_like(m),torch.logit(m).clamp(-12,12)/12,n.log()/5,avg/4,mx/4,mean/4),-1)
        y=x@self.gain.double().T
        return 8*y[:,0].tanh(),y[:,1].sigmoid()
