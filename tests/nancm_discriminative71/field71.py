"""v71 signed discriminative lexical transport, exact open-law conservation."""
import math,re
import numpy as np
import torch
from torch import nn
from field70 import words

def top(p,k):
    k=min(k,len(p));i=np.argpartition(-p,k-1)[:k]
    return sorted(i,key=lambda j:(-float(p[j]),int(j)))

def evidence(bank,root,a,b,direct,carry,prior,wrong=False):
    # Fixed candidate union, including parent challengers. Known-word indices
    # are zero based in Lift.p and one based in the inherited signed code.
    ids=set(int(i)+1 for law in (a,b) for i in top(law.p,32))
    for law in (direct,carry):
        if law:
            ids.update(bank.ids[w] for w in law.p if bank.ids.get(w,0)>0)
    ids=np.array(sorted(i for i in ids if bank.valid[i]),dtype=np.int64)
    h=words(root);hi=[bank.ids.get(w,0) for w in h[-128:]]
    if wrong and len(hi)>4:hi=hi[:-4][::-1]+hi[-4:]
    context=np.zeros((6,bank.code.shape[1]),np.float32)
    for j in range(min(3,len(hi))):context[j]=bank.code[hi[-1-j]]
    for j,(lo,high) in enumerate(((0,8),(8,32),(32,128)),3):
        chunk=hi[-high:len(hi)-lo if lo else None]
        if chunk:context[j]=bank.code[chunk].mean(0)
    p=np.stack((a.p[ids-1],b.p[ids-1]),0)
    dp=np.array([direct.complete(bank.vocab[i]) if direct else 0. for i in ids])
    cp=np.array([carry.complete(bank.vocab[i]) if carry else 0. for i in ids])
    pp=prior[ids];z=np.maximum(p.mean(0),1e-15)
    freq=np.array([h.count(bank.vocab[i]) for i in ids],np.float64)
    feat=np.stack((np.log(z)/12,np.log(np.maximum(pp,1e-15))/12,
        np.clip(np.log(z/np.maximum(pp,1e-15)),-12,12)/6,
        np.log1p(dp*100),np.log1p(cp*100),
        np.log1p(dp/z).clip(0,12)/6,np.log1p(cp/z).clip(0,12)/6,
        np.log1p(freq)/4,np.array([len(bank.vocab[i])/12 for i in ids]),
        np.clip(np.log(np.maximum(p[1],1e-15)/np.maximum(p[0],1e-15)),-12,12)/6),1)
    return ids,p,feat.astype(np.float32),context,bank.code[ids]

class Discriminator(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.q=nn.Linear(6*dim,32,bias=False);self.k=nn.Linear(dim,32,bias=False)
        self.net=nn.Sequential(nn.Linear(10+6+32,48),nn.SiLU(),nn.Linear(48,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,feat,ctx,code):
        q=self.q(ctx.flatten(-2));k=self.k(code)
        interaction=q.unsqueeze(-2)*k
        cos=torch.einsum('...nd,...rd->...nr',code,ctx)
        return 5*self.net(torch.cat((feat,cos,interaction),-1)).squeeze(-1).tanh()

def factors(logit,scale):return torch.exp(logit.double()*scale)

def tilt(law,ids,factor,Lift):
    if np.all(factor==1.):return law
    p=np.r_[law.unknown,law.p].copy();p[ids]*=factor;p/=math.fsum(p)
    return Lift(law.outside,p)

def ratio(p,factor,target):
    """p is (...,2,N), target is candidate offset or -1. Exact normalizer."""
    z=1.+(p*(factor.unsqueeze(-2)-1.)).sum(-1)
    ix=target.clamp_min(0).unsqueeze(-1)
    hit=factor.gather(-1,ix).squeeze(-1)
    hit=torch.where(target>=0,hit,torch.ones_like(hit))
    return hit.unsqueeze(-1)/z

def selftest():
    from open_word import Spelling,Outside,Lift
    o=Outside(['a','ab','b'],Spelling([['a','ab','b']]))
    a=Lift(o,np.array([.2,.3,.1,.4]));ids=np.array([1,3]);f=np.array([.2,4.])
    b=tilt(a,ids,f,Lift)
    assert tilt(a,ids,np.ones(2),Lift) is a
    for w in ('a','ab','b','aa','z'):
        expected=a.complete(w)*({'a':.2,'b':4.}.get(w,1.))/(1+.3*(-.8)+.4*3)
        assert abs(b.complete(w)-expected)<1e-14
    for p in ('','a','b','ab','z'):
        assert abs(b.mass(p)-b.complete(p)-sum(b.mass(p+c) for c in 'abcdefghijklmnopqrstuvwxyz'))<1e-12
    m=Discriminator(8);x=torch.randn(4,7,10);c=torch.randn(4,6,8);k=torch.randn(4,7,8)
    logits=m(x,c,k);assert torch.count_nonzero(logits)==0
    p=torch.full((4,2,7),.05,dtype=torch.float64)
    r=ratio(p,factors(logits,1.),torch.tensor([0,1,2,-1]));assert torch.equal(r,torch.ones_like(r))
    (-r.log().mean()).backward();assert m.net[-1].weight.grad.abs().sum()>0
    print('v71 exact tilt, unknown mass, signed gradient, normalization and rollback passed')
if __name__=='__main__':selftest()
