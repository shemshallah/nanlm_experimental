"""Full vocabulary signed scout with exact open-vocabulary event marginal."""
import bisect, math
import numpy as np
import torch
from torch import nn
from scout import Scout
from adaptive_retention import Retention

class DenseScout(nn.Module):
    def __init__(self,code,log_prior,hidden_dim=128):
        super().__init__()
        code=torch.as_tensor(code,dtype=torch.float32)
        self.scout=Scout(code)
        self.register_buffer('log_prior',torch.as_tensor(log_prior,dtype=torch.float32))
        self.retention=Retention()
        d=code.shape[1]
        self.memory_gain=nn.Parameter(torch.zeros(8))
        self.field_plus=nn.Parameter(torch.empty(hidden_dim,d))
        self.field_minus=nn.Parameter(torch.empty(hidden_dim,d))
        nn.init.normal_(self.field_plus,std=.03);nn.init.normal_(self.field_minus,std=.03)
    def forward(self,rails,m,c,b,h,memory=True):
        if memory:
            r,n=self.retention(m,c,b)
            rails=rails+(n/(n+4))[:,:,None]*self.memory_gain.sigmoid()[None,:,None]*r
        z=(h.relu()@self.field_plus-(-h).relu()@self.field_minus).tanh()
        return self.scout(rails)+2*(z@self.scout.code.T)+self.log_prior

class DenseLaw:
    """Sorted known-word distribution + immutable open-vocabulary parent."""
    def __init__(self,words,prob,parent,strength):
        assert 0<strength<1 and len(words)==len(prob) and len(words)>0
        self.words=words;self.prob=np.asarray(prob,np.float64).copy()
        assert (self.prob>=0).all() and abs(math.fsum(self.prob)-1)<1e-8
        self.prob/=math.fsum(self.prob)
        self.cum=np.r_[0.,np.cumsum(self.prob)];self.parent=parent;self.strength=float(strength)
    def complete(self,w):
        i=bisect.bisect_left(self.words,w)
        q=self.prob[i] if i<len(self.words) and self.words[i]==w else 0.
        return (1-self.strength)*self.parent.complete(w)+self.strength*float(q)
    def mass(self,p):
        if not p:return 1.
        a,b=interval(self.words,p,True);q=float(self.cum[b]-self.cum[a])
        # Avoid cancellation for a tiny prefix behind a high-mass prefix.
        if q<1e-10:q=math.fsum(self.prob[a:b])
        return (1-self.strength)*self.parent.mass(p)+self.strength*q

def fuse(words,prob,parent,strength):
    return parent if strength==0 else DenseLaw(words,prob,parent,strength)

def interval(words,p,prefix=False):
    a=bisect.bisect_left(words,p)
    if prefix:return a,bisect.bisect_left(words,p+'{')
    return a,a+int(a<len(words) and words[a]==p)

def event_prob(logits,lo,hi,base,surface,raw,folded,weight,strength):
    # Stable full-vocabulary reduction, including OOV, empty and capped events.
    q=logits.double().softmax(-1);idx=torch.arange(q.shape[-1],device=q.device)
    subset=(idx[None,:]>=lo[:,None])&(idx[None,:]<hi[:,None])
    dense=(q*subset).sum(-1)
    lex=(1-strength)*base.double()+strength*dense
    return (1-weight)*raw.double()+weight*surface.double()*lex,(1-weight)*folded.double()+weight*lex

def event_loss(logits,lo,hi,base,surface,raw,folded,length,weight,strength):
    lit,_=event_prob(logits,lo,hi,base,surface,raw,folded,weight,strength)
    return -lit.log().sum()/length.double().sum()
