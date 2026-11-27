"""Causal episodic donors and conserved source allocation."""
import numpy as np
import torch
from torch import nn

NAMES=('suffix','skip','topic')

def law(targets, scores, ages, total):
    if not len(targets): return {},np.zeros(5,np.float32)
    scores=np.asarray(scores,np.float64); weights=np.exp(scores-scores.max()); weights/=weights.sum()
    out={}
    for t,w in zip(targets,weights):out[int(t)]=out.get(int(t),0.)+float(w)
    features=np.array([np.log1p(len(targets))/8,max(out.values()),np.log1p(len(out))/8,1/(1+np.min(ages)/32),min(1.,total/8)],np.float32)
    return out,features

def experts(h):
    d=np.asarray(h,np.int64);n=len(d);t=np.arange(1,n);age=n-t
    # t is an ALREADY OBSERVED target position. Compare only preceding context.
    depth=np.zeros(len(t),np.int32);active=np.ones(len(t),bool)
    for j in range(1,min(8,n)+1):
        match=(t>=j)&(d[np.maximum(t-j,0)]==d[-j])&(d[-j]!=0)
        active &= match;depth+=active
    ok=(depth>0)&(d[t]>0)
    a=law(d[t[ok]],1.6*depth[ok]-.35*np.log1p(age[ok]),age[ok],int(depth.max()) if len(depth) else 0)
    # Non-contiguous syntax: current word and longer lag, distinct from exact phrase.
    s=np.zeros(len(t),np.float64)
    for j in (2,3,4,5,8):
        if n>=j:s+=(t>=j)&(d[np.maximum(t-j,0)]==d[-j])&(d[-j]!=0)
    ok=(t>=3)&(d[t-1]==d[-1])&(d[-1]!=0)&(s>0)&(d[t]>0) if n else np.zeros(len(t),bool)
    b=law(d[t[ok]],1.3*s[ok]-.5*np.log1p(age[ok]),age[ok],float(s[ok].max()) if ok.any() else 0.)
    t=np.arange(n);age=n-t;ok=d>0
    c=law(d[ok],-.8*np.log1p(age[ok]),age[ok],min(n,8))
    return a,b,c

class Credit(nn.Module):
    def __init__(self):
        super().__init__();self.gain=nn.Parameter(torch.zeros(3,6));self.gain.data[:,0]=-3.
    def weights(self,f,valid,strength=1.):
        logits=(self.gain[:,0]+(f*self.gain[:,1:]).sum(-1)).double().masked_fill(~valid,-torch.inf)
        p=torch.cat((torch.zeros_like(logits[:,:1]),logits),-1).softmax(-1)*strength
        return torch.cat((1-strength+p[:,:1],p[:,1:]),-1)
    def forward(self,p,f,valid,strength=1.):return (p.double()*self.weights(f,valid,strength)).sum(-1)

def mix(parent,laws,w):
    if not np.any(w[1:]):return parent
    q=w[0]*parent
    for law,a in zip(laws,w[1:]):
        if a:
            assert abs(sum(law.values())-1)<1e-9
            for t,p in law.items():q[t]+=a*p
    assert abs(q.sum()-1)<1e-8
    return q
