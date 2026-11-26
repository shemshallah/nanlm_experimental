"""Bounded document-local associative memory; write only after observation."""
import hashlib
import numpy as np
import torch
from collections import deque
CAP=1024
K=48
DIM=6

def bucket(w):return int.from_bytes(hashlib.blake2s(w.encode(),digest_size=4).digest(),'little')%16
def unit(x):return x/max(float(np.linalg.norm(x)),1e-8)
class Attractor:
    def __init__(self,code,ids,role,capacity=CAP):
        self.code=code;self.ids=ids;self.role=role;self.capacity=capacity;self.rows=deque(maxlen=capacity);self.h=[];self.t=0
    def key(self,h):
        a=np.zeros(16,np.float32);b=a.copy();c=a.copy()
        for i,w in enumerate(reversed(h[-8:])):
            weight=1/(1+i);a+=weight*self.code[self.ids.get(w,0)];b[bucket(w)]+=weight;c[int(self.role(w))%16]+=weight
        return np.stack([unit(a),unit(b),unit(c)])
    def observe(self,w):
        if self.h:self.rows.append((self.t,self.key(self.h),tuple(self.h[-4:]),w))
        self.h=(self.h+[w])[-128:];self.t+=1
    def query(self,h=None,min_age=8):
        h=self.h if h is None else h
        rows=[r for r in self.rows if self.t-r[0]>=min_age]
        f=np.zeros((K,DIM),np.float32);n=f.copy();valid=np.zeros(K,bool);words=['']*K
        if not h or not rows:return f,n,valid,words
        q=self.key(h);keys=np.stack([r[1] for r in rows]);sim=np.einsum('ncd,cd->nc',keys,q).clip(-1,1)
        match=np.array([next((k for k in range(min(4,len(h),len(r[2])),0,-1) if tuple(h[-k:])==r[2][-k:]),0) for r in rows])
        age=np.array([self.t-r[0] for r in rows]);rank=.8*sim[:,0]+.45*sim[:,1]+.25*sim[:,2]+.5*match
        order=np.argsort(-rank,kind='stable')[:K];l=len(order);s=sim[order];m=match[order]/4.;a=np.log1p(age[order])/np.log1p(self.capacity+1)
        f[:l]=np.c_[np.maximum(s,0),m,1-a,np.ones(l)]
        n[:l]=np.c_[np.maximum(-s,0),1-m,a,np.ones(l)]
        valid[:l]=True;words[:l]=[rows[i][3] for i in order]
        return f,n,valid,words
    def snapshot(self):return (self.t,list(self.h),[(t,k.copy(),h,w) for t,k,h,w in self.rows])
    def restore(self,s):self.t,self.h,rows=s;self.h=list(self.h);self.rows=deque([(t,k.copy(),h,w) for t,k,h,w in rows],maxlen=self.capacity)

def law(f,n,valid,support,oppose,gate,hidden=None,query=None):
    if hidden is not None:
        a,b=(hidden@query).chunk(2,-1)
        support=support+ .5*a.tanh()
        oppose=oppose+ .5*b.tanh()
        score=(f*torch.nn.functional.softplus(support)[:,None,:]).sum(-1)-(n*torch.nn.functional.softplus(oppose)[:,None,:]).sum(-1)
    else:score=f@torch.nn.functional.softplus(support)-n@torch.nn.functional.softplus(oppose)
    score=score.masked_fill(~valid,-1e9);p=score.softmax(-1)*valid
    p=p/p.sum(-1,keepdim=True).clamp_min(1e-12)
    best=(f*valid[...,None]).amax(1)
    if hidden is not None:best=torch.cat((best,hidden),-1)
    g=.65*torch.sigmoid(best@gate[:-1]+gate[-1])*valid.any(-1)
    return p,g

def preflight():
    torch.set_num_threads(2);rng=np.random.default_rng(36);code=rng.normal(size=(5,16)).astype(np.float32);code[0]=0
    ids={'alpha':1,'beta':2,'answer':3,'noise':4};role=lambda w:ids.get(w,0)
    a=Attractor(code,ids,role,capacity=64)
    for w in ['alpha','beta','answer']+['noise']*20+['alpha','beta']:a.observe(w)
    f,n,v,ws=a.query();assert ws[0]=='answer' and v.any()
    b=Attractor(code,ids,role,capacity=64);b.restore(a.snapshot());assert np.array_equal(b.query()[0],f) and b.query()[3]==ws
    for w in ['noise']*100:b.observe(w)
    assert len(b.rows)==64 and len(a.rows)<64
    empty=Attractor(code,ids,role);assert not empty.query()[2].any()
    t=lambda x:torch.tensor(x[None],dtype=torch.float64)
    s=torch.zeros(DIM,dtype=torch.float64,requires_grad=True);o=torch.zeros_like(s,requires_grad=True);g=torch.zeros(DIM+1,dtype=torch.float64,requires_grad=True)
    hit=t(np.array([w=='answer' for w in ws]));vv=torch.tensor(v[None])
    def loss():
        p,m=law(t(f),t(n),vv,s,o,g);return -((1-m)*.01+m*(p*hit).sum(-1)).log().mean()
    loss().backward();errors=[]
    for x in (s,o,g):
        assert x.grad.norm()>0;i=int(x.grad.abs().argmax());der=x.grad[i].item();eps=1e-5
        with torch.no_grad():x[i]+=eps;hi=loss().item();x[i]-=2*eps;lo=loss().item();x[i]+=eps
        errors.append(abs(der-(hi-lo)/(2*eps)))
    assert max(errors)<1e-8
    p,m=law(t(f),t(n),vv,s,o,g);assert abs(float(p.detach().sum())-1)<1e-12 and 0<float(m.detach())<.65
    p,m=law(t(f)*0,t(n)*0,torch.zeros_like(vv),s,o,g);assert p.sum()==0 and m.sum()==0
    print(dict(gradient_error=max(errors),retrieved=ws[0],reset=True,resume=True,eviction=True,empty_exact=True,age_min=8))
if __name__=='__main__':preflight()
