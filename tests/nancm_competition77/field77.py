"""v77 candidate-level signed competition over exact open word laws."""
import math
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from field70 import words

class Tilt:
    """Change finite complete-word masses, leaving all others proportional."""
    def __init__(self,base,candidates,factors):
        self.base=base;self.f=dict(zip(candidates,map(float,factors)))
        assert all(math.exp(-4)-1e-12<=v<=math.exp(4)+1e-12 for v in self.f.values())
        corrections=defaultdict(list)
        for w,v in self.f.items():
            d=base.complete(w)*(v-1.)
            for i in range(len(w)+1):corrections[w[:i]].append(d)
        self.delta={p:math.fsum(v) for p,v in corrections.items()}
        self.z=1.+self.delta.get('',0.);assert self.z>0
    def complete(self,w):return self.base.complete(w)*self.f.get(w,1.)/self.z
    def mass(self,p):
        if not p:return 1.
        return math.fsum((self.base.mass(p),self.delta.get(p,0.)))/self.z

def tilt77(base,candidates,factors):
    return base if np.all(factors==1.) else Tilt(base,candidates,factors)

def candidate_features(prefix,a75,b75,a,b,laws,index,wrong=False,pool=None):
    h=words(prefix)[-1024:];ids=index.ids;code=index.code/4.
    if pool is None:
        base=.5*(a75.p+b75.p)
        top=np.argsort(-base,kind='stable')[:48]
        candidates=set(a75.words[i] for i in top)
        for law,_,_ in laws:
            if law:candidates.update(sorted(law.p,key=lambda w:(-law.p[w],w))[:32])
        candidates.update(reversed(h[-96:]))
        pool=tuple(sorted(w for w in candidates if 0<len(w)<=63 and w.isascii() and w.isalpha()))
    # Keep the same candidate set and donor evidence in the counterfactual;
    # perturb only the chronology used to compute occurrence compatibility.
    hh=h[:-3][::-1]+h[-3:] if wrong and len(h)>3 else h
    hi=np.array([ids.get(w,0) for w in hh],np.int64)
    loc=defaultdict(list)
    for j,w in enumerate(hh):loc[w].append(j)
    current=np.zeros((3,code.shape[1]),np.float32)
    for k in range(min(3,len(hi))):current[k]=code[hi[-1-k]]
    topic=code[hi[-32:]].mean(0) if len(hi) else np.zeros(code.shape[1],np.float32)
    feats=[];keys=[];bp=[]
    for w in pool:
        p=np.array([a.complete(w),b.complete(w)]);bp.append(p)
        mean=max(float(p.mean()),1e-300);wid=ids.get(w,0);prior=float(index.prior[wid])
        donor=np.array([law.complete(w) if law else 0. for law,_,_ in laws])
        positions=loc[w];n=len(hh);d=n-positions[-1] if positions else 2048
        best=0.;exact=0.;recent=0.
        for j in positions[-16:]:
            context=np.zeros_like(current)
            for k in range(min(3,j)):context[k]=code[hi[j-1-k]]
            score=float((context*current).sum(-1)@np.array([.6,.3,.1]))
            best=max(best,score);recent=score
            for k in range(1,min(3,j,n)+1):
                if hh[j-k:j]==hh[-k:]:exact=max(exact,k)
        feats.append([math.log(mean)/18.,math.log(max(prior,1e-300))/18.,
            np.clip(math.log(mean/max(prior,1e-300)),-12,12)/6,
            math.log1p(len(positions))/5.,math.log1p(d)/8.,
            float(bool(positions)),best,recent,exact/3.,len(w)/16.,
            float(code[wid]@topic),*np.log1p(100*donor),
            *np.clip(np.log1p(donor/mean),0,12)/6])
        keys.append(code[wid])
    return pool,np.asarray(bp,np.float64).T,np.asarray(feats,np.float32),np.asarray(keys,np.float32),current,topic

class Competition(nn.Module):
    def __init__(self,dim=128):
        super().__init__()
        self.q=nn.Linear(4*dim,32,bias=False);self.k=nn.Linear(dim,32,bias=False)
        self.net=nn.Sequential(nn.Linear(19+32,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,f,k,c,t):
        q=self.q(torch.cat((c.flatten(-2),t),-1))
        interaction=q.unsqueeze(-2)*self.k(k)
        return 4*self.net(torch.cat((f,interaction),-1)).squeeze(-1).tanh()

def ratios(p,energy,targets,scale=1.):
    # p [B,2,N], target indices [B,K], -1 means outside finite pool.
    fac=(energy.double()*scale).exp()
    z=1.+(p*(fac[:,None,:]-1.)).sum(-1)
    hit=fac.gather(-1,targets.clamp_min(0))
    hit=torch.where(targets>=0,hit,torch.ones_like(hit))
    return hit[:,:,None]/z[:,None,:]

def selftest():
    from open_word import Outside,Spelling,Lift
    o=Outside(['a','ab','b'],Spelling([['a','ab','b']]))
    p=Lift(o,np.array([.2,.3,.1,.4]))
    w=['a','b','z'];f=np.exp(np.array([-4.,2.,3.]))
    q=tilt77(p,w,f);assert tilt77(p,w,np.ones(3)) is p
    for x in ('','a','ab','b','z','zz'):
        assert abs(q.mass(x)-q.complete(x)-sum(q.mass(x+c) for c in 'abcdefghijklmnopqrstuvwxyz'))<1e-12
    bp=torch.tensor([[p.complete(x) for x in w]]*2,dtype=torch.float64)[None]
    r=ratios(bp,torch.tensor([[-4.,2.,3.]]),torch.tensor([[0,1,2,-1]]))
    for i,x in enumerate(w+['ab']):assert abs(float(r[0,i,0])*p.complete(x)-q.complete(x))<1e-12
    m=Competition(8);ff=torch.randn(2,3,19);k=torch.randn(2,3,8);c=torch.randn(2,3,8);t=torch.randn(2,8)
    z=m(ff,k,c,t);assert torch.count_nonzero(z)==0
    (-z[:,0].mean()).backward();assert m.net[-1].weight.grad.abs().sum()>0
    print('v77 exact signed mass, OOV tilt, target ratios, identity and gradient passed')
if __name__=='__main__':selftest()
