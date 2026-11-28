"""v88 signed candidate competition driven by contextual likelihood ratios.
The signed field reads contextual evidence, not observed targets.
The only deployed modification is a normalized finite complete-word tilt.
"""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from field77 import tilt77,ratios
from field70 import words

def features88(prefix,a,b,ls,index,mode='exact',wrong=False,pool=None):
    """Fixed candidate union; wrong control permutes evidence across candidates."""
    h=words(prefix);code=index.code/4.;ids=index.ids
    if pool is None:
        # Exact context and base proposals both receive a fixed budget.
        candidates=set(w for w in h[-32:] if 0<len(w)<=63)
        for law,_,_ in ls[4:] if mode=='exact' else ls:
            if law:candidates.update(sorted(law.p,key=lambda z:(-law.p[z],z))[:12])
        # Use the known word baseline before the finite tilt; callers seed the
        # parent candidates separately (the composed law may not expose .p).
        pool=tuple(sorted(w for w in candidates if w.isascii() and w.isalpha() and w.islower()))
    p=np.array([[a.complete(w) for w in pool],[b.complete(w) for w in pool]],np.float64)
    wid=np.array([ids.get(w,0) for w in pool],np.int64)
    prior=np.maximum(index.prior[wid],1e-30)
    mean=np.maximum(p.mean(0),1e-300)
    donor=np.array([[l.complete(w) if l else 0. for w in pool] for l,_,_ in ls],np.float64).T
    caps=np.array([c for _,_,c in ls])
    if mode=='exact':donor[:,:4]=0.;caps[:4]=0.
    if wrong and len(pool)>1:donor=np.roll(donor,1,axis=0)
    # Evidence versus both prior and integrated concept distribution.
    # Zero support carries zero affirmative evidence rather than log(0).
    rr=np.where(donor>0,np.clip(np.log(np.maximum(donor,1e-300)/prior[:,None]),-12,12)/6,0.)
    pp=np.where(donor>0,np.clip(np.log(np.maximum(donor,1e-300)/mean[:,None]),-12,12)/6,0.)
    f=np.column_stack((np.log(mean)/18,np.log(prior)/18,np.clip(np.log(mean/prior),-12,12)/6,
        np.log1p(100*donor),rr,pp,np.broadcast_to(caps,(len(pool),10)),np.array([len(w)/16 for w in pool])))
    context=np.zeros((4,code.shape[1]),np.float32)
    hi=[ids.get(w,0) for w in h[-64:]]
    for j in range(min(3,len(hi))):context[j]=code[hi[-1-j]]
    if hi:context[3]=code[hi].mean(0)
    return pool,p,f.astype(np.float32),code[wid].astype(np.float32),context

class Field88(nn.Module):
    def __init__(self,d=128):
        super().__init__()
        self.query=nn.Linear(4*d,128,bias=False)
        self.key=nn.Linear(d,128,bias=False)
        self.drive=nn.Linear(44+128,256)
        self.exchange=nn.Linear(256,256,bias=False)
        self.read=nn.Sequential(nn.Linear(128+44,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.read[-1].weight);nn.init.zeros_(self.read[-1].bias)
    def forward(self,f,k,c,mask):
        q=self.query(c.flatten(-2));key=self.key(k)
        u=self.drive(torch.cat((f,key*q.unsqueeze(-2)),dim=-1))
        # Two competing 128-coordinate populations. The context evidence
        # changes a shared field; every candidate reads the resulting field.
        h=u.tanh()*mask.unsqueeze(-1)
        for _ in range(3):
            p,n=h.chunk(2,-1)
            # Positive and negative candidate population currents.
            ap=p.square().mean(-1).masked_fill(~mask,-torch.inf).softmax(-1)
            an=n.square().mean(-1).masked_fill(~mask,-torch.inf).softmax(-1)
            pool=torch.cat(((ap.unsqueeze(-1)*p).sum(-2),(an.unsqueeze(-1)*n).sum(-2)),-1)
            e=self.exchange(pool).unsqueeze(-2)
            p,n=(u+.5*e).chunk(2,-1)
            h=torch.cat((p.tanh()*torch.sigmoid(p-n),n.tanh()*torch.sigmoid(n-p)),-1)*mask.unsqueeze(-1)
        p,n=h.chunk(2,-1)
        return (4*self.read(torch.cat((f,p-n),-1)).squeeze(-1).tanh()).masked_fill(~mask,0.)

def apply88(a,b,pool,energy,gain):
    if gain==0:return a,b
    z=np.exp(gain*np.asarray(energy,np.float64))
    return tilt77(a,pool,z),tilt77(b,pool,z)

def objective88(field,data):
    p,f,k,c,mask,target,q,wp,w=data
    energy=field(f,k,c,mask)
    rr=ratios(p,energy,target)
    ratio=torch.cat((torch.ones_like(rr[...,:1]),rr),-1)
    ev=(q*ratio[:,0]*w).sum(-1).clamp_min(1e-300)
    probs=(wp*ratio*w[:,None,:]).sum(-1).clamp_min(1e-300)
    return -ev.log().mean(),-probs[:,0].log().mean(),F.softplus(probs[:,1:].log()-probs[:,:1].log()).mean()

def selftest():
    from open_word import Lift,Outside,Spelling,fuse
    from trajectory69 import Sparse
    torch.manual_seed(88)
    o=Outside(['a','b','c'],Spelling([['a','b','c']]))
    a=Lift(o,np.array([.1,.2,.3,.4]));b=fuse(a,Sparse({'a':.4,'z':.6}),.3)
    pool=['a','b','c','z'];p=torch.tensor([[l.complete(w) for w in pool] for l in (a,b)],dtype=torch.float64)[None]
    f=torch.randn(1,4,44);k=torch.randn(1,4,128);c=torch.randn(1,4,128);mask=torch.ones(1,4,dtype=torch.bool)
    net=Field88();z=net(f,k,c,mask);assert torch.count_nonzero(z)==0
    assert apply88(a,b,pool,z[0].detach().numpy(),0.)==(a,b)
    opt=torch.optim.AdamW(net.parameters(),lr=.003)
    q=torch.tensor([[.01,.02,.03]],dtype=torch.float64);wp=torch.cat((torch.full((1,4,1),.01,dtype=torch.float64),p.transpose(1,2)),2)
    weights=torch.tensor([[.25,.35,.4]],dtype=torch.float64);targets=torch.arange(4)[None]
    data=(p,f,k,c,mask,targets,q,wp,weights)
    initial=float(sum(objective88(net,data)).detach())
    for _ in range(24):
        opt.zero_grad();loss=sum(objective88(net,data));loss.backward();opt.step()
    assert float(sum(objective88(net,data)).detach())<initial
    z=net(f,k,c,mask)[0].detach().numpy();aa,bb=apply88(a,b,pool,z,.5)
    r=ratios(p,torch.tensor(z)[None],targets,.5).detach().numpy()[0]
    for j,w in enumerate(pool):
        assert abs(aa.complete(w)-a.complete(w)*r[j,0])<1e-12
        assert abs(bb.complete(w)-b.complete(w)*r[j,1])<1e-12
    for l in (aa,bb):
        for w in ('','a','b','z'):
            assert abs(l.mass(w)-l.complete(w)-math.fsum(l.mass(w+c) for c in 'abcdefghijklmnopqrstuvwxyz'))<1e-11
    print('v88 signed recurrent competition, optimization, exact finite tilt and OOV mass conservation passed')
if __name__=='__main__':selftest()
