"""v72 full-vocabulary signed generative field, two prospective horizons."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from field70 import words

H=32
class Generator(nn.Module):
    def __init__(self,code,prior,d=64):
        super().__init__()
        code=torch.as_tensor(code,dtype=torch.float32)
        self.register_buffer('code',code/4.)
        self.register_buffer('prior',torch.as_tensor(prior,dtype=torch.float32).log())
        v,dim=code.shape
        self.lex=nn.Embedding(v,d,padding_idx=0)
        nn.init.normal_(self.lex.weight,std=.08)
        with torch.no_grad():self.lex.weight[0].zero_()
        self.input=nn.Linear(dim+d,128)
        self.recur=nn.GRU(128,128,batch_first=True)
        self.pos=nn.Linear(128+4*(d+dim),128)
        self.neg=nn.Linear(128+4*(d+dim),128)
        self.exchange=nn.Linear(256,256)
        self.out=nn.Linear(256,d)
        self.future=nn.Linear(256,d)
        self.semantic=nn.Linear(dim,d,bias=False)
        self.bias=nn.Parameter(torch.zeros(v))
    def state(self,h):
        x=torch.cat((self.lex(h),self.code[h]),-1)
        # Chronological local recurrence and four nested history bands.
        _,r=self.recur(torch.tanh(self.input(x[:,-8:])))
        bands=[]
        for lo,hi in ((0,1),(1,4),(4,12),(12,H)):
            z=x[:,H-hi:H-lo];mask=(h[:,H-hi:H-lo]!=0).unsqueeze(-1)
            bands.append((z*mask).sum(1)/mask.sum(1).clamp_min(1))
        z=torch.cat((r[0],*bands),-1)
        p,n=F.softplus(self.pos(z)),F.softplus(self.neg(z))
        pair=torch.cat((p,n),-1)
        a,b=self.exchange(pair).chunk(2,-1)
        # Inhibition acts on the opposing channel, not a second word head.
        p=p*torch.sigmoid(a-b);n=n*torch.sigmoid(b-a)
        return torch.cat((p,n),-1)
    def logits(self,h,prospective=False):
        z=self.state(h)
        keys=self.lex.weight+self.semantic(self.code)
        base=self.prior+self.bias
        q=self.out(z)
        a=q@keys.T+base
        return (a,self.future(z)@keys.T+base) if prospective else a
    def forward(self,h):return self.logits(h).softmax(-1)
    def root(self,prefix,ids,wrong=False):
        h=np.array([ids.get(w,0) for w in words(prefix)[-H:]],np.int64)
        x=np.zeros(H,np.int64)
        if len(h):x[-len(h):]=h
        if wrong:
            # Keep the observed multiset, reverse its temporal order.
            nz=np.flatnonzero(x);start=nz[0] if len(nz) else H
            x[start:]=x[start:][::-1]
        return torch.from_numpy(x).unsqueeze(0)

def batch(seqs,rng,n):
    # Sample proportionally to book length, without crossing book boundaries.
    lengths=np.array([max(0,len(s)-2) for s in seqs]);p=lengths/lengths.sum()
    x=np.zeros((n,H),np.int64);y=np.zeros((n,2),np.int64)
    for j,i in enumerate(rng.choice(len(seqs),size=n,p=p)):
        s=seqs[i];t=int(rng.integers(1,len(s)-1));h=s[max(0,t-H):t]
        x[j,-len(h):]=h;y[j]=s[t:t+2]
    return torch.from_numpy(x),torch.from_numpy(y)

def selftest():
    rng=np.random.default_rng(72);torch.manual_seed(72)
    code=rng.normal(size=(12,128)).astype(np.float32);code[0]=0
    m=Generator(code,np.ones(12)/12);o=torch.optim.AdamW(m.parameters(),lr=.01)
    # Repeated deterministic paths, next-word and next-next-word targets.
    docs=[np.tile([1,2,3,4,5,6],40),np.tile([7,8,9,10,11],40)]
    h,y=batch(docs,rng,32)
    a,b=m.logits(h,True);initial=float(F.cross_entropy(a,y[:,0]).detach())
    for _ in range(40):
        o.zero_grad();a,b=m.logits(h,True);loss=F.cross_entropy(a,y[:,0])+.25*F.cross_entropy(b,y[:,1])
        loss.backward();o.step()
    p=m(h);assert torch.allclose(p.sum(-1),torch.ones(len(h)),atol=1e-6)
    final=float(F.cross_entropy(m.logits(h),y[:,0]).detach())
    assert final<initial*.5 and all(torch.isfinite(x).all() for x in m.parameters())
    assert m.root(b'a b c ',{'a':1,'b':2,'c':3})[0,-3:].tolist()==[1,2,3]
    print('v72 full-vocabulary prospective gradient and causal fixture',initial,final)
if __name__=='__main__':
    torch.set_num_threads(2);selftest()
