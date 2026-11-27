"""v80: resume v78 addresses, attach a 64-word signed relational residual."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from field78 import Address

H=64
class Relation(nn.Module):
    def __init__(self):
        super().__init__()
        self.position=nn.Embedding(H,96)
        nn.init.normal_(self.position.weight,std=.02)
        self.q=nn.Linear(96,96,bias=False);self.k=nn.Linear(96,96,bias=False)
        self.value=nn.Linear(96,256)
        self.seed=nn.Linear(96,256)
        self.exchange=nn.Linear(256,256,bias=False)
        self.read=nn.Linear(256,96,bias=False)
        self.gate=nn.Linear(96,96)
        nn.init.zeros_(self.read.weight)
        nn.init.normal_(self.exchange.weight,std=.025)
    def forward(self,x,mask,z):
        # Each history position binds its ordered predecessor to its own
        # concept. The last position is always observed, never the target.
        x=x+self.position.weight[None]
        previous=F.pad(x[:,:-1],(0,0,1,0))
        x=x+.5*previous*F.pad(mask[:,:-1],(1,0)).unsqueeze(-1)
        q=self.q(z).reshape(-1,4,24)
        k=self.k(x).reshape(len(x),H,4,24).transpose(1,2)
        score=(q[:,:,None,:]*k).sum(-1)/np.sqrt(24.)
        valid=mask[:,None,:].expand_as(score)
        score=score.masked_fill(~valid,-1e4)
        # Four independent relational addresses, then shared signed update.
        a=score.softmax(-1)*valid
        a=a/a.sum(-1,keepdim=True).clamp_min(1e-9)
        values=self.value(x).reshape(len(x),H,4,64).transpose(1,2)
        u=(a[:,:,:,None]*values).sum(2).reshape(len(x),256)
        anchor=torch.tanh(self.seed(z)+u)
        # Three bounded recurrent exchanges. Absolute row norm prevents
        # runaway signed amplification; the anchor is reintroduced each pass.
        w=self.exchange.weight
        w=w/w.abs().sum(-1,keepdim=True).clamp_min(1.)
        state=anchor
        for _ in range(3):
            state=torch.tanh(anchor+.45*F.linear(state,w))
        p,n=state.chunk(2,-1)
        # Explicit mutually inhibitory signed contrast, preserving both rails.
        contrast=torch.cat((p*torch.sigmoid(p-n),n*torch.sigmoid(n-p)),-1)
        return self.read(contrast)*self.gate(z).sigmoid()

class Address80(Address):
    def __init__(self,*args):
        super().__init__(*args)
        self.relation=Relation()
        self.scale=1.
    def state(self,h):
        assert h.shape[-1] in (16,H)
        z=super().state(h[:,-16:])
        if self.scale==0. or h.shape[-1]==16:return z
        x=torch.tanh(self.project(self.embed(h)))
        return z+self.scale*self.relation(x,h!=0,z)
    def root(self,prefix,ids,words,wrong=False):
        h=[ids.get(w,0) for w in words(prefix)[-H:]]
        if wrong and len(h)>4:h=h[:-4][::-1]+h[-4:]
        x=np.zeros(H,np.int64)
        if h:x[-len(h):]=h
        return torch.from_numpy(x)[None]

class Positions:
    """Unique sampled positions; no cross-document histories."""
    def __init__(self,seqs,seed,probe_books=0):
        self.seqs=seqs;self.rng=np.random.default_rng(seed)
        self.offset=np.cumsum([0]+[max(0,len(s)-1) for s in seqs])
        self.order=self.rng.permutation(np.arange(self.offset[probe_books],self.offset[-1]))
        self.at=0
    def take(self,n):
        assert self.at+n<=len(self.order)
        pos=self.order[self.at:self.at+n];self.at+=n
        books=np.searchsorted(self.offset,pos,side='right')-1
        x=np.zeros((n,H),np.int64);y=np.zeros(n,np.int64)
        for i,b in enumerate(books):
            j=int(pos[i]-self.offset[b]+1);s=self.seqs[b];h=s[max(0,j-H):j]
            x[i,-len(h):]=h;y[i]=s[j]
        return torch.from_numpy(x),torch.from_numpy(y)

def selftest():
    torch.manual_seed(80);rng=np.random.default_rng(80)
    code=rng.normal(size=(13,8)).astype(np.float32);code[0]=0
    members=np.array([[0,-1,-1,-1],[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    cls=np.repeat([0,1,2,3],[1,4,4,4]);slot=np.r_[0,np.tile(np.arange(4),3)]
    args=(code,np.ones(13)/13,members,cls,slot,np.zeros((13,6),np.float32))
    old=Address(*args);m=Address80(*args)
    miss=m.load_state_dict(old.state_dict(),strict=False)
    assert not miss.unexpected_keys and all(k.startswith('relation.') for k in miss.missing_keys)
    docs=[np.tile([1,2,3,4,5,6],40),np.tile([7,8,9,10,11,12],40)]
    pos=Positions(docs,80);h,y=pos.take(32)
    assert torch.equal(m.log_probs(h),old.log_probs(h[:,-16:]))
    with torch.no_grad():m.relation.read.weight.normal_(std=.03)
    m.scale=0.
    assert torch.equal(m.log_probs(h),old.log_probs(h[:,-16:]))
    m.scale=1.
    z=m.log_probs(h);t=m.target_log(h,y)
    assert torch.allclose(z[torch.arange(len(y)),y],t,atol=2e-6)
    assert torch.allclose(z.exp().sum(-1),torch.ones(len(y)),atol=2e-6)
    (-t.mean()).backward()
    assert m.relation.exchange.weight.grad.abs().sum()>0
    assert m.delta.weight.grad.abs().sum()>0
    assert pos.at==32 and len(np.unique(pos.order))==len(pos.order)
    print('v80 inherited exact rollback, full/target parity, signed exchange gradient, unique positions passed')
if __name__=='__main__':
    torch.set_num_threads(2);selftest()
