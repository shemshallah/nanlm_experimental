"""Signed eight-rail field with exact factorized lexical probability."""
import numpy as np
import torch
from torch import nn


def partition(code, prior, cap=128):
    code=np.asarray(code);prior=np.asarray(prior);n=len(code)
    assert n==len(prior) and n>1 and cap>0 and np.all(prior>0)
    # Deterministic signed geometry classes, capped to bound training work.
    axes=np.argmax(np.abs(code[1:]),axis=1)
    keys=2*axes+(code[1+np.arange(n-1),axes]<0)
    groups=[[0]]
    for k in np.unique(keys):
        ix=np.flatnonzero(keys==k)+1
        ix=ix[np.argsort(-prior[ix],kind='stable')]
        groups.extend(ix[i:i+cap].tolist() for i in range(0,len(ix),cap))
    width=max(map(len,groups));members=np.full((len(groups),width),-1,np.int64)
    cls=np.empty(n,np.int64);slot=np.empty(n,np.int64)
    for g,row in enumerate(groups):
        members[g,:len(row)]=row;cls[row]=g;slot[row]=np.arange(len(row))
    assert np.array_equal(np.sort(members[members>=0]),np.arange(n))
    return members,cls,slot

class Field(nn.Module):
    def __init__(self,code,prior,cap=128,rank=24,identity=24):
        super().__init__();code=np.asarray(code,np.float32);prior=np.asarray(prior,np.float64);prior=prior/prior.sum()
        members,cls,slot=partition(code,prior,cap);d=code.shape[1]
        for k,v in [('code',code),('members',members),('cls',cls),('slot',slot),('logprior',np.log(prior).astype(np.float32)),('classprior',np.log(np.bincount(cls,weights=prior)).astype(np.float32))]:self.register_buffer(k,torch.as_tensor(v))
        self.identity=nn.Embedding(len(code),identity,padding_idx=0);nn.init.normal_(self.identity.weight,std=.04)
        with torch.no_grad():self.identity.weight[0].zero_()
        self.id_in=nn.Linear(identity,d,bias=False);self.id_out=nn.Linear(d,identity,bias=False)
        self.plus_a=nn.Parameter(torch.randn(8,d,rank)*.04);self.plus_b=nn.Parameter(torch.randn(8,rank,d)*.04)
        self.minus_a=nn.Parameter(torch.randn(8,d,rank)*.04);self.minus_b=nn.Parameter(torch.randn(8,rank,d)*.04)
        self.cross_plus=nn.Parameter(torch.zeros(8,8));self.cross_minus=nn.Parameter(torch.zeros(8,8))
        self.route=nn.Linear(d,1,bias=False);self.class_head=nn.Linear(d,len(members));self.word_query=nn.Linear(d,d,bias=False)
    def hidden(self,x,ids,coupled=True):
        x=torch.cat((x[:,:4]+self.id_in(self.identity(ids)),x[:,4:]),1)
        lo=x.tanh();seed=torch.einsum('bri,rij,rjk->brk',lo.relu(),self.plus_a,self.plus_b)-torch.einsum('bri,rij,rjk->brk',(-lo).relu(),self.minus_a,self.minus_b);hi=seed.tanh()
        if coupled:
            # Bounded signed cross-rail transfers; fixed anchor each pass.
            a=self.cross_plus.tanh();b=self.cross_minus.tanh();a=.25*a/(1+a.abs().sum(1,keepdim=True));b=.15*b/(1+b.abs().sum(1,keepdim=True))
            for _ in range(2):hi=(seed+torch.einsum('rs,bsd->brd',a,hi.relu())-torch.einsum('rs,bsd->brd',b,hi.square())).tanh()
        gate=self.route(lo+hi).squeeze(-1).softmax(-1)
        return ((lo+hi)*gate[...,None]).sum(1)
    def class_log(self,h):return (self.class_head(h)+self.classprior).log_softmax(-1)
    def target_logprob(self,x,ids,y,coupled=True):
        h=self.hidden(x,ids,coupled);c=self.cls[y];s=self.slot[y];rows=self.members[c];safe=rows.clamp_min(0)
        z=torch.einsum('bd,bkd->bk',self.word_query(h),self.code[safe])+torch.einsum('bd,bkd->bk',self.id_out(h),self.identity(safe))+self.logprior[safe]
        z=z.masked_fill(rows<0,-torch.inf);i=torch.arange(len(y),device=y.device)
        return self.class_log(h)[i,c]+z.log_softmax(-1)[i,s]
    def log_probs(self,x,ids,coupled=True):
        h=self.hidden(x,ids,coupled);scores=self.word_query(h)@self.code.T+self.id_out(h)@self.identity.weight.T+self.logprior
        rows=self.members;safe=rows.clamp_min(0);z=scores[:,safe].masked_fill(rows[None]<0,-torch.inf)
        z=z.log_softmax(-1)+self.class_log(h)[:,:,None]
        return z[:,self.cls,self.slot]


def preflight():
    torch.set_num_threads(2);torch.manual_seed(50);rng=np.random.default_rng(50)
    code=rng.normal(size=(37,16)).astype(np.float32);code[0]=0;prior=rng.random(37)+.1;prior/=prior.sum()
    model=Field(code,prior,cap=5,rank=4,identity=4).double();x=torch.randn(9,8,16,dtype=torch.float64);ids=torch.randint(0,37,(9,4));y=torch.randint(0,37,(9,));y[0]=0
    dense=model.log_probs(x,ids);sparse=model.target_logprob(x,ids,y)
    err=float((dense[torch.arange(len(y)),y]-sparse).detach().abs().max());mass=float((dense.exp().sum(-1)-1).detach().abs().max());assert max(err,mass)<1e-12
    loss=-sparse.mean();loss.backward();norms={k:float(p.grad.norm()) for k,p in model.named_parameters() if p.grad is not None};assert min(norms.values())>0
    fd=0.
    for k,p in model.named_parameters():
        ix=tuple(torch.unravel_index(p.grad.abs().argmax(),p.shape));g=float(p.grad[ix]);v=float(p[ix].detach());eps=1e-5
        with torch.no_grad():p[ix]=v+eps;hi=float(-model.target_logprob(x,ids,y).mean());p[ix]=v-eps;lo=float(-model.target_logprob(x,ids,y).mean());p[ix]=v
        fd=max(fd,abs((hi-lo)/(2*eps)-g))
    assert fd<1e-7
    with torch.no_grad():
        assert torch.equal(model.hidden(x,ids,True),model.hidden(x,ids,False))
        assert torch.allclose(model.log_probs(x,ids,False).exp().sum(-1),torch.ones(9,dtype=x.dtype),atol=1e-12,rtol=0)
    print(dict(vocabulary=37,groups=len(model.members),parameters=sum(p.numel() for p in model.parameters()),target_logprob_max_error=err,normalization_max_error=mass,gradient_groups=len(norms),gradient_max_error=fd,min_gradient_norm=min(norms.values())))
if __name__=='__main__':preflight()
