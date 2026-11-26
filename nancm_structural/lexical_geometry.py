"""Tied lexical coordinate codebook: causal intake and candidate readout."""
import torch
import numpy as np
DIM=16
DEPTH=8

def excite(ids,code,lower,upper):
    emb=code[ids]*(ids!=0).unsqueeze(-1)
    return .7*(emb@lower).tanh(), .7*(emb@upper).tanh()

def score(hidden,query,code):
    return (hidden@query)@code.T

def preflight():
    torch.set_num_threads(2);torch.manual_seed(3201)
    code=torch.randn(37,DIM,dtype=torch.float64,requires_grad=True)
    lo=torch.zeros(DIM,128,dtype=code.dtype,requires_grad=True)
    hi=torch.zeros_like(lo,requires_grad=True);q=torch.zeros(128,DIM,dtype=code.dtype,requires_grad=True)
    ids=torch.randint(1,37,(3,DEPTH));h=torch.randn(3,128,dtype=code.dtype)
    a,b=excite(ids,code,lo,hi);assert torch.equal(a,torch.zeros_like(a)) and torch.equal(b,torch.zeros_like(b))
    assert torch.equal(score(h,q,code),torch.zeros(3,37,dtype=code.dtype))
    with torch.no_grad():
        for x in (lo,hi,q):x.normal_(0,.1)
    def loss():
        a,b=excite(ids,code,lo,hi);z=h+(a+b).mean(1)
        return torch.nn.functional.cross_entropy(score(z,q,code),torch.tensor([1,2,3]))
    loss().backward();errs=[];norms=[]
    for x in (code,lo,hi,q):
        norms.append(float(x.grad.norm()));idx=tuple(torch.unravel_index(x.grad.abs().argmax(),x.shape));g=float(x.grad[idx]);eps=1e-5
        with torch.no_grad():x[idx]+=eps;plus=float(loss());x[idx]-=2*eps;minus=float(loss());x[idx]+=eps
        errs.append(abs((plus-minus)/(2*eps)-g))
    assert min(norms)>0 and max(errs)<1e-8
    with torch.no_grad():
        a,b=excite(ids,code,lo,hi);aa,bb=excite(ids.flip(1),code,lo,hi);assert torch.equal(a.flip(1),aa)
        zz=excite(torch.zeros_like(ids),code,lo,hi);assert all(torch.count_nonzero(z)==0 for z in zz)
        c=code.detach().numpy();hh=h.numpy();qq=q.detach().numpy();up=np.random.default_rng(3).normal(size=(3,37))
    s=score(h,q,code);gd=torch.autograd.grad(s,(code,q),torch.tensor(up));project=hh@qq
    assert np.allclose(gd[0].numpy(),up.T@project) and np.allclose(gd[1].numpy(),hh.T@(up@c))
    print(dict(zero_parity=True,gradient_norms=norms,max_abs_gradient_error=max(errs),unknown_zero=True,rail_order_preserved=True,readout_gradient_parity=True))
if __name__=='__main__':preflight()
