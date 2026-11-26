import torch
import numpy as np
DIM=16
DEPTH=8

def excite(ids,code,lower,upper):
    e=code[ids]*(ids!=256).unsqueeze(-1)
    return .7*(e@lower).tanh(),.7*(e@upper).tanh()

def logits(h,ids,code,query,local,bias,prior):
    e=code[ids]*(ids!=256).unsqueeze(-1)
    z=h@query+torch.einsum('bri,rij->bj',e,local)
    return prior.log()+z@code[:256].T+bias

def preflight():
    torch.set_num_threads(2);torch.manual_seed(3301)
    c=torch.randn(257,DIM,dtype=torch.float64,requires_grad=True)
    lo=torch.zeros(DIM,128,dtype=c.dtype,requires_grad=True);hi=torch.zeros_like(lo,requires_grad=True)
    q=torch.zeros(128,DIM,dtype=c.dtype,requires_grad=True);loc=torch.zeros(DEPTH,DIM,DIM,dtype=c.dtype,requires_grad=True);b=torch.zeros(256,dtype=c.dtype,requires_grad=True)
    ids=torch.randint(0,256,(4,DEPTH));ids[0,0]=256;h=torch.randn(4,128,dtype=c.dtype);prior=torch.rand(4,256,dtype=c.dtype);prior/=prior.sum(1,keepdim=True)
    def output():
        a,z=excite(ids,c,lo,hi);return logits(h+(a+z).mean(1),ids,c,q,loc,b,prior)
    assert torch.allclose(output().softmax(1),prior,atol=1e-14,rtol=1e-14)
    with torch.no_grad():
        for p in (lo,hi,q,loc):p.normal_(0,.03)
    def loss():return torch.nn.functional.cross_entropy(output(),torch.tensor([0,255,32,10]))
    loss().backward();errors=[];norms=[]
    for p in (c,lo,hi,q,loc,b):
        norms.append(float(p.grad.norm()));idx=tuple(torch.unravel_index(p.grad.abs().argmax(),p.shape));g=float(p.grad[idx]);eps=1e-5
        with torch.no_grad():p[idx]+=eps;plus=float(loss());p[idx]-=2*eps;minus=float(loss());p[idx]+=eps
        errors.append(abs((plus-minus)/(2*eps)-g))
    assert min(norms)>0 and max(errors)<1e-8
    assert c.grad[256].abs().max()==0
    print({'zero_residual_parity':True,'grad_norms':norms,'max_gradient_error':max(errors),'padding_gradient_zero':True})
if __name__=='__main__':preflight()
