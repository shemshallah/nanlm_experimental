import torch

def allocation(anchor,support,oppose):
    score=anchor*support.tanh()-anchor.square()*oppose.tanh()
    mass=score.softmax(-1)*anchor.shape[-1]
    return mass/(1+mass)

def route(gate,raw,uniform=False):
    if uniform:gate=torch.full_like(gate,.5)
    return 1+.02*raw.tanh()*(2*gate-1)

def read(points,gate,plus,minus,uniform=False):
    if uniform:gate=torch.full_like(gate,.5)
    a=(2*gate*points.relu()).mean(1)
    b=(2*gate*points.square()).mean(1)
    return .7*(a@plus-b@minus).tanh()

def preflight():
    torch.set_num_threads(2);torch.manual_seed(3501)
    x=torch.rand(3,8,256,dtype=torch.float64)*2-1
    y=torch.rand_like(x)*2-1
    s=torch.full((8,256),.1,dtype=x.dtype,requires_grad=True)
    o=torch.full_like(s,.1,requires_grad=True);r=torch.zeros_like(s,requires_grad=True)
    p=torch.zeros(256,128,dtype=x.dtype,requires_grad=True);n=torch.zeros_like(p,requires_grad=True)
    assert torch.equal(route(allocation(x,s,o),r),torch.ones_like(x))
    assert torch.equal(read(y,allocation(x,s,o),p,n),torch.zeros(3,128,dtype=x.dtype))
    with torch.no_grad():
        r.normal_(0,.2);p.normal_(0,.03);n.normal_(0,.03)
    def f():
        gate=allocation(x,s,o);z=route(gate,r)*y
        return read(z,gate,p,n).square().mean()
    f().backward();errs=[];norms=[]
    for a in (s,o,r,p,n):
        norms.append(float(a.grad.norm()));idx=tuple(torch.unravel_index(a.grad.abs().argmax(),a.shape));g=float(a.grad[idx]);eps=1e-5
        with torch.no_grad():a[idx]+=eps;hi=float(f());a[idx]-=2*eps;lo=float(f());a[idx]+=eps
        errs.append(abs((hi-lo)/(2*eps)-g))
    assert min(norms)>0 and max(errs)<1e-9
    with torch.no_grad():
        gate=allocation(x,s,o);assert gate.min()>0 and gate.max()<1
        assert not torch.allclose(read(y,gate,p,n),read(y,gate,p,n,True))
        assert route(gate,r).min()>=.98 and route(gate,r).max()<=1.02
    print(dict(exact_zero=True,gradient_errors=errs,gradient_norms=norms,gate_range=[float(gate.min()),float(gate.max())],uniform_distinct=True))
if __name__=='__main__':preflight()
