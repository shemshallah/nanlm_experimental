import torch

SCALES=(.035,.015)

def matrices(raw):
    i,j=torch.triu_indices(8,8,1,device=raw.device)
    a=raw.new_zeros((128,8,8))
    v=raw.tanh();a[:,i,j]=v;a[:,j,i]=v
    return a/(1+a.abs().sum(-1,keepdim=True))

def exchange(state,positive,negative):
    return SCALES[0]*torch.einsum('ncd,bdn->bcn',positive,state.relu())-SCALES[1]*torch.einsum('ncd,bdn->bcn',negative,state.square())

def preflight():
    torch.set_num_threads(2);torch.manual_seed(2901)
    x=torch.rand(4,8,128,dtype=torch.float64)*2-1
    p=torch.zeros(128,28,dtype=x.dtype,requires_grad=True);n=torch.zeros_like(p,requires_grad=True)
    assert torch.equal(exchange(x,matrices(p),matrices(n)),torch.zeros_like(x))
    with torch.no_grad(): p.uniform_(.03,.2);n.uniform_(.03,.2)
    def f():return exchange(x,matrices(p),matrices(n)).square().mean()
    val=f();val.backward();errs=[]
    for a in (p,n):
        assert a.grad.norm()>0
        idx=tuple(torch.unravel_index(a.grad.abs().argmax(),a.shape));g=a.grad[idx].item()
        with torch.no_grad():
            a[idx]+=1e-5;hi=f().item();a[idx]-=2e-5;lo=f().item();a[idx]+=1e-5
        errs.append(abs(g-(hi-lo)/2e-5))
    with torch.no_grad():
        inp=torch.zeros_like(x);inp[:,0,0]=.7;out=exchange(inp,matrices(p),matrices(n));assert out[:,1:,0].abs().max()>0
        y=torch.rand_like(x)*2-1
        a,b=matrices(p),matrices(n);observed=(exchange(x,a,b)-exchange(y,a,b)).abs().max()/(x-y).abs().max();assert observed<=.065+1e-12
    assert max(errs)<1e-10
    print({'zero_parity':True,'gradient_errors':errs,'cross_coordinate_nonzero':True,'observed_lipschitz':float(observed),'joint_bound':.905})
if __name__=='__main__':preflight()
