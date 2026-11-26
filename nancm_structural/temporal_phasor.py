"""Causal signed phasor memory. One-word streaming and differentiable closed-form scan."""
import torch
import math
LENGTH=96
def coefficients(raw_decay,raw_phase,raw_amp):
    base=raw_decay.new_tensor([2.,8.,32.,128.])[None,:,None]
    rho=torch.exp(-math.log(2)/(base*torch.exp(raw_decay.tanh())))
    return rho,.5*raw_phase.tanh(),raw_amp.sigmoid()
def scan(events,rho,theta,gain):
    B,T,_,_=events.shape;dtype=rho.dtype
    age=torch.arange(T-1,-1,-1,dtype=dtype,device=rho.device)[None,:,None,None]
    idx=events.clamp_max(127);ind=idx.unsqueeze(-1)
    gather=lambda a:a[None,None].expand(B,T,-1,-1,-1).gather(-1,ind).squeeze(-1)
    rr,th,gg=map(gather,(rho,theta,gain))
    w=(1-rr)*gg*rr**age*(events<128)
    src=torch.stack([w*torch.cos(age*th),w*torch.sin(age*th)],-1).permute(0,2,3,1,4)
    return rho.new_zeros((B,2,4,128,2)).scatter_add(3,idx.permute(0,2,3,1).unsqueeze(-1).expand(-1,-1,-1,-1,2),src)
def step(state,event,rho,theta,gain):
    a,b=state.unbind(-1);co=theta.cos();si=theta.sin()
    u=torch.zeros_like(a).scatter_add(-1,event.clamp_max(127).unsqueeze(-1),(event<128).to(a.dtype).unsqueeze(-1))
    return torch.stack([rho*(co*a-si*b)+(1-rho)*gain*u,rho*(si*a+co*b)],-1)
def contrast(state):
    # Distinguish supporting signed phase from independently accumulated opposition.
    a,b=state.unbind(1)
    return (a-b*b.abs()).permute(0,1,3,2).reshape(len(state),8,128)
def preflight():
    torch.set_num_threads(2);torch.manual_seed(3001)
    p=torch.zeros(2,4,128,dtype=torch.float64,requires_grad=True)
    t=torch.full_like(p,.13,requires_grad=True);g=torch.zeros_like(p,requires_grad=True)
    x=torch.randint(0,128,(3,LENGTH,2,4));x[:,:7]=128
    rho,theta,gain=coefficients(p,t,g);out=scan(x,rho,theta,gain);state=torch.zeros_like(out)
    for i in range(LENGTH):state=step(state,x[:,i],rho,theta,gain)
    err=float((out-state).detach().abs().max());assert err<1e-12
    assert float(out.detach().norm(dim=-1).max())<=1
    loss=out.square().mean();loss.backward();norms=[float(a.grad.norm()) for a in (p,t,g)];assert min(norms)>0
    with torch.no_grad():
        rev=scan(x.flip(1),rho,theta,gain);assert not torch.allclose(out,rev)
        # Identical prefix -> identical state, independent of unseen suffix.
        half=scan(x[:,:40],rho,theta,gain);other=x.clone();other[:,40:]=0
        assert torch.equal(half,scan(other[:,:40],rho,theta,gain))
        assert torch.equal(scan(torch.full_like(x,128),rho,theta,gain),torch.zeros_like(out))
        restored=state.clone();assert torch.equal(step(state,x[:,0],rho,theta,gain),step(restored,x[:,0],rho,theta,gain))
    print({"stream_scan_error":err,"gradient_norms":norms,"order_sensitive":True,"reset":True,"suffix_invariant":True,"resume_exact":True})
if __name__=="__main__":preflight()
