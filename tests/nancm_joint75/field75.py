"""v75: exact dense density-ratio coupling of inherited and generative word laws."""
import torch
from torch import nn
import torch.nn.functional as F

class Coupler(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate=nn.Linear(256,2)
        nn.init.zeros_(self.gate.weight)
        with torch.no_grad():self.gate.bias.copy_(torch.tensor([-2.,-3.]))
    def forward(self,base,logits,prior,state,scale=1.):
        # base: [batch, inherited experts, vocabulary including escape].
        # Every candidate receives a bounded signed density correction.
        if scale==0.:return base
        g=self.gate(state).sigmoid().double()
        beta=2*g[:,0:1]*scale;alpha=.25*g[:,1:2]*scale
        energy=(logits.double()-prior.double()).clamp(-12,12)
        energy=energy-energy.mean(-1,keepdim=True)
        f=(beta*energy).clamp(-8,8).exp().unsqueeze(1)
        tilted=base*f;tilted=tilted/tilted.sum(-1,keepdim=True)
        # A normalized exploration current prevents zero-base candidates
        # from being excluded forever, including the unknown spelling mass.
        gen=logits.double().softmax(-1).unsqueeze(1)
        return (1-alpha.unsqueeze(-1))*tilted+alpha.unsqueeze(-1)*gen

def likelihood_loss(p,target,base,event,word,weights):
    ix=target[:,None,None].expand(-1,2,1)
    t=p.gather(-1,ix).squeeze(-1)
    b=base.gather(-1,ix).squeeze(-1)
    # Known-word spelling/case and unknown escape spelling are unchanged.
    ratio=t/b.clamp_min(1e-300)
    r=torch.cat((torch.ones_like(ratio[:,:1]),ratio),-1)
    ev=-(weights*event*r).sum(-1).clamp_min(1e-300).log().mean()
    wd=-(weights*word*r).sum(-1).clamp_min(1e-300).log().mean()
    return ev,wd

def selftest():
    torch.manual_seed(75)
    c=Coupler();base=torch.rand(4,2,11,dtype=torch.float64);base/=base.sum(-1,keepdim=True)
    l=torch.randn(4,11,requires_grad=True);prior=torch.randn(11).log_softmax(-1);s=torch.randn(4,256)
    assert c(base,l,prior,s,0.) is base
    p=c(base,l,prior,s);assert torch.allclose(p.sum(-1),torch.ones(4,2,dtype=torch.float64),atol=1e-12)
    assert p.min()>0
    t=torch.tensor([0,2,4,5]);b=base.gather(-1,t[:,None,None].expand(-1,2,1)).squeeze(-1)
    q=torch.cat((torch.full((4,1),.01,dtype=torch.float64),b*.3),-1)
    w=torch.ones(4,3,dtype=torch.float64)/3
    ev,wd=likelihood_loss(p,t,base,q,q,w)
    expected=-(w*torch.cat((q[:,:1],p.gather(-1,t[:,None,None].expand(-1,2,1)).squeeze(-1)*.3),-1)).sum(-1).log().mean()
    assert torch.allclose(ev,expected,atol=1e-12)
    (ev+wd).backward();assert l.grad.abs().sum()>0 and c.gate.weight.grad.abs().sum()>0
    print('v75 dense normalization, escape, exact likelihood and joint gradient preflight passed')
if __name__=='__main__':selftest()
