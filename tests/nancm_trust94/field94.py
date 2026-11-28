"""v94: trust-region continuation of the trained v88 signed field."""
import torch
from torch import nn
import torch.nn.functional as F
from field88 import Field88
from field77 import ratios

class Trust94(nn.Module):
    def __init__(self,field):
        super().__init__();self.field=field
        self.theta=nn.Parameter(torch.zeros(8,dtype=torch.float64))
        # Maximum gain .5, initial gain .125 exactly in real arithmetic.
        with torch.no_grad():self.theta[0]=-1.0986122886681098
    def forward(self,f,k,c,mask):
        e=self.field(f,k,c,mask).double()
        n=mask.sum(-1).clamp_min(1)
        avg=lambda x:(x*mask).sum(-1)/n
        mx=lambda x:x.masked_fill(~mask,-torch.inf).amax(-1)
        features=torch.stack((torch.ones_like(e[...,0]),n.double().log()/5.,
            avg(e.abs())/4.,mx(e.abs())/4.,
            avg(f[...,0].double()),mx(f[...,0].double()),
            avg(f[...,3:13].double().amax(-1)),
            avg(f[...,33:43].double().amax(-1))),-1)
        g=.5*(features@self.theta).sigmoid()
        return e*g.unsqueeze(-1)

def kl_exact(p,e):
    # Exact D_KL(parent || tilted), including all unlisted words.
    # Their energy is zero, so they need not be enumerated.
    e=e.double();z=1.+(p*torch.expm1(e[:,None,:])).sum(-1)
    return z.log()-(p*e[:,None,:]).sum(-1)

def objective94(model,data):
    p,f,k,c,mask,target,q,wp,w=data
    energy=model(f,k,c,mask)
    rr=ratios(p,energy,target)
    ratio=torch.cat((torch.ones_like(rr[...,:1]),rr),-1)
    ev=(q*ratio[:,0]*w).sum(-1).clamp_min(1e-300)
    probs=(wp*ratio*w[:,None,:]).sum(-1).clamp_min(1e-300)
    base=(wp*w[:,None,:]).sum(-1).clamp_min(1e-300)
    # Penalize excess likelihood loss relative to the inherited law.
    regress=F.relu(base[:,0].log()-probs[:,0].log()).mean()
    return -ev.log().mean(),-probs[:,0].log().mean(),F.softplus(probs[:,1:].log()-probs[:,:1].log()).mean(),kl_exact(p,energy).mean(),regress

def selftest():
    torch.manual_seed(94)
    p=torch.tensor([[[.12,.2,.05],[.24,.08,.1]]],dtype=torch.float64)
    e=torch.tensor([[.2,-.5,.3]],dtype=torch.float64)
    pp=torch.cat((p,1-p.sum(-1,keepdim=True)),-1)
    ee=torch.cat((e,torch.zeros(1,1,dtype=torch.float64)),-1)
    tilted=pp*ee[:,None].exp();tilted/=tilted.sum(-1,keepdim=True)
    assert torch.allclose(kl_exact(p,e),(pp*(pp.log()-tilted.log())).sum(-1),atol=1e-15)
    model=Trust94(Field88())
    with torch.no_grad():model.field.read[-1].bias.fill_(.2)
    f=torch.randn(1,3,44);k=torch.randn(1,3,128);c=torch.randn(1,4,128);mask=torch.ones(1,3,dtype=torch.bool)
    assert torch.allclose(model(f,k,c,mask),model.field(f,k,c,mask).double()*.125,atol=1e-15)
    t=torch.tensor([[0,1,2]]);q=torch.tensor([[.1,.2,.1]],dtype=torch.float64)
    wp=torch.cat((torch.full((1,3,1),.1,dtype=torch.float64),p.transpose(1,2)),2)
    w=torch.tensor([[.3,.4,.3]],dtype=torch.float64)
    loss=sum(objective94(model,(p,f,k,c,mask,t,q,wp,w)));loss.backward()
    assert model.theta.grad.abs().sum()>0 and model.field.read[-1].weight.grad.abs().sum()>0
    print('v94 exact trust-region KL, initial inherited gain, joint gradients passed',flush=True)
if __name__=='__main__':selftest()
