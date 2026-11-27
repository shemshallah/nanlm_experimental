"""Context-conditioned Markov expert state; exact event likelihood, no target in transition."""
import numpy as np
import torch
from torch import nn

D=7
class Transition(nn.Module):
    def __init__(self,prior,hazard):
        super().__init__();pi=torch.as_tensor(prior,dtype=torch.float64)
        assert pi.shape==(3,) and pi.min()>0 and abs(float(pi.sum())-1)<1e-12 and 0<hazard<=1
        self.register_buffer('prior',pi)
        self.register_buffer('base',((1-hazard)*torch.eye(3,dtype=torch.float64)+hazard*pi[None,:]).log())
        self.theta=nn.Parameter(torch.zeros(3,3,D,dtype=torch.float64))
    def matrix(self,f,scale=1.):
        return (self.base+3*scale*torch.einsum('...d,ijd->...ij',f,self.theta).tanh()).softmax(-1)
    def forward(self,lik,features,mask,scale=1.):
        assert lik.ndim==3 and lik.shape[-1]==3 and features.shape==(*lik.shape[:2],D) and mask.shape==lik.shape[:2]
        state=self.prior.expand(lik.shape[0],3);total=lik.new_zeros(())
        for t in range(lik.shape[1]):
            w=state if t==0 else torch.einsum('bi,bij->bj',state,self.matrix(features[:,t],scale))
            x=w*lik[:,t];z=x.sum(-1);assert bool((z>0).all())
            total=total+torch.where(mask[:,t],-z.log(),0.).sum()
            state=torch.where(mask[:,t,None],x/z[:,None],state)
        return total
    def runtime(self,scale=1.):return Runtime(self.prior.detach().numpy(),self.base.detach().numpy(),self.theta.detach().numpy(),scale)

class Runtime:
    def __init__(self,prior,base,theta,scale):
        self.prior=prior.copy();self.base=base.copy();self.theta=theta.copy();self.scale=scale;self.reset()
    def reset(self):self.state=self.prior.copy();self.started=False
    def weights(self,f):
        if not self.started:return self.state.copy()
        z=self.base+3*self.scale*np.tanh(np.einsum('d,ijd->ij',f,self.theta));p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
        w=self.state@p;assert w.min()>0 and abs(w.sum()-1)<1e-12;return w
    def observe(self,f,lik):
        x=self.weights(f)*lik;z=float(x.sum());assert z>0;self.state=x/z;self.started=True;return z
