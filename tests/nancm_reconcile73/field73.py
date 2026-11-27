"""v73 causal prospective reconciliation; exact full-vocabulary normalization."""
import torch
from torch import nn
import torch.nn.functional as F

class Reconciliation(nn.Module):
    def __init__(self,d=64):
        super().__init__()
        self.gate=nn.Sequential(nn.Linear(512+3*d,128),nn.SiLU(),nn.Linear(128,d))
        self.residual=nn.Sequential(nn.Linear(512+3*d,128),nn.SiLU(),nn.Linear(128,d))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias,-3.)
        self.strength=nn.Parameter(torch.tensor(0.))
    def project(self,net,h,previous=None,scale=1.):
        z=net.state(h);now=net.out(z)
        if scale==0.:return now,z
        if previous is None:
            previous=F.pad(h[:,:-1],(1,0))
        zp=net.state(previous);anticipated=net.future(zp)
        actual=net.lex(h[:,-1])
        x=torch.cat((z,zp,actual,now,anticipated),-1)
        # A coordinate-wise correction to the prediction, informed by the
        # preceding forecast and the intervening observed word.
        gate=self.gate(x).sigmoid()
        delta=gate*(anticipated-now)+self.residual(x)
        valid=(h[:,-1]!=0).to(now.dtype).unsqueeze(-1)
        return now+scale*self.strength.tanh()*delta*valid,z
    def logits(self,net,h,previous=None,scale=1.):
        if scale==0.:return net.logits(h)
        q,_=self.project(net,h,previous,scale)
        keys=net.lex.weight+net.semantic(net.code)
        return q@keys.T+net.prior+net.bias

def shifted(h,w):
    return torch.cat((h[:,1:],w[:,None]),-1)

def selftest():
    import numpy as np
    from field72 import Generator,batch
    torch.manual_seed(73);rng=np.random.default_rng(73)
    x=rng.normal(size=(12,128)).astype(np.float32);x[0]=0.
    net=Generator(x,np.ones(12)/12);b=Reconciliation()
    h,y=batch([np.tile([1,2,3,4,5,6],40),np.tile([7,8,9,10,11],40)],rng,24)
    assert torch.equal(b.logits(net,h,scale=0.),net.logits(h))
    assert torch.allclose(b.logits(net,h).softmax(-1).sum(-1),torch.ones(len(h)),atol=1e-6)
    # The same future position can be scored prospectively and after actual
    # intervening observation; the two conditional laws need not be equal.
    nxt=shifted(h,y[:,0]);assert torch.equal(nxt[:,-1],y[:,0])
    loss=F.cross_entropy(b.logits(net,nxt,h),y[:,1]);loss.backward()
    assert b.strength.grad is not None and torch.isfinite(b.strength.grad)
    assert b.strength.grad.abs()>0
    with torch.no_grad():b.strength.fill_(.5)
    net.zero_grad();b.zero_grad();F.cross_entropy(b.logits(net,nxt,h),y[:,1]).backward()
    assert b.gate[-1].weight.grad.abs().sum()>0 and b.residual[-1].weight.grad.abs().sum()>0
    assert net.future.weight.grad.abs().sum()>0 and net.out.weight.grad.abs().sum()>0
    print('v73 rollback, normalization, causal shift and bidirectional gradient preflight passed')
if __name__=='__main__':
    torch.set_num_threads(2);selftest()
