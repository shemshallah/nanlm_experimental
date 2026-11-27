"""v74 persistent signed state: bounded recurrent memory, exact disabled path."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from field70 import words

class Persistent(nn.Module):
    def __init__(self,d=64,dim=128):
        super().__init__()
        self.drive=nn.Linear(d+dim,256)
        self.exchange=nn.Linear(256,256,bias=False)
        self.decay=nn.Linear(d+dim,256)
        self.read=nn.Linear(128,d,bias=False)
        self.context=nn.Linear(256,d)
        nn.init.zeros_(self.exchange.weight)
        nn.init.zeros_(self.read.weight)
        nn.init.constant_(self.decay.bias,2.)
    def initial(self,n,device=None):
        return self.drive.weight.new_zeros((n,256),device=device)
    def step(self,net,state,w):
        x=torch.cat((net.lex(w),net.code[w]),-1)
        a,b=(self.drive(x)+self.exchange(state)).chunk(2,-1)
        # Opposed proposals share a fixed occupancy budget per coordinate.
        p=torch.sigmoid(a-b)*torch.sigmoid(a+b)
        n=torch.sigmoid(b-a)*torch.sigmoid(a+b)
        decay=.8+.199*torch.sigmoid(self.decay(x))
        s=decay*state+(1-decay)*torch.cat((p,n),-1)
        return torch.where((w!=0)[:,None],s,state)
    def scan(self,net,seq,state=None):
        if state is None:state=self.initial(len(seq),seq.device)
        for w in seq.unbind(-1):state=self.step(net,state,w)
        return state
    def logits(self,net,bridge,h,state,scale=1.):
        if scale==0.:return bridge.logits(net,h)
        q,z=bridge.project(net,h)
        p,n=state.chunk(2,-1)
        q=q+scale*self.read(p-n)*torch.sigmoid(self.context(z))
        keys=net.lex.weight+net.semantic(net.code)
        return q@keys.T+net.prior+net.bias

class Runtime:
    """Document-local state. No access to an unobserved target."""
    def __init__(self,net,field,ids):
        self.net=net;self.field=field;self.ids=ids;self.reset()
    def reset(self):
        self.seen=();self.state=self.field.initial(1)
    def predict(self,prefix):
        h=words(prefix)
        assert h[:len(self.seen)]==self.seen
        if len(h)>len(self.seen):
            x=torch.tensor([[self.ids.get(w,0) for w in h[len(self.seen):]]],dtype=torch.long)
            self.state=self.field.scan(self.net,x,self.state)
        self.seen=h
        return self.state
    def wrong(self,prefix):
        h=words(prefix)
        # Reverse only the old history, preserve the current 32-word window.
        h=h[:-32][::-1]+h[-32:]
        x=torch.tensor([[self.ids.get(w,0) for w in h]],dtype=torch.long)
        return self.field.scan(self.net,x)

def chunks(seqs,rng,n,warm=96,steps=8):
    lengths=np.array([max(0,len(s)-steps) for s in seqs]);p=lengths/lengths.sum()
    x=np.zeros((n,warm+steps),np.int64)
    for j,i in enumerate(rng.choice(len(seqs),size=n,p=p)):
        s=seqs[i];t=int(rng.integers(1,len(s)-steps+1));h=s[max(0,t-warm):t]
        x[j,warm-len(h):warm]=h;x[j,warm:]=s[t:t+steps]
    return torch.from_numpy(x)

def sequence_loss(net,bridge,field,x,warm=96):
    # Truncated BPTT: initialize state on observed prefix; train next steps.
    with torch.no_grad():state=field.scan(net,x[:,:warm])
    logits=[]
    for j in range(warm,x.shape[1]):
        h=x[:,j-32:j]
        logits.append(field.logits(net,bridge,h,state))
        state=field.step(net,state,x[:,j])
    return F.cross_entropy(torch.stack(logits,1).flatten(0,1),x[:,warm:].flatten())

def selftest():
    from field72 import Generator
    from field73 import Reconciliation
    torch.manual_seed(74)
    code=np.random.default_rng(74).normal(size=(12,128)).astype(np.float32);code[0]=0
    net=Generator(code,np.ones(12)/12);bridge=Reconciliation();field=Persistent()
    x=torch.randint(1,12,(3,104))
    s=field.scan(net,x[:,:96]);h=x[:,64:96]
    assert torch.equal(field.logits(net,bridge,h,s,0.),bridge.logits(net,h))
    assert torch.equal(field.logits(net,bridge,h,s),bridge.logits(net,h))
    assert torch.equal(s,field.scan(net,x[:,48:96],field.scan(net,x[:,:48])))
    assert s.min()>=0 and s.max()<=1
    loss=sequence_loss(net,bridge,field,x);loss.backward()
    assert field.read.weight.grad.abs().sum()>0
    with torch.no_grad():field.read.weight.normal_(std=.1)
    field.zero_grad();net.zero_grad()
    sequence_loss(net,bridge,field,x).backward()
    assert field.exchange.weight.grad.abs().sum()>0
    assert field.drive.weight.grad.abs().sum()>0
    r=Runtime(net,field,{'a':1,'b':2,'c':3})
    with torch.no_grad():
        s=r.predict(b'a b ');t=r.predict(b'a b c ');assert torch.equal(t,r.predict(b'a b c '))
        r.reset();assert torch.equal(t,r.predict(b'a b c '))
    print('v74 persistent signed state, chunk parity, zero rollback, gradient and causal replay passed')
if __name__=='__main__':
    torch.set_num_threads(2);selftest()
