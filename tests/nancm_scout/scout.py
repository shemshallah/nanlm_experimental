"""Eight-rail signed vocabulary scout. Candidate proposals only."""
import torch
from torch import nn
import numpy as np

class Scout(nn.Module):
    def __init__(self,code):
        super().__init__();self.register_buffer('code',torch.as_tensor(code,dtype=torch.float32))
        d=self.code.shape[1];self.plus=nn.Parameter(torch.empty(8,d,d));self.minus=nn.Parameter(torch.empty(8,d,d));self.route=nn.Parameter(torch.zeros(8));self.temperature=nn.Parameter(torch.tensor(0.))
        nn.init.normal_(self.plus,std=.03);nn.init.normal_(self.minus,std=.03)
    def forward(self,x):
        a=torch.einsum('bri,rij->brj',x.relu(),self.plus)
        b=torch.einsum('bri,rij->brj',(-x).relu(),self.minus)
        z=((a-b).tanh()*self.route.softmax(0)[None,:,None]).sum(1)
        return (1+7*self.temperature.sigmoid())*(z@self.code.T)

def rails(code,history):
    out=np.zeros((8,code.shape[1]),np.float32)
    for i in range(4):
        if len(history)>i:out[i]=code[history[-1-i]]
    for i,(a,b) in enumerate(((0,8),(8,32),(32,128),(128,512)),4):
        h=history[max(0,len(history)-b):len(history)-a] if len(history)>a else []
        if len(h):out[i]=code[h].mean(0)
    return out

def preflight():
    torch.set_num_threads(2);torch.manual_seed(4801)
    rng=np.random.default_rng(4801);code=np.vstack([np.zeros((1,16)),np.eye(16)]).astype(np.float32)
    def sample(n):
        y=rng.integers(1,17,size=n);x=rng.normal(0,.1,(n,8,16)).astype(np.float32)
        x[:,2]=code[y];x[:,5]=-code[y];return torch.tensor(x),torch.tensor(y)
    tr,y=sample(512);va,vy=sample(256);model=Scout(code);opt=torch.optim.Adam(model.parameters(),lr=.03)
    def metrics(x,y):
        with torch.no_grad():z=model(x);return float(torch.nn.functional.cross_entropy(z,y)),float((z.argmax(1)==y).float().mean())
    before=metrics(va,vy)
    for i in range(192):
        ix=torch.randint(len(y),(32,));opt.zero_grad();loss=torch.nn.functional.cross_entropy(model(tr[ix]),y[ix]);loss.backward()
        if i==0:assert all(p.grad is not None and p.grad.norm()>0 for p in model.parameters())
        opt.step()
    after=metrics(va,vy);wrong=metrics(va.roll(1,0),vy)
    assert after[0]<before[0] and after[1]>wrong[1]
    print({'before':before,'after':after,'wrong':wrong,'parameters':sum(p.numel() for p in model.parameters())})
if __name__=='__main__':preflight()
