"""Differentiable admission/retention over bounded causal age/repeat moments."""
import torch
from torch import nn
import numpy as np

AGES=(16,64,256,1024)
class Retention(nn.Module):
    def __init__(self, rails=8):
        super().__init__()
        self.age=nn.Parameter(torch.zeros(rails,4))
        self.admission=nn.Parameter(torch.zeros(rails,2))
    def weights(self,adaptive=True):
        a=2*self.age.tanh();b=2*self.admission.tanh()
        return (a[:,:,None]+b[:,None,:]).exp() if adaptive else torch.ones_like(a[:,:,None]+b[:,None,:])
    def forward(self,moments,counts,bg,adaptive=True):
        # B,R,A,repeat,P sums, B,R,A,repeat counts, B,P background.
        w=self.weights(adaptive)
        n=(counts*w).sum((2,3));s=(moments*w[...,None]).sum((2,3))
        residual=(s/n.clamp_min(1e-12)[...,None]-bg[:,None,:])*(n>0)[...,None]
        return residual,n

def loss(lift,prior,hit,target):
    lift=lift.double();prior=prior.double();hit=hit.double();target=target.double()
    return (torch.log1p((prior*lift.expm1()).sum(-1))-(hit*lift).sum(-1)-target.log()).mean()

def preflight():
    torch.set_num_threads(2);torch.manual_seed(4401)
    def sample(n):
        y=torch.randint(4,(n,));good=torch.nn.functional.one_hot(y,4).float()
        bad=torch.nn.functional.one_hot((y+torch.randint(1,4,(n,)))%4,4).float()
        m=torch.zeros(n,1,4,2,4);c=torch.zeros(n,1,4,2)
        # Three fresh, first-seen distractors; one older recurring association.
        m[:,0,0,0]=3*bad;c[:,0,0,0]=3
        m[:,0,2,1]=good;c[:,0,2,1]=1
        bg=torch.zeros(n,4);p=torch.full((n,4),.2);t=torch.full((n,),.2)
        return m,c,bg,p,good,t
    tr=sample(512);va=sample(256);model=Retention(1)
    def pred(data,adaptive=True):
        m,c,bg,p,hit,t=data;r,_=model(m,c,bg,adaptive);return 4*r[:,0],p,hit,t
    def evaluate(data,adaptive=True):
        with torch.no_grad():
            l,p,h,t=pred(data,adaptive);return float(loss(l,p,h,t)),float((l.argmax(1)==h.argmax(1)).float().mean())
    before=evaluate(va);assert before==evaluate(va,False)
    opt=torch.optim.Adam(model.parameters(),lr=.05)
    for _ in range(128):
        i=torch.randint(512,(32,));batch=tuple(x[i] for x in tr)
        opt.zero_grad();loss(*pred(batch)).backward();opt.step()
    after=evaluate(va);fixed=evaluate(va,False)
    wrong=(va[0].roll(1,0),*va[1:]);wrong_result=evaluate(wrong)
    assert after[0]<fixed[0] and after[1]>fixed[1] and after[1]>wrong_result[1]
    empty=torch.zeros_like(va[0]);c=torch.zeros_like(va[1]);r,n=model(empty,c,va[2]);assert torch.count_nonzero(r)==torch.count_nonzero(n)==0
    print({'fixed':fixed,'before':before,'adaptive':after,'wrong_history':wrong_result,'weights':model.weights().detach().tolist(),'empty_exact':True,'parameters':sum(p.numel() for p in model.parameters())})
if __name__=='__main__':preflight()
