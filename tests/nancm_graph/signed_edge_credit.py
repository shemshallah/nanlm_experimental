"""Learned signed edge competition under the exact finite-candidate likelihood."""
import torch
from torch import nn

class EdgeCredit(nn.Module):
    def __init__(self,dim=25,rails=8,points=128):
        super().__init__()
        self.linear=nn.Parameter(torch.zeros(dim))
        self.support=nn.Parameter(torch.zeros(rails,points))
        self.opposition=nn.Parameter(torch.zeros(rails,points))
        self.routing=nn.Parameter(torch.zeros(rails,2))
        self.interaction=nn.Parameter(torch.zeros(rails,rails))
    def forward(self,features,residual,values,counts,geometry=True):
        linear=features@self.linear
        if not geometry:return 4*torch.tanh(linear/4)
        product=residual[:,None,:,:]*values[:,:,None,:]
        a=(product.relu()*self.support.sigmoid()).sum(-1)
        b=((-product).relu()*self.opposition.sigmoid()).sum(-1)
        strength=counts/(counts+4)
        route=(self.routing[:,0]+self.routing[:,1]*strength).softmax(-1)*strength
        signed=a-b
        pair=torch.einsum('bkr,rs,bks->bk',signed,self.interaction.tanh(),signed)/64
        return 4*torch.tanh((linear+8*(signed*route[:,None]).sum(-1)+pair)/4)

def likelihood(lift,prior,hit,target,scale=1.):
    lift=lift.double()*scale;prior=prior.double();hit=hit.double();target=target.double()
    return (torch.log1p((prior*lift.expm1()).sum(-1))-(hit*lift).sum(-1)-target.log()).mean()

def preflight():
    torch.set_num_threads(2);torch.manual_seed(43)
    # A held-out synthetic task: only the sign at a designated point and rail
    # identifies the target. Shared lexical features are deliberately uninformative.
    def sample(n):
        r=torch.randn(n,8,16)*.12;v=torch.randn(n,6,16)*.12
        r[:,2,3]=torch.randn(n);v[:,:,3]=torch.randn(n,6)
        label=(r[:,None,2,3]*v[:,:,3]).argmax(1)
        f=torch.zeros(n,6,25);c=torch.full((n,8),16.);p=torch.full((n,6),.1)
        h=torch.nn.functional.one_hot(label,6).float();t=torch.full((n,),.1)
        return f,r,v,c,p,h,t
    train=sample(512);valid=sample(256);model=EdgeCredit(points=16).double()
    ds=tuple(a.double() for a in train);z=tuple(a[:12] for a in ds)
    with torch.no_grad():
        for p in model.parameters():p.normal_(0,.05)
    def loss():return likelihood(model(*z[:4]),*z[4:])
    loss().backward();errs={};norms={}
    for name,p in model.named_parameters():
        idx=tuple(torch.unravel_index(p.grad.abs().argmax(),p.shape));g=float(p.grad[idx]);norms[name]=float(p.grad.norm())
        with torch.no_grad():p[idx]+=1e-5;hi=float(loss());p[idx]-=2e-5;lo=float(loss());p[idx]+=1e-5
        errs[name]=abs((hi-lo)/2e-5-g)
    assert max(errs.values())<1e-8
    assert all(v>0 for k,v in norms.items() if k!='linear')
    model=EdgeCredit(points=16);opt=torch.optim.Adam(model.parameters(),lr=.04)
    def evaluate(data,geometry=True,wrong=False):
        f,r,v,c,p,h,t=data
        if wrong:r=r.roll(1,0)
        with torch.no_grad():l=model(f,r,v,c,geometry);return float(likelihood(l,p,h,t)),float((l.argmax(1)==h.argmax(1)).float().mean())
    before=evaluate(valid)
    for i in range(256):
        ix=torch.randint(len(train[0]),(32,));a=tuple(t[ix] for t in train)
        opt.zero_grad();loss=likelihood(model(*a[:4]),*a[4:]);loss.backward();opt.step()
    after=evaluate(valid);wrong=evaluate(valid,wrong=True);off=evaluate(valid,geometry=False)
    assert after[0]<before[0] and after[0]<wrong[0] and after[1]>off[1]
    assert likelihood(torch.zeros_like(valid[4]),*valid[4:]) == -valid[-1].double().log().mean()
    print({'gradient_max_error':max(errs.values()),'gradient_norms':norms,'before':before,'after':after,'wrong_history':wrong,'geometry_off':off,'zero_lift_exact':True})
if __name__=='__main__':preflight()
