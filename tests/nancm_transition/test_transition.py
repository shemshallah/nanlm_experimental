import numpy as np,torch,itertools,math
from transition import Transition,D

def main():
    torch.set_num_threads(2);torch.manual_seed(61);rng=np.random.default_rng(61)
    m=Transition([.3,.3,.4],.2);m.theta.data.normal_(0,.1)
    lik=torch.tensor(rng.uniform(.1,.9,(1,4,3)));f=torch.tensor(rng.normal(size=(1,4,D)));mask=torch.ones(1,4,dtype=torch.bool)
    p=float((-m(lik,f,mask)).exp().detach());mat=m.matrix(f[0],1.).detach().numpy();oracle=0.
    for path in itertools.product(range(3),repeat=4):
        z=float(m.prior[path[0]]*lik[0,0,path[0]])
        for t in range(1,4):z*=mat[t,path[t-1],path[t]]*float(lik[0,t,path[t]])
        oracle+=z
    run=m.runtime();q=math.prod(run.observe(a,b) for a,b in zip(f[0].numpy(),lik[0].numpy()))
    assert max(abs(p-oracle),abs(p-q))<1e-14
    loss=lambda:m(lik,f,mask);loss().backward();g=m.theta.grad.clone();fd=0.
    for ix in np.ndindex(tuple(g.shape)):
        with torch.no_grad():m.theta[ix]+=1e-5;hi=float(loss());m.theta[ix]-=2e-5;lo=float(loss());m.theta[ix]+=1e-5
        fd=max(fd,abs((hi-lo)/2e-5-float(g[ix])))
    assert fd<1e-8 and torch.count_nonzero(g)==g.numel()
    z=m.runtime(0.);pi=m.prior.numpy();s=pi.copy();err=0.
    for a,b in zip(f[0].numpy(),lik[0].numpy()):
        w=.8*s+.2*pi;err=max(err,float(abs(w-z.weights(a)).max()));s=w*b;s/=s.sum();z.observe(a,b)
    assert err<1e-14
    def sample(n):
        signal=rng.choice([-1.,1.],(n,12));x=np.zeros((n,12,D));x[:,:,0]=1;x[:,:,1]=signal
        l=np.full((n,12,3),.1);l[:,:,0]=.3;l[:,:,1]=np.where(signal>0,.9,.1);l[:,:,2]=np.where(signal<0,.9,.1)
        return torch.tensor(l),torch.tensor(x),torch.ones(n,12,dtype=torch.bool)
    tr=sample(48);va=sample(32);m=Transition([.3,.35,.35],.5);opt=torch.optim.Adam(m.parameters(),lr=.04)
    def score(ds):return float((m(*ds)/ds[2].sum()/math.log(2)).detach())
    before=score(va)
    for _ in range(96):opt.zero_grad();(m(*tr)/tr[2].sum()).backward();opt.step()
    after=score(va);wrong=(va[0],va[1]*torch.tensor([1.,-1.,1.,1.,1.,1.,1.]),va[2]);bad=score(wrong)
    assert after<before and after<bad
    print(dict(paths=81,enumeration_error=abs(p-oracle),runtime_error=abs(p-q),gradient_error=fd,gradient_coordinates=g.numel(),fixed_switch_error=err,constructed_before=before,constructed_after=after,constructed_wrong=bad))
if __name__=='__main__':main()
