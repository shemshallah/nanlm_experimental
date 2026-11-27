import itertools,math,torch,numpy as np
from joint import Joint
from transition import Transition
from credit import laws
torch.set_num_threads(2);torch.manual_seed(63)
D,V,T=5,11,5
m=Joint([.5,.2,.3],.35,D);m.core.theta.data.normal_(0,.1)
base=torch.randn(T,5,dtype=torch.float64);hidden=torch.randn(T,D,dtype=torch.float64,requires_grad=True)
logits=torch.randn(T,V,dtype=torch.float64,requires_grad=True)
anchor=torch.rand(T,V,dtype=torch.float64);anchor*=.75/anchor.sum(-1,keepdim=True)
share=torch.full((T,),.25,dtype=torch.float64);copy=torch.rand(T,V,dtype=torch.float64);copy/=copy.sum(-1,keepdim=True)
wt=torch.rand(T,V,dtype=torch.float64).exp();ct=torch.rand(T,V,dtype=torch.float64).exp();credit=torch.rand(T,4,dtype=torch.float64);credit/=credit.sum(-1,keepdim=True)
e=torch.zeros(T,V,dtype=torch.float64);e[1:,2]=1;s=torch.full((T,),.2,dtype=torch.float64);raw=torch.full((T,),.03,dtype=torch.float64);target=torch.tensor([-1,2,2,2,2])
def get():
 a,b=laws(logits.log_softmax(-1),anchor,share,copy,wt,ct,credit)
 return a,b
a,b=get();lik=torch.stack((raw,(a*e).sum(-1)*s,(b*e).sum(-1)*s),-1);f=torch.cat((base,torch.zeros(T,2,dtype=torch.float64)),1)
v=lik[:-1].clamp_min(1e-300).log();f[1:,5:]=((v[:,1:]-v[:,:1])/8).clamp(-1,1)
res=m.trajectory(a,b,raw,e,s,base,hidden,target);ref=m.core(lik[None],f[None],torch.ones(1,T,dtype=torch.bool));err=float(abs(res[3]-ref).detach());assert err<1e-12
m.signed.data.normal_(0,.05)
def loss():return m.trajectory(*get(),raw,e,s,base,hidden,target)[0]
loss().backward();fd=0.;groups={}
for name,p in [('transition',m.core.theta),('signed',m.signed),('logits',logits),('hidden',hidden)]:
 g=p.grad.clone();groups[name]=float(g.norm());assert groups[name]>0
 for ix in np.ndindex(tuple(g.shape)):
  with torch.no_grad():p[ix]+=1e-5;hi=loss().item();p[ix]-=2e-5;lo=loss().item();p[ix]+=1e-5
  fd=max(fd,abs((hi-lo)/2e-5-float(g[ix])))
assert fd<1e-8
a,b=get();full=m.trajectory(a,b,raw,e,s,base,hidden,target);first=m.trajectory(a[:2],b[:2],raw[:2],e[:2],s[:2],base[:2],hidden[:2],target[:2]);rest=m.trajectory(a[2:],b[2:],raw[2:],e[2:],s[2:],base[2:],hidden[2:],target[2:],first[1],first[2]);chunk=float(abs(full[3]-first[3]-rest[3]).detach());assert chunk<1e-12
print(dict(parent_parity_error=err,finite_difference_max=fd,checked=sum(p.numel() for p in (m.core.theta,m.signed,logits,hidden)),gradient_norms=groups,chunk_error=chunk))
