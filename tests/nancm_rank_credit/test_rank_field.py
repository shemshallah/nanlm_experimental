import torch,numpy as np,math
from rank_field import Transport,credit
from transition import Transition
torch.set_num_threads(2);torch.manual_seed(65)
V,D,T=11,5,6;code=torch.randn(V,D);code[0]=0
m=Transport(code,rank=4).double();x=torch.randn(T,9,D,dtype=torch.float64);p=torch.rand(T,2,V,dtype=torch.float64);p/=p.sum(-1,keepdim=True)
assert m(x,p,0) is p and torch.allclose(m(x,p),p,atol=1e-15,rtol=0)
core=Transition([.5,.2,.3],.35);core.theta.data.normal_(0,.1);core.requires_grad_(False)
m.read.data.normal_(0,.05);e=torch.zeros(T,V,dtype=torch.float64);e[1:,1]=1;s=torch.ones(T,dtype=torch.float64)*.2;r=torch.ones(T,dtype=torch.float64)*.01;f=torch.randn(T,5,dtype=torch.float64)
ids=[torch.tensor([],dtype=torch.long)]+[torch.tensor([1,2,3,4]) for _ in range(T-1)];wr=[torch.ones(len(i),dtype=torch.float64)*.001 for i in ids];wc=[torch.ones(len(i),dtype=torch.float64) for i in ids]
def loss():return credit(core,m(x,p),e,s,r,f,ids,wr,wc,rank_weight=.5)[0]
loss().backward();fd=0.;ng=0
for a in m.parameters():
 g=a.grad;assert g.norm()>0
 ix=tuple(torch.unravel_index(g.abs().argmax(),g.shape));v=a[ix].item();eps=1e-5
 with torch.no_grad():a[ix]=v+eps;hi=loss().item();a[ix]=v-eps;lo=loss().item();a[ix]=v
 fd=max(fd,abs((hi-lo)/(2*eps)-g[ix].item()));ng+=1
assert fd<1e-8
q=m(x,p);err=float((q.sum(-1)-1).abs().max().detach());assert err<1e-12
a=credit(core,q,e,s,r,f,ids,wr,wc,rank_weight=.5);b=credit(core,q[:2],e[:2],s[:2],r[:2],f[:2],ids[:2],wr[:2],wc[:2],rank_weight=.5);c=credit(core,q[2:],e[2:],s[2:],r[2:],f[2:],ids[2:],wr[2:],wc[2:],b[1],b[2],rank_weight=.5)
chunk=abs(float((a[3]-b[3]-c[3]).detach()));assert chunk<1e-12
print(dict(zero_identity=True,normalization_error=err,gradient_groups=ng,gradient_error=fd,chunk_event_error=chunk))
