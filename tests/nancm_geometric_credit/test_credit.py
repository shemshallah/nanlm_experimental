import torch,numpy as np
from credit import laws,literal,sequence
torch.set_num_threads(2);torch.manual_seed(62);rng=np.random.default_rng(62)
N,V=8,17
z=torch.randn(N,V,dtype=torch.float64,requires_grad=True)
anchor=torch.rand(N,V,dtype=torch.float64);anchor=.75*anchor/anchor.sum(-1,keepdim=True)
share=torch.full((N,),.25,dtype=torch.float64)
copy=torch.rand(N,V,dtype=torch.float64);copy/=copy.sum(-1,keepdim=True)
wt=torch.rand(N,V,dtype=torch.float64).exp();ct=torch.rand(N,V,dtype=torch.float64).exp()
credit=torch.rand(N,4,dtype=torch.float64);credit/=credit.sum(-1,keepdim=True)
e=torch.zeros(N,V,dtype=torch.float64);e[1,0]=.003;e[2:5,4]=1;e[5,2:7]=1;e[6,0]=.002;e[6,6:9]=1;e[7,8]=1
s=torch.rand(N,dtype=torch.float64);r=torch.rand(N,dtype=torch.float64)*.001+.0001;w=(.5,.125,.375)
args=(anchor,share,copy,wt,ct,credit,e,s,r,w)
p,q=laws(z.log_softmax(-1),*args[:6]);err=max(float((p.sum(-1)-1).detach().abs().max()),float((q.sum(-1)-1).detach().abs().max()))
objective=lambda:-literal(z.log_softmax(-1),*args).log().mean()
objective().backward();g=z.grad.clone();fd=0.
for ix in np.ndindex(tuple(z.shape)):
 with torch.no_grad():
  z[ix]+=1e-5;hi=objective().item();z[ix]-=2e-5;lo=objective().item();z[ix]+=1e-5
 fd=max(fd,abs((hi-lo)/2e-5-float(g[ix])))
assert err<1e-12 and fd<1e-8 and torch.count_nonzero(g[0])==0 and torch.count_nonzero(g[1:])==7*V
print(dict(normalization_error=err,gradient_error=fd,gradient_coordinates=g.numel(),empty_event_zero_gradient=True,nonempty_gradient_coordinates=int(torch.count_nonzero(g[1:]))))
# Verify temporal likelihood against a direct recursion, including cross-event credit.
from switching import Switch
pi=torch.tensor(w,dtype=torch.float64)
def temporal():
 a,b=laws(z.log_softmax(-1),*args[:6])
 q=torch.stack((r,(a*e).sum(-1)*s,(b*e).sum(-1)*s),-1)
 return sequence(q,pi,pi,.35),q
loss,q=temporal();m=Switch(w,.35)
assert abs(float(loss.detach())+sum(np.log(m.observe(t)) for t in q.detach().numpy()))<1e-12
z.grad=None;loss.backward();assert torch.count_nonzero(z.grad[0])==0 and torch.count_nonzero(z.grad[1:])==7*V
