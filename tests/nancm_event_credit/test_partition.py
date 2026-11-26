import itertools,math,torch,numpy as np
from partition_credit import *
import folded_event as FE
from event_credit import factors
class Word:
    d={'a':.79,'an':.1,'ant':1e-7,'ante':.0099999,'z':.1}
    def complete(self,w):return self.d.get(w,0.)
    def mass(self,p):return math.fsum(v for w,v in self.d.items() if w.startswith(p))
def raw(s):return np.full(256,1/256)
base=Word();words=['a','ant','z'];p=torch.tensor([[base.complete(w) for w in words]],dtype=torch.float64)
valid=torch.ones_like(p,dtype=torch.bool);l=torch.tensor([[.1,.9,-.4]],dtype=torch.float64,requires_grad=True)
model=Budget().double();shift,innov=model(p,valid,l);q,c=allocate(p,valid,l,shift,innov)
assert torch.equal(allocate(p,valid,l,shift,innov,0)[0],p)
assert torch.equal(allocate(p,valid,l,shift,innov,0)[1],torch.ones_like(c))
assert abs(float((q.sum(-1)+c*(1-p.sum(-1))).detach())-1)<1e-14
law=AllocationLaw(base,words,q[0].detach().numpy(),c[0].detach());surface=FE.Surface([b'Ant an! a z.'])
donor=FE.Event(base,raw,b'abc ',surface,0.,cap=3);e=FE.Event(law,raw,b'abc ',surface,.6,cap=3)
err=0.;fold_err=0.;marg=0.
for s in [b'A ',b'An!',b'AnT',b'zz?',b'!',b'\xff']:
    a,b,d,r,f,n=factors(s,words,base,donor)
    args=[torch.tensor(np.asarray(v)[None],dtype=torch.float64) for v in (a,b,d,r,f)]
    lit,fold=event_probs(q,c,p,*args,.6)
    err=max(err,abs(float(lit.detach())-e.mass(s)))
    w=s if FE.IS_LETTER[s[-1]] else s[:-1]
    expected=e.prefix_mass(w.lower()) if FE.IS_LETTER[s[-1]] else e.complete(w.lower()) if w else sum(e.mass(bytes([x])) for x in FE.STOP)
    fold_err=max(fold_err,abs(float(fold.detach())-expected))
for s in [b'',b'A',b'an']:
    t=e.tokens(s);m=np.zeros(256)
    for k,v in t.items():m[k[0]]+=v
    marg=max(marg,float(np.abs(m-e.next(s)).max()))
assert max(err,fold_err,marg)<1e-12
def obj():
    a,b=model(p,valid,l);q,c=allocate(p,valid,l,a,b)
    return -q[0,1].log()-.3*(c[0]*base.complete('an')).log()
obj().backward();fd=0.
for param in (l,model.gain):
    for ix in np.ndindex(tuple(param.shape)):
        g=float(param.grad[ix]);v=float(param[ix].detach());eps=1e-5
        with torch.no_grad():param[ix]=v+eps;hi=float(obj());param[ix]=v-eps;lo=float(obj());param[ix]=v
        fd=max(fd,abs((hi-lo)/(2*eps)-g))
assert fd<1e-7
torch.set_num_threads(2);torch.manual_seed(4701)
def sample(n):
    y=torch.randint(4,(n,));p=torch.tensor([.1,1e-7,.02],dtype=torch.float64).repeat(n,1)
    hit=torch.nn.functional.one_hot(y,4)[:,:3].double();l=8*hit-4
    l[y==3]=-4.;base=torch.where(y==3,1-p.sum(-1),(p*hit).sum(-1));v=torch.ones_like(p,dtype=torch.bool)
    return p,v,l,hit,base
tr,va=sample(512),sample(256);budget=Budget();opt=torch.optim.Adam(budget.parameters(),lr=.05)
def metrics(ds,adaptive):
    p,v,l,h,b=ds
    if adaptive:
        a,g=budget(p,v,l);q,c=allocate(p,v,l,a,g)
    else:
        d=p*torch.exp(l);den=1+(d-p).sum(-1);q=d/den[:,None];c=1/den
    probs=torch.cat((q,(c*(1-p.sum(-1)))[:,None]),-1)
    y=torch.where(h.any(-1),h.argmax(-1),3);loss=-probs[torch.arange(len(y)),y].log().mean()
    return loss,float((probs.argmax(-1)==y).double().mean()),float(q[y==1,1].mean().detach())
with torch.no_grad():before=metrics(va,True);bounded=metrics(va,False)
for _ in range(192):
    ix=torch.randint(len(tr[0]),(32,));ds=tuple(a[ix] for a in tr);opt.zero_grad();loss,*_=metrics(ds,True);loss.backward();opt.step()
with torch.no_grad():after=metrics(va,True)
assert after[0]<bounded[0] and after[0]<before[0]
print(dict(literal_error=err,folded_error=fold_err,token_error=marg,gradient_error=fd,zero_exact=True,baseline_bounded=(float(bounded[0]),*bounded[1:]),initial=(float(before[0]),*before[1:]),trained=(float(after[0]),*after[1:])))
