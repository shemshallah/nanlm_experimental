import math, numpy as np, torch
import folded_event as FE
from event_credit import factors
from dense_scout import DenseScout, fuse, interval,event_prob,event_loss
class Word:
    d={'a':.3,'an':.2,'ant':.2,'ante':.1,'z':.2}
    def complete(self,w):return self.d.get(w,0.)
    def mass(self,p):return math.fsum(v for w,v in self.d.items() if w.startswith(p))

def main():
    torch.set_num_threads(2);torch.manual_seed(4901)
    words=['a','an','ant','ante'];base=Word();surface=FE.Surface([b'An ANT! a an. z\n']);raw=lambda s:np.full(256,1/256)
    donor=FE.Event(base,raw,b'root ',surface,0.,cap=3)
    events=[b'A ',b'An!',b'AnT',b'z?',b'xy?',b'!',b'\xff']
    rows=[]
    for event in events:
        _,b,s,r,f,n=factors(event,[],base,donor)
        capped=FE.IS_LETTER[event[-1]];p=event if capped else event[:-1]
        lo,hi=interval(words,FE.folded(p),capped) if p else (0,0)
        rows.append((lo,hi,b,s,r,f,n))
    args=tuple(torch.as_tensor(np.asarray(x),dtype=torch.int64 if i<2 else torch.float64) for i,x in enumerate(zip(*rows)))
    logits=(torch.randn(len(events),len(words),dtype=torch.float64)*.4).requires_grad_();err=fold_err=marg=0.
    for weight in (0.,.25,.75):
        for strength in (0.,.25,.9):
            lit,fold=event_prob(logits,*args[:-1],weight,strength)
            for i,event in enumerate(events):
                law=fuse(words,logits[i].softmax(0).detach().numpy(),base,strength)
                if strength==0:assert law is base
                e=FE.Event(law,raw,b'root ',surface,weight,cap=3)
                err=max(err,abs(float(lit[i].detach())-e.mass(event)))
                p=event if FE.IS_LETTER[event[-1]] else event[:-1]
                expected=e.prefix_mass(p.lower()) if FE.IS_LETTER[event[-1]] else e.complete(p.lower()) if p else sum(e.mass(bytes([d])) for d in FE.STOP)
                fold_err=max(fold_err,abs(float(fold[i].detach())-expected))
                q=e.next(b'');t=e.tokens(b'');m=np.zeros(256)
                for k,v in t.items():m[k[0]]+=v
                marg=max(marg,float(np.abs(m-q).max()))
    assert max(err,fold_err,marg)<1e-12
    loss=lambda:event_loss(logits,*args,.6,.5)
    loss().backward();g=logits.grad.clone();fd=0.
    for ix in np.ndindex(tuple(logits.shape)):
        with torch.no_grad():logits[ix]+=1e-5;hi=float(loss());logits[ix]-=2e-5;lo=float(loss());logits[ix]+=1e-5
        fd=max(fd,abs((hi-lo)/2e-5-float(g[ix])))
    assert fd<1e-8 and torch.count_nonzero(g[-2:])==0
    print(dict(literal_max_error=err,folded_max_error=fold_err,token_marginal_max_error=marg,gradient_max_error=fd,gradient_coordinates=g.numel(),zero_strength_exact=True,empty_gradients_exact=True))
    dim=16;code=torch.eye(dim)*4;model=DenseScout(code,torch.full((dim,),-math.log(dim)),hidden_dim=dim)
    def sample(n):
        y=torch.randint(dim,(n,));target=torch.nn.functional.one_hot(y,dim).float()
        rr=torch.randn(n,8,dim)*.03;rr[:,0]=target*.1
        m=torch.randn(n,8,4,2,dim)*.01;c=torch.ones(n,8,4,2);m[:,5,2,1]+=target*3
        h=torch.randn(n,dim)*.03;h+=target*.2;bg=torch.zeros(n,dim)
        return rr,m,c,bg,h,y,y+1,torch.full((n,),1/dim),torch.full((n,),.2),torch.full((n,),.001),torch.full((n,),.05),torch.full((n,),4.)
    tr,va=sample(384),sample(192);opt=torch.optim.Adam(model.parameters(),lr=.025)
    def evaluate(data):
        with torch.no_grad():
            l=model(*data[:5]);return float(event_loss(l,*data[5:],.6,.75)/math.log(2)),float((l.argmax(-1)==data[5]).float().mean())
    before=evaluate(va);norms={}
    for i in range(160):
        ix=torch.randint(384,(24,));ds=tuple(a[ix] for a in tr);opt.zero_grad();event_loss(model(*ds[:5]),*ds[5:],.6,.75).backward()
        if i==0:norms={k:float(p.grad.norm()) for k,p in model.named_parameters()};assert min(norms.values())>0
        opt.step()
    after=evaluate(va);wrong=evaluate((*[a.roll(1,0) for a in va[:5]],*va[5:]));assert after[0]<before[0] and after[1]>wrong[1]
    print(dict(train=384,validation=192,updates=160,before=before,after=after,wrong_context=wrong,gradient_groups=len(norms),min_gradient_norm=min(norms.values())))
if __name__=='__main__':main()
