import numpy as np,torch
from crossfit import Crossfit,Exclusion,Gate,loss
class Index:
    def __init__(self,docs,V=9):
        self.seq=[np.array(d,np.int64) for d in docs];self.V=V;self.tables=[]
        for n in (1,2,3):
            rows={}
            for seq in self.seq:
                for i in range(n,len(seq)):
                    k=0
                    for v in seq[i-n:i]:k=k*V+int(v)
                    t=(k,int(seq[i]));rows[t]=rows.get(t,0)+1
            pairs=sorted(rows);self.tables.append(tuple(np.array(a,np.int64) for a in zip(*[(k,w,rows[k,w]) for k,w in pairs])))
if __name__=='__main__':
    rng=np.random.default_rng(51);docs=[rng.integers(1,9,size=80) for _ in range(5)];full=Crossfit(Index(docs));err=feat=scalar=0.;n=0
    for i,seq in enumerate(docs):
        ex=Exclusion(seq,9);rebuild=Crossfit(Index(docs[:i]+docs[i+1:]));
        for j in range(3,len(seq)):
            h=seq[j-3:j];t=seq[j];p,x=full.query(h,exclude=ex);q,y=rebuild.query(h);z,_=full.query(h,t,ex)
            err=max(err,float(abs(p-q).max()));feat=max(feat,float(abs(x-y).max()));scalar=max(scalar,abs(p[t]-z));n+=1
    assert max(err,feat,scalar)<1e-12
    torch.manual_seed(51);torch.set_num_threads(2);m=Gate().double();x=torch.randn(128,12,dtype=torch.float64);p=torch.rand(128,dtype=torch.float64)*.9+.01;q=(torch.rand(128,dtype=torch.float64)*.9+.01).log().requires_grad_()
    obj=lambda:loss(q,p,m(x,.4));obj().backward();fd=0.
    for param in [m.linear.weight,m.linear.bias]:
        for ix in np.ndindex(tuple(param.shape)):
            g=param.grad[ix].item();v=param[ix].item();eps=1e-5
            with torch.no_grad():param[ix]=v+eps;hi=obj().item();param[ix]=v-eps;lo=obj().item();param[ix]=v
            fd=max(fd,abs(g-(hi-lo)/(2*eps)))
    assert fd<1e-8 and q.grad.norm()>0
    assert torch.equal(m(x,.4,0),torch.full((128,),.4,dtype=torch.float64))
    print(dict(queries=n,loo_vs_rebuilt=err,feature_error=feat,scalar_dense_error=scalar,gate_gradient_error=fd,field_gradient_norm=q.grad.norm().item(),zero_gate_exact=True))
