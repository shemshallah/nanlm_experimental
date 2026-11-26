import numpy as np
import torch
import triangle_field as G
from persistent_triangle import TriangleAnchor,blend
from adaptive_retention import Retention
from event_credit import objective

def main():
    torch.set_num_threads(2);torch.manual_seed(4601)
    topo=G.topology();v,pairs,_,_=topo
    anchor=TriangleAnchor(triangles=len(v)).double();ret=Retention().double()
    s=torch.randn(2,8,128,dtype=torch.float64)*.3;z=torch.randn_like(s)*.3
    moments=torch.randn(2,8,4,2,128,dtype=s.dtype)*.3
    counts=torch.rand(2,8,4,2,dtype=s.dtype)*4+.5;bg=torch.randn(2,128,dtype=s.dtype)*.02
    plus=torch.rand(8,len(v),3,dtype=s.dtype)*.3;minus=torch.rand_like(plus)*.3
    code=torch.randn(2,5,16,dtype=s.dtype);prior=torch.full((2,5),.15,dtype=s.dtype)
    hit=torch.zeros_like(prior);hit[:,0]=1;base=prior[:,0];surface=torch.tensor([.1,.3],dtype=s.dtype)
    raw=torch.full((2,),.002,dtype=s.dtype);folded=torch.full((2,),.02,dtype=s.dtype);length=torch.tensor([4.,5.],dtype=s.dtype)
    def solve(scale=1.,empty=False,wrong=False):
        c=counts*0 if empty else counts;m=moments*0 if empty else moments
        if wrong:m=m.roll(1,0);c=c.roll(1,0)
        residual,n=ret(m,c,bg);seed,gate,dp,dm=anchor(s,z,residual,n,pairs,scale)
        x=torch.cat((s,z),-1).tanh();edges=None
        for _ in range(4):
            edges=blend(G.edge_update(x,edges,pairs),seed,gate)
            x=(torch.cat((s,z),-1)+.2*x+G.message(edges,plus+dp,minus+dm,topo)).tanh()
        return x.mean(1)[:,128:],edges,gate
    parent,edges,_=solve(0);empty,empty_edges,_=solve(empty=True)
    assert torch.equal(parent,empty) and torch.equal(edges,empty_edges)
    def loss():
        out,_,_=solve();lift=anchor.lift(out-parent.detach(),code)
        return objective(lift,prior,hit,base,surface,raw,folded,length,.5)
    loss().backward();err={};norms={}
    for prefix,model in [('anchor',anchor),('retention',ret)]:
        for k,p in model.named_parameters():
            assert p.grad is not None
            ix=tuple(torch.unravel_index(p.grad.abs().argmax(),p.shape));g=float(p.grad[ix]);eps=1e-4
            with torch.no_grad():p[ix]+=eps;hi=float(loss());p[ix]-=2*eps;lo=float(loss());p[ix]+=eps
            err[prefix+'.'+k]=abs((hi-lo)/(2*eps)-g);norms[prefix+'.'+k]=float(p.grad.norm())
    assert max(err.values())<1e-8 and min(norms.values())>0
    with torch.no_grad():
        x,e,g=solve();wrong,_,_=solve(wrong=True)
        assert g.min()>=0 and g.max()<=.5 and e.abs().max()<=1
        assert not torch.equal(x,wrong)
        print(dict(triangles=len(v),edges=len(pairs),parameters=sum(p.numel() for p in anchor.parameters()),groups=len(norms),gradient_max_error=max(err.values()),min_gradient_norm=min(norms.values()),empty_exact=True,zero_scale_exact=True,gate_max=float(g.max()),edge_abs_max=float(e.abs().max()),wrong_history_delta=float((x-wrong).norm())))
if __name__=='__main__':main()
