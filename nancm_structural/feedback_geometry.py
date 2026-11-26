"""Bounded signed associative feedback into 8 x (128+128) field points."""
import torch
import torch.nn.functional as F
SHAPES={'fb_support':(6,), 'fb_oppose':(6,), 'fb_support_rails':(8,16), 'fb_oppose_rails':(8,16), 'fb_lower_plus':(16,128), 'fb_lower_minus':(16,128), 'fb_upper_plus':(16,128), 'fb_upper_minus':(16,128), 'fb_relational':(8,256), 'fb_read':(256,128), 'fb_query':(128,16)}
def masked_weights(scores,valid):
    p=scores.masked_fill(~valid,-1e9).softmax(-1)*valid
    return p/p.sum(-1,keepdim=True).clamp_min(1e-12)
def pulse(f,n,valid,values,p):
    a=masked_weights(f@F.softplus(p['fb_support']),valid)
    b=masked_weights(n@F.softplus(p['fb_oppose']),valid)
    u=(a[...,None]*values).sum(1)[:,None,:]*(1+p['fb_support_rails'].tanh())
    v=(b[...,None]*values).sum(1)[:,None,:]*(1+p['fb_oppose_rails'].tanh())
    lo=.35*((u@p['fb_lower_plus']).tanh()-(v@p['fb_lower_minus']).tanh())
    hi=.35*((u@p['fb_upper_plus']).tanh()-(v@p['fb_upper_minus']).tanh())
    return lo,hi
def relation(pulse,raw):return 1+.02*raw.tanh()*pulse.tanh()
def read(points,signal,matrix):return .4*((points*signal).mean(1)@matrix).tanh()
def lift(delta,code,query):return 4*((delta@query)[:,None,:]*code).sum(-1).div(4).tanh()
def nll(lifts,prior,hit,target):
    den=1+(prior*lifts.expm1()).sum(-1)
    return (den.log()-(hit*lifts).sum(-1)-target.log()).mean()
def preflight():
    torch.set_num_threads(2);torch.manual_seed(40)
    p={k:(torch.randn(s,dtype=torch.float64)*.08).requires_grad_() for k,s in SHAPES.items()}
    f=torch.rand(3,12,6,dtype=torch.float64);n=torch.rand_like(f);valid=torch.ones(3,12,dtype=torch.bool);valid[0,-2:]=False
    values=torch.randn(3,12,16,dtype=torch.float64);anchors=torch.rand(3,8,256,dtype=torch.float64)*2-1
    code=torch.randn(3,7,16,dtype=torch.float64);prior=torch.rand(3,7,dtype=torch.float64);prior*=.8/prior.sum(-1,keepdim=True);hit=torch.zeros_like(prior);hit[:2,0]=1;target=torch.where(hit.any(1),prior[:,0],.02)
    def run():
        lo,hi=pulse(f,n,valid,values,p);signal=torch.cat((lo,hi),-1)
        points=(anchors+signal).tanh()
        for _ in range(4):points=(anchors+signal+.2*relation(signal,p['fb_relational'])*points).tanh()
        delta=read(points,signal,p['fb_read']);return nll(lift(delta,code,p['fb_query']),prior,hit,target)
    run().backward();errs={};norms={}
    for k,a in p.items():
        idx=tuple(torch.unravel_index(a.grad.abs().argmax(),a.shape));g=float(a.grad[idx]);eps=1e-5
        with torch.no_grad():a[idx]+=eps;hi=float(run());a[idx]-=2*eps;lo=float(run());a[idx]+=eps
        errs[k]=abs((hi-lo)/(2*eps)-g);norms[k]=float(a.grad.norm())
    assert max(errs.values())<1e-8 and min(norms.values())>0
    with torch.no_grad():
        a,b=pulse(f,n,torch.zeros_like(valid),values,p);assert torch.count_nonzero(a)==torch.count_nonzero(b)==0
        signal=torch.cat((a,b),-1);assert torch.equal(relation(signal,p['fb_relational']),torch.ones_like(signal))
        assert torch.count_nonzero(read(anchors,signal,p['fb_read']))==0
        a,b=pulse(f,n,valid,values,p);r=relation(torch.cat((a,b),-1),p['fb_relational']);assert r.min()>=.98 and r.max()<=1.02
        assert nll(torch.zeros_like(prior),prior,hit,target)==-target.log().mean()
    print({'parameters':sum(a.numel() for a in p.values()),'gradient_groups':len(p),'max_error':max(errs.values()),'min_grad_norm':min(norms.values()),'empty_exact':True,'zero_lift_exact':True,'route_bounds':[float(r.min()),float(r.max())]})
if __name__=='__main__':preflight()
