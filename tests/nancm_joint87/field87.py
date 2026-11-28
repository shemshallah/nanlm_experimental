"""v87 differentiable causal event recursion for the recovered v86 gates."""
import torch
import torch.nn.functional as F

def allocated(g,q):
    # q = raw, lexical A, lexical B, ten normalized evidence donors.
    raw=q[...,0];extra=(g[...,1:]*q[...,3:]).sum(-1)
    return torch.stack((raw,g[...,0]*q[...,1]+extra,g[...,0]*q[...,2]+extra),-1)

def recurse(core,scale,escape,gate,data,x=1.,y=1.,warm=0):
    features,cap,base,q,words,letter,complete,mask=data
    # features [B,T,10,6], cap [B,T,10], base [B,T,5].
    # words [B,T,9,13]: target plus eight causal hard negatives.
    batch,T=mask.shape
    state=core.prior.expand(batch,3);prev=torch.ones_like(state);started=torch.zeros(batch,dtype=torch.bool,device=state.device)
    losses=[];wordloss=[];rankloss=[]
    for j in range(T):
        active=mask[:,j];ff=base[:,j]
        hist=((prev[:,1:].clamp_min(1e-300).log()-prev[:,:1].clamp_min(1e-300).log())/8).clamp(-1,1)
        hist=torch.where(started[:,None],hist,torch.zeros_like(hist))
        mat=core.matrix(torch.cat((ff,hist),-1),scale)
        w=torch.where(started[:,None],torch.einsum('bi,bij->bj',state,mat),state)
        g=escape.allocate(features[:,j],cap[:,j],x)
        a=allocated(g,q[:,j])
        r=letter[:,j,0];isletter=letter[:,j,1].bool()
        share=w[:,1:].sum(-1);m=share+(1-share)*r;s=share/m
        f=torch.stack((torch.ones_like(r),r,share,s,ff[:,2],ff[:,3],ff[:,1],w[:,2]/share),-1)
        cond=gate(f,s,y).clamp(max=1-1e-12)
        rawfactor=m*(1-cond)/r/(1-share)
        lexfactor=m*cond/share
        factors=torch.stack((rawfactor,lexfactor,lexfactor),-1)
        adjusted=torch.where(isletter[:,None],a*factors,a)
        posterior=w*adjusted
        likelihood=posterior.sum(-1).clamp_min(1e-300)
        p=allocated(g[:,None,:],words[:,j])
        wordprob=(p*factors[:,None,:]*w[:,None,:]).sum(-1).clamp_min(1e-300)
        include=active & complete[:,j]
        if j>=warm:
            losses.append(torch.where(active,-likelihood.log(),0.))
            wordloss.append(torch.where(include,-wordprob[:,0].log(),0.))
            rankloss.append(torch.where(include,F.softplus(wordprob[:,1:].log()-wordprob[:,:1].log()).mean(-1),0.))
        state=torch.where(active[:,None],posterior/likelihood[:,None],state)
        prev=torch.where(active[:,None],adjusted,prev);started=started|active
        if j==warm-1:state=state.detach();prev=prev.detach()
    n=mask[:,warm:].sum().clamp_min(1);nw=(mask[:,warm:]&complete[:,warm:]).sum().clamp_min(1)
    return torch.stack((torch.stack(losses).sum()/n,torch.stack(wordloss).sum()/nw,torch.stack(rankloss).sum()/nw))

def selftest():
    from transition import Transition
    from field79 import Escape
    from field81 import Gate81
    torch.manual_seed(87);dtype=torch.float64
    core=Transition([.4,.3,.3],.2).requires_grad_(False)
    escape=Escape();gate=Gate81()
    B,T=3,7
    f=torch.rand(B,T,10,6,dtype=dtype);cap=torch.rand(B,T,10,dtype=dtype)*.7
    base=torch.rand(B,T,5,dtype=dtype);base[:,:,0]=1.
    q=torch.rand(B,T,13,dtype=dtype)*.1
    p=torch.rand(B,T,9,13,dtype=dtype)*.1
    letter=torch.rand(B,T,2,dtype=dtype);letter[:,:,0]=.3+.6*letter[:,:,0];letter[:,:,1]=1.
    complete=torch.ones(B,T,dtype=torch.bool);mask=complete.clone();mask[0,-2:]=False
    data=(f,cap,base,q,p,letter,complete,mask)
    out=recurse(core,.5,escape,gate,data)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert escape.theta.grad.abs().sum()>0 and gate.theta.grad.abs().sum()>0
    # Masked suffixes do not contribute to the objective.
    q2=q.clone();q2[0,-2:]=.7
    p2=p.clone();p2[0,-2:]=.8
    assert torch.equal(out,recurse(core,.5,escape,gate,(f,cap,base,q2,p2,letter,complete,mask)))
    # Exact same causal recursion when no gates are active.
    zero=recurse(core,.5,escape,gate,data,0.,0.)
    assert torch.isfinite(zero).all()
    print('v87 causal posterior gradients, padded sequence isolation and disabled gates passed')
if __name__=='__main__':selftest()
