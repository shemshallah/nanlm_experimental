import numpy as np,torch
from episodic import experts,Credit,mix

def main():
    torch.set_num_threads(2);torch.manual_seed(54);rng=np.random.default_rng(54)
    assert all(not p for p,f in experts([]))
    h=[1,2,3,9,5,2,3,8]*8+[1,2,3]
    a,b,c=experts(h);assert max(a[0],key=a[0].get)==9
    h2=h[:-3]+[5,2,3];assert max(experts(h2)[0][0],key=experts(h2)[0][0].get)==8
    assert all(abs(sum(p.values())-1)<1e-12 for p,f in (a,b,c) if p)
    parent=np.ones(12)/12
    def sample(n):
        rows=[]
        for _ in range(n):
            y=int(rng.integers(2));past=h if y==0 else h2;t=9-y;rr=experts(past)
            rows.append(([parent[t]]+[p.get(t,0) for p,f in rr],np.stack([f for p,f in rr]),[bool(p) for p,f in rr]))
        return tuple(torch.tensor(np.asarray(v),dtype=torch.bool if i==2 else torch.float64) for i,v in enumerate(zip(*rows)))
    tr,va=sample(128),sample(64);model=Credit().double();opt=torch.optim.Adam(model.parameters(),lr=.04)
    def obj(ds):return -model(*ds).log().mean()
    before=float(obj(va).detach());obj(tr).backward();g=model.gain.grad.clone();fd=0.
    for ix in np.ndindex(tuple(g.shape)):
        with torch.no_grad():model.gain[ix]+=1e-5;hi=float(obj(tr));model.gain[ix]-=2e-5;lo=float(obj(tr));model.gain[ix]+=1e-5
        fd=max(fd,abs((hi-lo)/2e-5-float(g[ix])))
    for _ in range(128):opt.zero_grad();obj(tr).backward();opt.step()
    after=float(obj(va).detach());assert after<before and fd<1e-8
    f=torch.tensor(np.stack([f for p,f in (a,b,c)])[None],dtype=torch.float64);valid=torch.tensor([[bool(p) for p,f in (a,b,c)]])
    w=model.weights(f,valid,0)[0].detach().numpy();assert mix(parent,[p for p,f in (a,b,c)],w) is parent
    w=model.weights(f,valid)[0].detach().numpy();q=mix(parent,[p for p,f in (a,b,c)],w)
    assert q.argmax()==9
    print(dict(before=before,after=after,gradient_error=fd,normalization_error=abs(q.sum()-1),disabled_exact=True,deep_context_disambiguation=True,parameters=model.gain.numel()))
if __name__=='__main__':main()
