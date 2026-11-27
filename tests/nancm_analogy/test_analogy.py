import numpy as np,torch
from analogy import Analogy,Credit,mix

def main():
    torch.set_num_threads(2);torch.manual_seed(53)
    code=np.zeros((13,4),np.float32);code[1]=code[2]=[1,0,0,0];code[3]=code[4]=[-1,0,0,0];code[5]=[0,1,0,0];code[6]=[0,0,1,0];code[7]=[0,0,0,1];code[8]=[0,-1,0,0];code[9]=[0,0,-1,0];code[10]=[0,0,0,-1];code[11]=[1,1,0,0];code[12]=[-1,1,0,0]
    docs=[[6,7,1,5,8]*20,[6,7,3,5,9]*20,[6,7,1,5,8]*20,[6,7,3,5,9]*20]
    bank=Analogy(code,docs);parent=np.ones(13)/13
    for h,t in [([6,7,2,5],8),([6,7,4,5],9)]:
        laws=bank.experts(h);assert max(laws[0][0],key=laws[0][0].get)==t
        excl=bank.retrieve(h,False,exclude=0)[0];assert max(excl,key=excl.get)==t
        for law,f in laws:
            if law:assert abs(sum(law.values())-1)<1e-12
        model=Credit().double();f=torch.tensor(np.stack([x[1] for x in laws])[None],dtype=torch.float64);valid=torch.tensor([[bool(x[0]) for x in laws]])
        w=model.weights(f,valid,0)[0].detach().numpy();assert mix(parent,[x[0] for x in laws],w) is parent
    rng=np.random.default_rng(53)
    def sample(n):
        rows=[]
        for i in range(n):
            s=int(rng.integers(2));h=[6,7,2 if s==0 else 4,5];t=8+s;rr=bank.experts(h)
            rows.append(([parent[t]]+[l.get(t,0.) for l,_ in rr],np.stack([f for _,f in rr]),[bool(l) for l,_ in rr],rr,t))
        return rows
    tr=sample(128);va=sample(64);model=Credit().double();opt=torch.optim.Adam(model.parameters(),lr=.05)
    def tensors(rows):return torch.tensor([r[0] for r in rows],dtype=torch.float64),torch.tensor(np.stack([r[1] for r in rows]),dtype=torch.float64),torch.tensor([r[2] for r in rows])
    a=tensors(tr);b=tensors(va);objective=lambda ds:-model(*ds).log().mean()
    before=float(objective(b).detach());objective(a).backward();g=model.gain.grad.clone();fd=0.
    for ix in np.ndindex(tuple(g.shape)):
        with torch.no_grad():model.gain[ix]+=1e-5;hi=float(objective(a));model.gain[ix]-=2e-5;lo=float(objective(a));model.gain[ix]+=1e-5
        fd=max(fd,abs((hi-lo)/2e-5-float(g[ix])))
    assert fd<1e-8
    for _ in range(128):opt.zero_grad();objective(a).backward();opt.step()
    after=float(objective(b).detach());nll=0.;hits=0;wrong_nll=0.;wrong_hits=0;err=0.
    for _,f,v,rr,t in va:
        wt=model.weights(torch.tensor(f[None],dtype=torch.float64),torch.tensor([v]))[0].detach().numpy();p=mix(parent,[l for l,_ in rr],wt);hits+=p.argmax()==t;nll-=np.log(p[t]);err=max(err,abs(p.sum()-1))
        wrong=bank.experts([6,7,4 if t==8 else 2,5]);q=mix(parent,[l for l,_ in wrong],wt);wrong_hits+=q.argmax()==t;wrong_nll-=np.log(q[t])
    assert hits==len(va) and wrong_hits==0 and after<before and err<1e-12
    local,_=bank.local([6,7,1,5,8,6,7,1,5]);assert max(local,key=local.get)==8
    print(dict(reference_rows=len(bank.rows),validation=len(va),nll_before=before,nll_after=after,actual_dense_nll=nll/len(va),correct_top1=hits,wrong_top1=wrong_hits,wrong_nll=wrong_nll/len(va),gradient_error=fd,mass_error=err,zero_exact=True,local_recall=True))
if __name__=='__main__':main()
