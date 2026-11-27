import torch,numpy as np
from pair_field import PairField,PairInteraction
from hierarchical_field import Field

def main():
    torch.set_num_threads(2);torch.manual_seed(52);rng=np.random.default_rng(52)
    code=rng.normal(size=(37,16)).astype(np.float32);code[0]=0;prior=rng.random(37)+.1;prior/=prior.sum()
    parent=Field(code,prior,cap=5,rank=4,identity=4).double()
    model=PairField(code,prior,cap=5,rank=4,identity=4).double()
    missing,unexpected=model.load_state_dict(parent.state_dict(),strict=False);assert all(k.startswith('pair.') for k in missing) and not unexpected
    x=torch.randn(9,8,16,dtype=torch.float64);ids=torch.randint(0,37,(9,4));y=torch.randint(0,37,(9,))
    model.pair_scale=0.;assert torch.equal(parent.log_probs(x,ids),model.log_probs(x,ids));model.pair_scale=1.
    logs=model.log_probs(x,ids);sparse=model.target_logprob(x,ids,y);err=float((logs[torch.arange(len(y)),y]-sparse).detach().abs().max());mass=float((logs.exp().sum(-1)-1).detach().abs().max());assert max(err,mass)<1e-12
    loss=lambda:-model.target_logprob(x,ids,y).mean();loss().backward();fd=0.;norms={}
    for k,p in model.pair.named_parameters():
        norms[k]=float(p.grad.norm());ix=tuple(torch.unravel_index(p.grad.abs().argmax(),p.shape));g=float(p.grad[ix]);v=float(p[ix].detach());eps=1e-5
        with torch.no_grad():p[ix]=v+eps;hi=float(loss());p[ix]=v-eps;lo=float(loss());p[ix]=v
        fd=max(fd,abs((hi-lo)/(2*eps)-g))
    assert min(norms.values())>0 and fd<1e-8
    print(dict(pair_count=28,pair_parameters=sum(p.numel() for p in model.pair.parameters()),zero_exact=True,normalization_error=mass,target_error=err,gradient_error=fd,norms=norms))
    def sample(n):
        a=torch.randint(0,2,(n,))*2-1;b=torch.randint(0,2,(n,))*2-1
        x=torch.randn(n,8,16)*.05;x[:,0,0]=a;x[:,1,0]=b
        return x,(a==b).long()
    tr,ty=sample(512);va,vy=sample(256);pair=PairInteraction(16,8);read=torch.nn.Linear(16,2);opt=torch.optim.Adam(list(pair.parameters())+list(read.parameters()),lr=.03)
    def metric():
        with torch.no_grad():z=read(pair(va));return float(torch.nn.functional.cross_entropy(z,vy)),float((z.argmax(1)==vy).float().mean())
    before=metric()
    for i in range(192):
        ix=torch.randint(len(ty),(32,));opt.zero_grad();torch.nn.functional.cross_entropy(read(pair(tr[ix])),ty[ix]).backward();opt.step()
    after=metric();print(dict(before=before,after=after));assert after[1]>.95
if __name__=='__main__':main()
