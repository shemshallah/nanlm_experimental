"""v78 corpus-scale signed address field; exact hierarchical softmax."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

H=16
class Address(nn.Module):
    def __init__(self,code,prior,members,cls,slot,identity):
        super().__init__()
        code=torch.as_tensor(code,dtype=torch.float32)/4.
        prior=torch.as_tensor(prior,dtype=torch.float64);prior=prior/prior.sum()
        members=torch.as_tensor(members,dtype=torch.long);cls=torch.as_tensor(cls,dtype=torch.long);slot=torch.as_tensor(slot,dtype=torch.long)
        identity=torch.as_tensor(identity,dtype=torch.float32)
        v,d=code.shape;G=len(members)
        for k,x in [('code',code),('logprior',prior.log().float()),('members',members),('cls',cls),('slot',slot),('identity',identity.clone())]:
            self.register_buffer(k,x)
        cp=torch.zeros(G,dtype=torch.float64).scatter_add_(0,cls,prior)
        self.register_buffer('classprior',cp.log().float())
        # Learn the signed address, not a new independent spelling system.
        self.delta=nn.Embedding(v,32,padding_idx=0);nn.init.normal_(self.delta.weight,std=.02)
        with torch.no_grad():self.delta.weight[0].zero_()
        self.project=nn.Linear(d+identity.shape[1]+32,96)
        self.rail=nn.Linear(96*6,256)
        self.exchange=nn.Linear(256,256,bias=False)
        nn.init.zeros_(self.exchange.weight)
        self.read=nn.Linear(256,96)
        self.class_head=nn.Linear(96,G)
        self.word_head=nn.Linear(96,96,bias=False)
        self.word_bias=nn.Parameter(torch.zeros(v))
        self.key=nn.Linear(d+identity.shape[1]+32,96,bias=False)
        self.norm=nn.LayerNorm(96)
    def embed(self,h):
        return torch.cat((self.code[h],self.identity[h],self.delta(h)),-1)
    def state(self,h):
        assert h.shape[-1]==H
        x=torch.tanh(self.project(self.embed(h)))
        x=x*(h!=0).unsqueeze(-1)
        rails=[x[:,-1],x[:,-2],x[:,-3],x[:,-4]]
        for lo,hi in ((4,8),(8,16)):
            z=x[:,H-hi:H-lo];m=(h[:,H-hi:H-lo]!=0).sum(-1,keepdim=True).clamp_min(1)
            rails.append(z.sum(1)/m)
        p,n=self.rail(torch.cat(rails,-1)).chunk(2,-1)
        p,n=F.softplus(p),F.softplus(n)
        a,b=self.exchange(torch.cat((p,n),-1)).chunk(2,-1)
        # One mutually inhibited positive/negative coordinate pair.
        p=p*torch.sigmoid(a-b);n=n*torch.sigmoid(b-a)
        return self.norm(self.read(torch.cat((p,n),-1)))
    def class_log(self,z):
        return (self.class_head(z)+self.classprior).log_softmax(-1)
    def target_log(self,h,y):
        z=self.state(h);c=self.cls[y];r=self.members[c];safe=r.clamp_min(0)
        q=self.word_head(z);keys=self.key(self.embed(safe))
        score=(q[:,None,:]*keys).sum(-1)/np.sqrt(96.)+self.logprior[safe]+self.word_bias[safe]
        score=score.masked_fill(r<0,-torch.inf)
        i=torch.arange(len(y),device=y.device)
        return self.class_log(z)[i,c]+score.log_softmax(-1)[i,self.slot[y]]
    def log_probs(self,h):
        z=self.state(h);q=self.word_head(z)
        ids=torch.arange(len(self.code),device=h.device)
        score=q@self.key(self.embed(ids)).T/np.sqrt(96.)+self.logprior+self.word_bias
        r=self.members;safe=r.clamp_min(0)
        conditional=score[:,safe].masked_fill(r[None]<0,-torch.inf).log_softmax(-1)
        full=conditional+self.class_log(z)[:,:,None]
        return full[:,self.cls,self.slot]
    def root(self,prefix,ids,words,wrong=False):
        h=[ids.get(w,0) for w in words(prefix)[-H:]]
        if wrong and len(h)>1:h=h[::-1]
        x=np.zeros(H,np.int64)
        if h:x[-len(h):]=h
        return torch.from_numpy(x)[None]

def batch(seqs,rng,n):
    lens=np.array([max(0,len(s)-1) for s in seqs]);book=rng.choice(len(seqs),size=n,p=lens/lens.sum())
    x=np.zeros((n,H),np.int64);y=np.zeros(n,np.int64);positions=[]
    for i,b in enumerate(book):
        s=seqs[b];j=int(rng.integers(1,len(s)));h=s[max(0,j-H):j]
        x[i,-len(h):]=h;y[i]=s[j];positions.append((int(b),j))
    return torch.from_numpy(x),torch.from_numpy(y),positions

def selftest():
    torch.manual_seed(78);rng=np.random.default_rng(78)
    code=rng.normal(size=(13,8)).astype(np.float32);code[0]=0
    members=np.array([[0,-1,-1,-1],[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    cls=np.repeat([0,1,2,3],[1,4,4,4]);slot=np.r_[0,np.tile(np.arange(4),3)]
    m=Address(code,np.ones(13)/13,members,cls,slot,np.zeros((13,6),np.float32))
    docs=[np.tile([1,2,3,4,5,6],40),np.tile([7,8,9,10,11,12],40)]
    h,y,_=batch(docs,rng,32);dense=m.log_probs(h);sparse=m.target_log(h,y)
    assert torch.allclose(dense[torch.arange(len(y)),y],sparse,atol=2e-6)
    assert torch.allclose(dense.exp().sum(-1),torch.ones(len(y)),atol=2e-6)
    opt=torch.optim.AdamW(m.parameters(),lr=.006)
    initial=float(-sparse.mean().detach())
    for _ in range(40):
        opt.zero_grad();loss=-m.target_log(h,y).mean();loss.backward();opt.step()
    final=float(-m.target_log(h,y).mean().detach())
    assert final<initial*.5,(initial,final)
    assert m.delta.weight.grad.abs().sum()>0 and m.exchange.weight.grad.abs().sum()>0
    print('v78 exact hierarchical parity, conservation and signed address learning',initial,final)
if __name__=='__main__':
    torch.set_num_threads(2);selftest()
