"""v92: train the signed field on within-length lexical identification.

Both length bands are predicted from the same causal history. The target's
length is used only in the conditional training loss, never as inference input.
Transport preserves each inherited known-word band's mass and the OOV law.
"""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from flat85 import Flat

H=32

def bands(vocab):
    return np.array([0 if 4<=len(w)<=6 else 1 if 7<=len(w)<=63 else -1 for w in vocab],np.int64)

class Content(nn.Module):
    def __init__(self,embedding,prior,members,vocab):
        super().__init__()
        emb=torch.as_tensor(embedding,dtype=torch.float32).detach().clone()
        p=np.asarray(prior,np.float64);band=bands(vocab)
        groups=[];groupband=[];cls=np.full(len(vocab),-1,np.int64);slot=cls.copy()
        for j in (0,1):
            for row in np.asarray(members):
                ids=[int(i) for i in row if i>=0 and band[i]==j]
                if not ids:continue
                g=len(groups);cls[ids]=g;slot[ids]=np.arange(len(ids))
                groups.append(ids);groupband.append(j)
        cap=max(map(len,groups));pad=np.full((len(groups),cap),-1,np.int64)
        for j,ids in enumerate(groups):pad[j,:len(ids)]=ids
        cp=np.array([p[ids].sum() for ids in groups])
        assert np.array_equal(cls>=0,band>=0) and set(groupband)=={0,1}
        for k,x in [('embedding',emb),('band',torch.from_numpy(band)),('members',torch.from_numpy(pad)),
                    ('groupband',torch.tensor(groupband)),('cls',torch.from_numpy(cls)),('slot',torch.from_numpy(slot)),
                    ('logprior',torch.from_numpy(np.log(np.maximum(p,1e-30))).float()),('classprior',torch.from_numpy(np.log(cp)).float())]:
            self.register_buffer(k,x)
        v,d=emb.shape
        self.project=nn.Linear(d,96)
        self.rail=nn.Linear(96*8,256)
        self.exchange=nn.Linear(256,256,bias=False)
        self.read=nn.Linear(256,128)
        self.norm=nn.LayerNorm(128)
        self.class_head=nn.Linear(128,len(groups))
        self.q=nn.Linear(128,128,bias=False)
        self.key=nn.Linear(d,128,bias=False)
        self.delta=nn.Embedding(v,32,padding_idx=0)
        self.delta_key=nn.Linear(32,128,bias=False)
        self.bias=nn.Parameter(torch.zeros(v))
        nn.init.normal_(self.exchange.weight,std=.025)
        nn.init.normal_(self.delta.weight,std=.02)
        with torch.no_grad():self.delta.weight[0].zero_()

    def state(self,h):
        assert h.shape[-1]==H
        x=self.project(self.embedding[h]).tanh();mask=h!=0
        x=x*mask.unsqueeze(-1)
        rails=[x[:,-j] for j in (1,2,3,4)]
        for lo,hi in ((4,8),(8,16),(16,32)):
            z=x[:,H-hi:H-lo];m=mask[:,H-hi:H-lo].sum(-1,keepdim=True).clamp_min(1)
            rails.append(z.sum(1)/m)
        content=(self.band[h]>=0)&mask
        rails.append((x*content.unsqueeze(-1)).sum(1)/content.sum(-1,keepdim=True).clamp_min(1))
        anchor=self.rail(torch.cat(rails,-1)).tanh()
        W=self.exchange.weight
        W=W/W.abs().sum(-1,keepdim=True).clamp_min(1.)
        s=anchor
        for _ in range(3):
            p,n=(anchor+.6*F.linear(s,W)).chunk(2,-1)
            s=torch.cat((p.tanh()*torch.sigmoid(p-n),n.tanh()*torch.sigmoid(n-p)),-1)
        return self.norm(self.read(s))

    def keys(self,ids):
        return self.key(self.embedding[ids])+self.delta_key(self.delta(ids))

    def class_log(self,z,band):
        q=self.class_head(z)+self.classprior
        return q.masked_fill(self.groupband[None]!=band[:,None],-torch.inf).log_softmax(-1)

    def target_log(self,h,y):
        band=self.band[y];assert (band>=0).all()
        z=self.state(h);c=self.cls[y];r=self.members[c];safe=r.clamp_min(0)
        q=self.q(z);score=(q[:,None]*self.keys(safe)).sum(-1)/math.sqrt(128.)+self.logprior[safe]+self.bias[safe]
        i=torch.arange(len(y),device=y.device)
        return self.class_log(z,band)[i,c]+score.masked_fill(r<0,-torch.inf).log_softmax(-1)[i,self.slot[y]]

    def log_probs(self,h,keys=None):
        z=self.state(h);q=self.q(z)
        if keys is None:keys=self.keys(torch.arange(len(self.embedding),device=h.device))
        score=q@keys.T/math.sqrt(128.)+self.logprior+self.bias
        r=self.members;safe=r.clamp_min(0)
        cond=score[:,safe].masked_fill(r[None]<0,-torch.inf).log_softmax(-1)
        ok=self.band>=0
        outs=[]
        for j in (0,1):
            g=self.class_log(z,torch.full((len(h),),j,dtype=torch.long,device=h.device))
            full=cond+g[:,:,None]
            v=torch.full_like(score,-torch.inf)
            v[:,ok]=full[:,self.cls[ok],self.slot[ok]]
            outs.append(v)
        return torch.stack(outs,1)

class Positions:
    def __init__(self,seqs,vocab,seed):
        self.seqs=seqs;self.band=bands(vocab)
        rng=np.random.default_rng(seed)
        rows=[]
        for b,s in enumerate(seqs):
            j=np.flatnonzero(self.band[s]>=0);j=j[j>0]
            rows.append(np.column_stack((np.full(len(j),b,np.int32),j.astype(np.int32))))
        self.pos=np.concatenate(rows);rng.shuffle(self.pos);self.at=0
    def take(self,n):
        assert self.at+n<=len(self.pos)
        rows=self.pos[self.at:self.at+n];self.at+=n
        x=np.zeros((n,H),np.int64);y=np.zeros(n,np.int64)
        for i,(b,j) in enumerate(rows):
            s=self.seqs[b];h=s[max(0,j-H):j];x[i,-len(h):]=h;y[i]=s[j]
        return torch.from_numpy(x),torch.from_numpy(y)

def history(prefix,ids,words,wrong=False):
    h=[ids.get(w,0) for w in words(prefix)[-H:]]
    if wrong and len(h)>4:h=h[:-4][::-1]+h[-4:]
    x=np.zeros(H,np.int64)
    if h:x[-len(h):]=h
    return torch.from_numpy(x)[None]

def transport(compiler,law,probs,band,gain):
    if gain==0:return law
    flat=compiler(law);p=flat.known.copy()
    for j in (0,1):
        mask=band[1:]==j;mass=math.fsum(p[mask])
        d=probs[j,1:][mask];d=d/math.fsum(d)
        p[mask]=(1-gain)*p[mask]+gain*mass*d
        assert abs(math.fsum(p[mask])-mass)<1e-12
    return Flat(compiler,p,flat.escape,flat.x)

def selftest():
    from open_word import Lift,Outside,Spelling
    from flat85 import Compiler
    torch.manual_seed(92);rng=np.random.default_rng(92)
    vocab=['','a','aaaa','bbbb','cccc','ddddddd','eeeeeee','fffffff']
    p=np.arange(1,9,dtype=np.float64);p/=p.sum()
    emb=rng.normal(size=(8,24)).astype(np.float32)
    members=np.array([[0,1,2,3],[4,5,6,7]])
    m=Content(emb,p,members,vocab)
    seqs=[np.tile(np.arange(1,8),40),np.tile(np.arange(7,0,-1),40)]
    pos=Positions(seqs,vocab,92);h,y=pos.take(32)
    lp=m.log_probs(h);tl=m.target_log(h,y)
    assert torch.allclose(lp[torch.arange(len(y)),m.band[y],y],tl,atol=2e-6)
    assert torch.allclose(lp.exp().sum(-1),torch.ones(len(y),2),atol=2e-6)
    opt=torch.optim.AdamW(m.parameters(),lr=.004)
    initial=float(-tl.mean().detach())
    for _ in range(40):
        opt.zero_grad();loss=-m.target_log(h,y).mean();loss.backward();opt.step()
    final=float(-m.target_log(h,y).mean().detach());assert final<initial
    assert m.exchange.weight.grad.abs().sum()>0 and m.delta.weight.grad.abs().sum()>0
    o=Outside(vocab[1:],Spelling([vocab[1:]]));a=Lift(o,p);comp=Compiler(o)
    pr=m.log_probs(h[:1])[0].detach().double().exp().numpy()
    b=transport(comp,a,pr,bands(vocab),.75)
    assert transport(comp,a,pr,bands(vocab),0.) is a
    assert b.complete('a')==a.complete('a') and b.complete('zzzz')==a.complete('zzzz')
    for j in (0,1):
        assert abs(sum(b.complete(w) for w,k in zip(vocab,bands(vocab)) if k==j)-sum(a.complete(w) for w,k in zip(vocab,bands(vocab)) if k==j))<1e-12
    for w in ('','a','aa','b','d','z'):
        assert abs(b.mass(w)-b.complete(w)-math.fsum(b.mass(w+c) for c in 'abcdefghijklmnopqrstuvwxyz'))<1e-11
    print('v92 exact length conditional likelihood, signed gradient, learned retrieval, band mass and OOV preservation',initial,final,flush=True)
if __name__=='__main__':
    torch.set_num_threads(2);selftest()
