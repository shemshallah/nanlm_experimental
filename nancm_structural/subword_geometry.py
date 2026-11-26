"""Complete bounded prefix code and shared compositional byte geometry."""
import torch
import numpy as np
SPLIT=tuple(b' etaoinr')
PAD=256
DEPTH=8
DIM=16
SINGLE=tuple(i for i in range(256) if i not in SPLIT)
TOKENS=tuple(bytes([i]) for i in SINGLE)+tuple(bytes([i,j]) for i in SPLIT for j in range(256))
INDEX={t:i for i,t in enumerate(TOKENS)}
def parse(data):
    out=[];i=0
    while i<len(data):
        n=2 if data[i] in SPLIT else 1
        if i+n>len(data):return out,data[i:]
        out.append(data[i:i+n]);i+=n
    return out,b''
def context(tokens):
    t=[(w[0],w[1] if len(w)==2 else PAD) for w in reversed(tokens[-DEPTH:])]
    return t+[(PAD,PAD)]*(DEPTH-len(t))
def compose(ids,code,pair):
    a=code[ids[...,0]]*(ids[...,0]!=PAD).unsqueeze(-1)
    b=code[ids[...,1]]*(ids[...,1]!=PAD).unsqueeze(-1)
    return a+b+(a*b)@pair
def second_logits(h,first,code,pair,query,cond,prior):
    # first [B,K]; score complete two-byte leaves using shared code.
    a=code[first];z=(h@query)[:,None,:]+a@cond
    b=code[:256];c=a[:,:,None,:]+b[None,None,:,:]+(a[:,:,None,:]*b[None,None,:,:])@pair
    return prior.log()+torch.einsum('bkd,bkvd->bkv',z,c)
def leaves(p,q):
    return np.concatenate((p[...,list(SINGLE)],(p[...,list(SPLIT),None]*q).reshape(*p.shape[:-1],-1)),axis=-1)
def preflight():
    torch.set_num_threads(2);torch.manual_seed(34)
    c=torch.randn(257,DIM,dtype=torch.float64,requires_grad=True)
    pair=torch.randn(DIM,DIM,dtype=c.dtype,requires_grad=True)
    q=torch.randn(128,DIM,dtype=c.dtype,requires_grad=True);cond=torch.randn(DIM,DIM,dtype=c.dtype,requires_grad=True)
    h=torch.randn(3,128,dtype=c.dtype);ids=torch.randint(0,256,(3,8,2));ids[0,0,1]=256
    first=torch.tensor(SPLIT[:3]).reshape(-1,1);p=torch.rand(3,1,256,dtype=c.dtype);p/=p.sum(-1,keepdim=True)
    def loss():
        z=h+compose(ids,c,pair).mean(1).repeat(1,8)
        return torch.nn.functional.cross_entropy(second_logits(z,first,c,pair,q,cond,p)[:,0],torch.tensor([0,255,32]))
    loss().backward();err=[]
    for a in (c,pair,q,cond):
        assert a.grad.norm()>0
        idx=tuple(torch.unravel_index(a.grad.abs().argmax(),a.shape));g=float(a.grad[idx]);eps=1e-5
        with torch.no_grad():
            a[idx]+=eps;hi=float(loss());a[idx]-=2*eps;lo=float(loss());a[idx]+=eps
        err.append(abs((hi-lo)/(2*eps)-g))
    assert max(err)<1e-7
    rng=np.random.default_rng(34);p=rng.random(256);p/=p.sum();q=rng.random((8,256));q/=q.sum(1,keepdims=True)
    v=leaves(p,q);assert len(v)==2296 and abs(v.sum()-1)<1e-14
    assert np.max(np.abs(v[248:].reshape(8,256).sum(1)-p[list(SPLIT)]))<1e-14
    raw=bytes(range(256))+b'\x00\xff UTF-8 \xc3\xa9\r\n ';t,tail=parse(raw);assert b''.join(t)+tail==raw
    print(dict(gradient_error=max(err),leaves=len(v),mass=float(v.sum()),roundtrip_bytes=len(raw),pending_suffix=tail.hex()))
if __name__=='__main__':preflight()
