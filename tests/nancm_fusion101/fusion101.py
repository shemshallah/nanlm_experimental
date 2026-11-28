"""Causal, calibrated mixture of the inherited corpus and chat laws.

Features are computed solely from next-byte distributions before observation.
The observed target contributes only to training loss. No answer-derived
features, lexical rules or inference-time target access are permitted.
"""
import math
import numpy as np
import torch
from torch import nn

EPS=1e-30

def features(base,chat):
    b=np.asarray(base,dtype=np.float64)
    c=np.asarray(chat,dtype=np.float64)
    if b.shape!=c.shape or b.shape[-1]!=256:
        raise ValueError("expected matching 256-byte distributions")
    for q in (b,c):
        if not np.isfinite(q).all() or np.any(q<0) or not np.allclose(q.sum(-1),1.,atol=1e-8):
            raise ValueError("invalid probability distribution")
    def entropy(q):
        return -(q*np.log(np.maximum(q,EPS))).sum(-1)/math.log(256)
    m=.5*(b+c)
    js=entropy(m)-.5*(entropy(b)+entropy(c))
    return np.stack((entropy(b),entropy(c),b.max(-1),c.max(-1),
                     js,np.abs(b-c).sum(-1)*.5,
                     (b.argmax(-1)==c.argmax(-1)).astype(float),
                     b[...,4],c[...,4]),axis=-1)

class Router(nn.Module):
    def __init__(self,fixed=.5):
        super().__init__()
        if not 0.<fixed<1.:raise ValueError("fixed gain must be inside (0,1)")
        self.register_buffer("fixed",torch.tensor(fixed,dtype=torch.float64))
        self.register_buffer("mean",torch.zeros(9,dtype=torch.float64))
        self.register_buffer("std",torch.ones(9,dtype=torch.float64))
        self.linear=nn.Linear(9,1,dtype=torch.float64)
        nn.init.zeros_(self.linear.weight);nn.init.zeros_(self.linear.bias)
        self.scale=1.
    def forward(self,x):
        if self.scale==0.:return self.fixed.expand(x.shape[:-1])
        # Limit changes in log odds to prevent an uncalibrated expert takeover.
        z=2.*torch.tanh(self.linear((x-self.mean)/self.std).squeeze(-1))
        return torch.sigmoid(torch.logit(self.fixed)+self.scale*z)
    @torch.no_grad()
    def gain(self,b,c):
        return float(self(torch.from_numpy(features(b,c))))
    def mix(self,b,c):
        g=self.gain(b,c)
        return (1-g)*b+g*c,g

def fit_router(router,records,steps=128):
    """Each record holds pre-observation features and two target probabilities.
    Fit only on the designated calibration cohort, never generation or DEV."""
    x=torch.from_numpy(np.concatenate([r[0] for r in records]))
    b=torch.from_numpy(np.concatenate([r[1] for r in records]))
    c=torch.from_numpy(np.concatenate([r[2] for r in records]))
    # Balance by record, not merely by number of bytes.
    weights=torch.from_numpy(np.concatenate([np.full(len(r[1]),1./len(r[1])) for r in records]))
    weights/=weights.sum()
    with torch.no_grad():
        router.mean.copy_(x.mean(0))
        router.std.copy_(x.std(0,unbiased=False).clamp_min(.02))
    opt=torch.optim.AdamW(router.parameters(),lr=.02,weight_decay=.01)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        g=router(x)
        q=(1-g)*b+g*c
        loss=-(weights*q.clamp_min(EPS).log()).sum()+.02*((g-router.fixed).square()*weights).sum()
        if not torch.isfinite(loss):raise FloatingPointError("nonfinite router loss")
        loss.backward();torch.nn.utils.clip_grad_norm_(router.parameters(),1.,error_if_nonfinite=True)
        opt.step()
    return router.eval()

def score(router,records):
    bits={k:0. for k in ("base","chat","fixed","adaptive")}
    gains=[];n=0
    with torch.no_grad():
        for x,b,c in records:
            g=router(torch.from_numpy(x)).numpy()
            f=float(router.fixed)
            for k,q in (("base",b),("chat",c),("fixed",(1-f)*b+f*c),("adaptive",(1-g)*b+g*c)):
                bits[k]-=float(np.log2(np.maximum(q,EPS)).sum())
            gains.extend(g.tolist());n+=len(b)
    return {"records":len(records),"bytes":n,"bpb":{k:v/n for k,v in bits.items()},
            "gain_mean":float(np.mean(gains)),"gain_min":float(np.min(gains)),"gain_max":float(np.max(gains))}

def selftest():
    rng=np.random.default_rng(101)
    b=rng.dirichlet(np.ones(256),32);c=rng.dirichlet(np.ones(256),32)
    router=Router()
    router.scale=0.
    for a,d in zip(b,c):
        q,g=router.mix(a,d);assert g==.5 and np.array_equal(q,.5*a+.5*d)
    router.scale=1.
    assert np.allclose(router(torch.from_numpy(features(b,c))).detach().numpy(),.5)
    # Known synthetic routing law; no test claims about language performance.
    base=np.tile(np.full(256,.1/255),(32,1));base[:,0]=.9
    chat=np.tile(np.full(256,1./256),(32,1));chat[:16]=base[:16,::-1]
    y=np.r_[np.full(16,255),np.zeros(16,dtype=int)].astype(int)
    r=(features(base,chat),base[np.arange(32),y],chat[np.arange(32),y])
    fit_router(router,[r])
    s=score(router,[r]);assert s["bpb"]["adaptive"]<s["bpb"]["fixed"]
    for a,d in zip(b,c):
        q,g=router.mix(a,d);assert abs(q.sum()-1)<1e-12 and 0<g<1
    print("v101 router: exact fixed fallback, normalized bounded causal fusion and learnable expert selection passed",flush=True)

if __name__=="__main__":
    torch.set_num_threads(2)
    selftest()
