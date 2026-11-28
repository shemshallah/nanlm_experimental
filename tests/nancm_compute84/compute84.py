"""v84 exact computation reuse, no probability model changes."""
from functools import lru_cache
import math,time
import numpy as np
import torch
from field76 import Suffix
from field78 import Address

class FastSuffix(Suffix):
    """Same immutable suffix index; binary search directly in the address view."""
    def __init__(self,seqs,order=8,classes=None):
        super().__init__(seqs,order,classes)
        a=self.address[self.pos-1]
        v,first=np.unique(a,return_index=True)
        self.first={int(x):(int(l),int(h)) for x,l,h in zip(v,first,np.r_[first[1:],len(a)])}
        self.values=v
        self.starts=first
    def _bisect(self,lo,hi,k,v,right):
        pos=self.pos;addr=self.address
        while lo<hi:
            m=(lo+hi)//2;x=int(addr[int(pos[m])-k])
            if x<v or (right and x==v):lo=m+1
            else:hi=m
        return lo
    @lru_cache(maxsize=4096)
    def bounds(self,h):
        if not h:return 0,len(self.pos)
        k=len(h);v=h[0]
        if k==1:
            if not v:return 0,0
            if v in self.first:return self.first[v]
            i=int(np.searchsorted(self.values,v));j=int(self.starts[i]) if i<len(self.starts) else len(self.pos)
            return j,j
        lo,hi=self.bounds(h[1:])
        if not v or lo==hi:return lo,lo
        l=self._bisect(lo,hi,k,v,False)
        r=self._bisect(l,hi,k,v,True)
        return l,r

class FrozenAddress:
    """Inference-only cached key matrix with explicit frozen-parameter guard."""
    def __init__(self,model,max_histories=16):
        assert not model.training and all(not p.requires_grad for p in model.parameters())
        self.model=model;self.max_histories=max_histories
        self.versions=tuple((p,p._version) for p in model.parameters())
        self.buffers=tuple((b,b._version) for b in model.buffers())
        with torch.no_grad():
            ids=torch.arange(len(model.code),device=model.code.device)
            self.keys=model.key(model.embed(ids))
        self.cache=lru_cache(maxsize=max_histories)(self._predict)
    def guard(self):
        assert not self.model.training and all(not p.requires_grad and p._version==v for p,v in self.versions)
        assert all(b._version==v for b,v in self.buffers)
    @torch.no_grad()
    def _predict(self,history,shape):
        m=self.model
        h=torch.tensor(history,dtype=torch.long,device=m.code.device).reshape(shape)
        z=m.state(h);q=m.word_head(z)
        score=q@self.keys.T/np.sqrt(96.)+m.logprior+m.word_bias
        r=m.members;safe=r.clamp_min(0)
        conditional=score[:,safe].masked_fill(r[None]<0,-torch.inf).log_softmax(-1)
        full=conditional+m.class_log(z)[:,:,None]
        return full[:,m.cls,m.slot]
    def log_probs(self,h):
        self.guard()
        # Return an independent tensor so callers cannot mutate the cache.
        return self.cache(tuple(h.detach().cpu().flatten().tolist()),tuple(h.shape)).clone()
    def root(self,*args,**kwargs):return self.model.root(*args,**kwargs)
    def stats(self):return self.cache.cache_info()._asdict()

def selftest():
    rng=np.random.default_rng(84);torch.manual_seed(84);torch.set_num_threads(2)
    # Skewed corpus: large one-word ranges are common in actual natural language.
    seqs=[np.minimum(rng.zipf(1.35,size=15000),2048).astype(np.int32) for _ in range(12)]
    old=Suffix(seqs,8);new=FastSuffix(seqs,8)
    assert np.array_equal(old.flat,new.flat) and np.array_equal(old.pos,new.pos)
    assert np.array_equal(old.address,new.address)
    queries=[]
    for _ in range(1200):
        d=seqs[int(rng.integers(len(seqs)))];i=int(rng.integers(8,len(d)));k=int(rng.integers(1,9))
        queries.append(tuple(map(int,d[i-k:i])))
    queries.extend([(0,),(0,1),(3000,),(1,3000),(1,0,1)])
    def measure(idx):
        t=time.perf_counter();out=[idx.bounds(h) for h in queries]
        return out,time.perf_counter()-t
    a,t0=measure(old);b,t1=measure(new);assert a==b
    for h in queries[::7]:
        assert all(np.array_equal(x,y) for x,y in zip(old.counts(h),new.counts(h)))
    print('SUFFIX84',dict(positions=len(old.pos),queries=len(queries),old_seconds=t0,new_seconds=t1),flush=True)
    # A realistically sized vocabulary with bounded hierarchical classes.
    v=32769;g=257
    members=np.full((g,128),-1,np.int64);members[0,0]=0
    members[1:]=np.arange(1,v).reshape(256,128)
    cls=np.r_[0,np.repeat(np.arange(1,g),128)];slot=np.r_[0,np.tile(np.arange(128),256)]
    code=rng.normal(size=(v,128)).astype(np.float32)
    identity=rng.normal(size=(v,64)).astype(np.float32)
    m=Address(code,np.ones(v)/v,members,cls,slot,identity).eval().requires_grad_(False)
    fast=FrozenAddress(m)
    hs=[torch.from_numpy(rng.integers(0,v,size=(1,16),dtype=np.int64)) for _ in range(24)]
    hs=[x for h in hs for x in (h,h,h)]
    with torch.no_grad():
        t=time.perf_counter();a=[m.log_probs(h) for h in hs];t0=time.perf_counter()-t
        t=time.perf_counter();b=[fast.log_probs(h) for h in hs];t1=time.perf_counter()-t
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    assert fast.stats()['hits']==48
    b[0].zero_();assert torch.equal(fast.log_probs(hs[0]),a[0])
    with torch.no_grad():m.word_bias[0]+=1.
    try:fast.log_probs(hs[0]);raise RuntimeError('stale cache accepted')
    except AssertionError:pass
    print('ADDRESS84',dict(vocabulary=v,events=len(hs),old_seconds=t0,new_seconds=t1,cache=fast.stats()),flush=True)
if __name__=='__main__':selftest()
