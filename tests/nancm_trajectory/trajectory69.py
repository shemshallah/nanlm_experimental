"""v69 immutable reference trajectories + signed causal context transport."""
import math,re
from collections import defaultdict
from functools import lru_cache
import numpy as np

class Sparse:
    def __init__(self,scores):
        z=math.fsum(scores.values());assert z>0
        self.p={w:float(v/z) for w,v in scores.items() if v>0}
        self.prefix=defaultdict(float)
        for w,v in self.p.items():
            for i in range(len(w)+1):self.prefix[w[:i]]+=v
    def mass(self,p):return self.prefix.get(p,0.)
    def complete(self,w):return self.p.get(w,0.)

class Trajectory:
    """Retrieve TRAIN-only continuations using signed context and path geometry.
    The future side of each reference is known TRAIN text, not evaluation text.
    No target, target length, or future DEV bytes enter propose().
    """
    def __init__(self,words,code,docs):
        self.words=tuple(words);self.ids={w:i for i,w in enumerate(words)}
        x=np.asarray(code,dtype=np.float32)
        self.code=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
        self.code[0]=0
        blocks=[]
        for doc in docs:
            ids=np.array([self.ids.get(w,0) for w in doc],dtype=np.int32)
            if len(ids)<7:continue
            # 3 antecedents, next word, 2 prospective words; no document crossing.
            block=np.lib.stride_tricks.sliding_window_view(ids,6).copy()
            blocks.append(block[(block[:,2]>0)&(block[:,3]>0)])
        self.paths=np.concatenate(blocks,axis=0)
        order=np.argsort(self.paths[:,2],kind='stable')
        self.paths=self.paths[order]
        counts=np.bincount(self.paths[:,2],minlength=len(words))
        self.offset=np.concatenate(([0],np.cumsum(counts)))
        self.active=np.flatnonzero(counts>=4)
        self.active=self.active[self.active>0]
        self.valid=np.array([bool(re.fullmatch('[a-z]{1,63}',w)) for w in words])
        self.paths.setflags(write=False);self.code.setflags(write=False)
    @lru_cache(maxsize=4096)
    def neighbors(self,i):
        if i<=0 or not len(self.active):return ()
        score=self.code[self.active]@self.code[i]
        k=min(8,len(score));ix=np.argpartition(-score,k-1)[:k]
        return tuple(int(self.active[j]) for j in sorted(ix,key=lambda j:(-float(score[j]),int(self.active[j]))) if score[j]>.35 and self.active[j]!=i)
    def propose(self,root,parent,other,raw0,mode='full'):
        h=[w.decode('ascii').lower() for w in re.findall(rb'[A-Za-z]+',root)][-128:]
        if len(h)<3:return None,0.,{}
        q=np.array([self.ids.get(w,0) for w in h[-3:]],dtype=np.int32)
        if not q[-1]:return None,0.,{}
        # Keep actual immediate word, invert older conceptual addresses.
        if mode=='wrong':q=q[[1,0,2]]
        near=() if mode=='exact' else self.neighbors(int(q[-1]))
        candidates=[]
        for key in (int(q[-1]),*near):
            lo,hi=self.offset[key:key+2];p=self.paths[lo:hi]
            if not len(p):continue
            if key==q[-1]:
                # All exact two-word addresses, plus deterministic capped
                # background support. Exact matches never lost to subsampling.
                match=np.flatnonzero((p[:,1]==q[1])&(q[1]>0))
                if len(match):
                    keep=match[np.linspace(0,len(match)-1,min(4096,len(match)),dtype=int)]
                    candidates.append((p[keep],len(match)/len(keep)))
            ix=np.linspace(0,len(p)-1,min(384,len(p)),dtype=int)
            sub=p[ix]
            if key==q[-1] and q[1]>0:
                keep=sub[:,1]!=q[1];sub=sub[keep]
            if len(sub):candidates.append((sub,len(p)/len(ix)))
        if not candidates:return None,0.,{}
        votes=defaultdict(float);support=0.;exact_hits=0
        hist_ids=[self.ids.get(w,0) for w in h[-12:]]
        if mode=='wrong':hist_ids=hist_ids[::-1]
        topic=self.code[hist_ids].mean(0);topic/=max(float(np.linalg.norm(topic)),1e-12)
        for p,expansion in candidates:
            eq=(p[:,:3]==q)&(q>0)
            exact=eq@np.array([.15,.35,.5])
            sim=np.stack([self.code[p[:,j]]@self.code[q[j]] for j in range(3)],axis=1)
            signed=sim@np.array([.15,.35,.5])
            if mode=='exact':weight=exact**5*(eq[:,2])
            else:weight=exact**5+.15*np.maximum(signed-.25,0.)**3
            # Path evidence rewards agreement of TRAIN prospective centroid with
            # observed topic. Negative agreement inhibits rather than adds mass.
            if mode in ('full','wrong'):
                future=self.code[p[:,3:6]].mean(1)
                coherence=(future@topic)/np.maximum(np.linalg.norm(future,axis=1),1e-12)
                weight*=np.exp(np.clip(coherence,-1,1)*1.5)
            # Bound sample correction to avoid a common address swamping all others.
            weight*=min(16.,expansion)*self.valid[p[:,3]]
            support+=float(weight.sum());exact_hits+=int(np.count_nonzero(eq[:,1]&eq[:,2]))
            for i,v in zip(p[:,3],weight):
                if v>0:votes[int(i)]+=float(v)
        if not votes:return None,0.,{}
        # Candidate set can include words never seen in this DEV prefix.
        scores={}
        for i,v in sorted(votes.items(),key=lambda kv:(-kv[1],kv[0]))[:256]:
            w=self.words[i];b=ord(w[0])
            # Weak reciprocal native/contrastive agreement, never a hard veto.
            agreement=1.+min(3.,100.*math.sqrt(max(0.,parent.complete(w)*other.complete(w))))
            first=1.+min(1.,float(raw0[b]+raw0[b-32])*4.)
            scores[w]=v*agreement*first
        confidence=min(1.,math.log1p(support)/4.)
        return Sparse(scores),confidence,dict(candidates=len(scores),exact_hits=exact_hits,support=support)

def selftest():
    words=['?','a','b','c','d','e','f','g'];q=np.ones(256)/256
    docs=[list('abccdeabccde'),list('abcfgeabcfge')]
    field=Trajectory(words,np.eye(len(words)),docs)
    base=Sparse({w:1. for w in words[1:]})
    p,conf,_=field.propose(b'a b c ',base,base,q)
    assert p.complete('c')>0 and p.complete('f')>0 and conf>0
    assert p.complete('f')>0 # not present in the evaluation prefix
    for s in ('','a','c','f','z'):
        assert abs(p.mass(s)-p.complete(s)-sum(p.mass(s+c) for c in 'abcdefghijklmnopqrstuvwxyz'))<1e-12
    p2,_,_=field.propose(b'a b c ',base,base,q);assert p.p==p2.p
    assert len(field.paths)==sum(len(d)-5 for d in docs)
    assert not field.paths.flags.writeable
    print('v69 immutable trajectory, unseen-prefix candidate and normalization tests passed')
if __name__=='__main__':selftest()
