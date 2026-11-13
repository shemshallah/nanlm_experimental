#!/usr/bin/env python3
"""FA1 R7.22: causal embedding-based associative retrieval (NOT proven semantic QA).

Uses the frozen inherited R7 vocabulary embeddings, a fixed 10-bit random
hyperplane sketch and one-bit neighbor buckets to gather *old* history entries.
Query/route never uses target; target is accessed solely to score likelihood.
Requires strictly one document per memory; reset after a document boundary.
The 32K-exclusion band is preserved to avoid overlapping inherited near reader.
The learned gate is a single submodule of the R7 model, not another model.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import torch
from torch import nn

HORIZON=131072
EXCLUSION=32768
STRIDE=8
DIM=96
N_BITS=10
FEATURES=16
WEIGHTS=np.array([.1,.2,.3,.4],dtype=np.float32)
EMPTY={'n':0,'target_probability':0.,'modal':1,'peak':0.,'entropy':0.,
       'recent':0.,'max_cosine':0.,'mean_cosine':0.,'effective_n':0.,
       'lexical_overlap':0.,'hash_radius':0,'feature':np.zeros(FEATURES,dtype=np.float32)}


def _unit(v):
    v=np.asarray(v,dtype=np.float32)
    d=np.sqrt(np.square(v).sum(axis=-1,keepdims=True,dtype=np.float32))
    return v/np.maximum(d,1e-8)


def _signatures(vecs,planes):
    dots=np.asarray(vecs,dtype=np.float32)@planes
    return ((dots>=0).astype(np.int32)*(1<<np.arange(N_BITS,dtype=np.int32))).sum(axis=-1)


class CausalSemanticIndex:
    """Document-local, stride-8 embedding sketches over disjoint far memory.

    This is an experimental approximate retriever. Its result is not
    guaranteed identical to full nearest-neighbor search, but every retrieved
    candidate is strictly prior to the query and inside the eligible age band.
    """
    def __init__(self,sequence,embeddings,position,document_start=0,
                 stride=STRIDE,max_hits=64,max_step=1024,
                 horizon=HORIZON,exclusion=EXCLUSION):
        self.sequence=np.asarray(sequence)
        self.embeddings=_unit(np.asarray(embeddings,dtype=np.float32))
        if self.embeddings.shape[1]!=DIM:raise ValueError('R7 inherited embeddings must be 96D')
        self.position=int(position);self.document_start=int(document_start)
        self.stride=int(stride);self.max_hits=int(max_hits);self.max_step=int(max_step)
        self.horizon=int(horizon);self.exclusion=int(exclusion)
        if not self.horizon>self.exclusion+8:raise ValueError('invalid tier boundaries')
        if self.position<self.document_start or self.position>len(self.sequence):raise ValueError('invalid position')
        if self.stride<1:raise ValueError('stride must be >=1')
        rng=np.random.default_rng(722)
        self.planes=_unit(rng.standard_normal((N_BITS,DIM)).astype(np.float32)).T.copy()
        self.reset_count=0;self.incremental_count=0
        self._rebuild()

    def _context_vectors(self,ps):
        ps=np.asarray(ps,dtype=np.int64)
        ids=self.sequence[ps[:,None]-np.arange(3,-1,-1,dtype=np.int64)]
        valid=(ids>1).all(axis=1) & (ids<self.embeddings.shape[0]).all(axis=1)
        # Guard on indexing even for invalid rows.
        z=self.embeddings[np.clip(ids,0,len(self.embeddings)-1)]
        vec=_unit((z*WEIGHTS[None,:,None]).sum(axis=1))
        return vec,valid

    def _rebuild(self):
        lo=max(self.document_start,self.position-self.horizon)
        hi=self.position-self.exclusion-2
        ps=np.arange((lo+3+self.stride-1)//self.stride*self.stride,hi+1,
                     self.stride,dtype=np.int32) if hi>=lo+3 else np.empty(0,dtype=np.int32)
        if len(ps):
            vec,valid=self._context_vectors(ps)
            ps=ps[valid];vec=vec[valid]
            hashes=_signatures(vec,self.planes)
        else:
            vec=np.empty((0,DIM),np.float32);hashes=np.empty(0,np.int32)
        self.positions=ps
        self.vectors=vec
        self.hashes=hashes
        self.buckets=defaultdict(list)
        for j,h in enumerate(hashes):self.buckets[int(h)].append(j)
        self.pending=[]
        self.reset_count+=1

    def advance(self,position):
        t=int(position)
        if t<self.position or t>len(self.sequence):raise ValueError('noncausal or out-of-bounds seek')
        if t-self.position>self.max_step:
            self.position=t;self._rebuild();return self
        old=self.position;self.position=t
        self.incremental_count+=t-old
        # New positions become eligible as the 32K exclusion window advances.
        newps=[p for p in range(old-self.exclusion-1,t-self.exclusion-1)
               if p%self.stride==0 and p>=self.document_start+3 and p+1<len(self.sequence)]
        if newps:
            arr=np.asarray(newps,dtype=np.int32)
            vec,valid=self._context_vectors(arr)
            arr=arr[valid];vec=vec[valid]
            hashes=_signatures(vec,self.planes)
            for p,v,h in zip(arr,vec,hashes):self.pending.append((int(p),v,int(h)))
        if len(self.pending)>512:self._rebuild()
        return self

    def _candidate_vectors(self,h):
        ids=self.buckets.get(int(h),())
        if ids:
            p=self.positions[ids];v=self.vectors[ids]
        else:
            p=np.empty(0,dtype=np.int32);v=np.empty((0,DIM),dtype=np.float32)
        extras=[(q,w) for q,w,hashv in self.pending if hashv==h]
        if extras:
            ep=np.asarray([e[0] for e in extras],dtype=np.int32)
            ev=np.asarray([e[1] for e in extras],dtype=np.float32)
            p=np.concatenate((p,ep));v=np.concatenate((v,ev))
        return p,v

    def query(self,target=None,position=None,context=None):
        if position is not None:self.advance(position)
        t=self.position
        if t<self.document_start+4:return dict(EMPTY,feature=EMPTY['feature'].copy())
        qids=np.asarray(self.sequence[t-4:t] if context is None else context,dtype=np.int64)
        if len(qids)<4 or np.any(qids<=1) or np.any(qids>=len(self.embeddings)):
            return dict(EMPTY,feature=EMPTY['feature'].copy())
        q=_unit((self.embeddings[qids]*WEIGHTS[:,None]).sum(axis=0))
        h=int(_signatures(q[None,:],self.planes)[0])
        cutoff=t-self.exclusion-2;low=max(self.document_start,t-self.horizon)
        ps=[];vs=[];radii=[]
        for k in [h]+[h^(1<<bit) for bit in range(N_BITS)]:
            p,v=self._candidate_vectors(k)
            if not len(p):continue
            valid=(p>=low+3)&(p<=cutoff)
            if valid.any():
                ps.append(p[valid]);vs.append(v[valid]);radii.extend([int(k!=h)]*int(valid.sum()))
        if not ps:return dict(EMPTY,feature=EMPTY['feature'].copy())
        p=np.concatenate(ps);v=np.concatenate(vs)
        radius=np.asarray(radii,dtype=np.int8)
        sim=v@q
        # Top-k scoring is target-free. A strict age limit was applied above.
        order=np.argsort(-sim,kind='stable')[:self.max_hits]
        p=p[order];sim=sim[order];radius=radius[order]
        # Cosine similarity and temporal decay both affect strength.
        scores=np.exp(np.clip((sim-sim.max())/0.12,-40,0)) * np.exp(
            np.clip((p-t+self.exclusion+2)/self.horizon,-1,0))
        w=scores/np.maximum(scores.sum(),1e-30)
        vals=self.sequence[p+1]
        unique,inv=np.unique(vals,return_inverse=True)
        totals=np.bincount(inv,weights=w,minlength=len(unique))
        imax=int(totals.argmax());maxp=float(totals[imax]);entropy=float(
            -(totals*np.log(totals+1e-30)).sum()/np.log1p(len(unique)))
        target_p=float(w[vals==target].sum()) if target is not None else 0.
        has_exact=int(np.any(np.isin(self.sequence[p],qids)))
        features=np.asarray([
            1.,np.log1p(len(p))/5.,float(sim.max()),float((w*sim).sum()),
            maxp,entropy,float((p.max()-low)/max(1,t-low)),
            float(np.sum(w*w)),float((radius==0).sum()/len(radius)),
            float((w*radius).sum()),float(has_exact),
            float((w*(p-t+self.horizon)/self.horizon).sum()),
            float((vals==int(unique[imax])).mean()),float(sim.std()),
            float(len(unique)/max(1,len(p))),float(len(p)>=4)
        ],dtype=np.float32)
        return {'n':int(len(p)),'target_probability':target_p,
                'modal':int(unique[imax]),'peak':maxp,'entropy':entropy,
                'recent':float((p.max()-low)/max(1,t-low)),
                'max_cosine':float(sim.max()),'mean_cosine':float((w*sim).sum()),
                'effective_n':float(1/max((w*w).sum(),1e-30)),
                'lexical_overlap':float(has_exact),'hash_radius':int(radius.min()),
                'feature':features,'positions':p.astype(np.int32)}


class SemanticGate(nn.Module):
    """Zero-contribution, trainable R7 semantic memory route (1 new optimizer group).

    Gate coefficient is nonnegative by construction and bounded by 0.125.
    `open_fraction=0` gives *exact* inherited predictions at initialization;
    the boundary has an update gradient in torch.clamp. A negative learned
    fraction safely leaves the route off and cannot hurt the model.
    """
    def __init__(self):
        super().__init__()
        self.trust=nn.Sequential(nn.Linear(FEATURES,24),nn.SiLU(),nn.Linear(24,1))
        self.open_fraction=nn.Parameter(torch.zeros(()))

    def forward(self,prior_nll,features,prob,count):
        if not self.training and float(self.open_fraction.detach()) == 0.:
            return prior_nll,torch.zeros_like(prior_nll)
        old=torch.exp(-prior_nll)
        raw=self.trust(features).squeeze(-1)
        weight=.125*torch.clamp(self.open_fraction,0,1)*torch.sigmoid(raw)
        weight=weight*(count>0).to(weight.dtype)
        out=-torch.log(torch.clamp((1-weight)*old+weight*prob,min=1e-30))
        return out,weight