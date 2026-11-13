"""FA2 R26: causal high-order phrase retrieval at 8K and 128K scales.

This is an *additional trainable readout of the SAME R25 PAL network*, not a
replacement model. Each candidate key is 4, 5, or 6 prior subword tokens.
Training stores target likelihoods rather than huge dense [time,vocab] arrays.
"""
from bisect import bisect_left
from collections import defaultdict
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ORDERS=(4,5,6)
HORIZONS=(8192,131072)
CHANNELS=tuple((n,h) for h in HORIZONS for n in ORDERS)

class CausalHighOrderIndex:
    def __init__(self, tokens):
        self.tokens=np.asarray(tokens,dtype=np.int32)
        self.index={}
        for n in ORDERS:
            d=defaultdict(list)
            ts=self.tokens
            # Key includes exactly n tokens before target j; no target or suffix.
            for j in range(n,len(ts)):
                d[tuple(ts[j-n:j])].append(j)
            self.index[n]=d

    def positions(self, query, q, n, horizon, source_cutoff):
        if q<n:return ()
        positions=self.index[n].get(tuple(query[q-n:q]),())
        lo=bisect_left(positions,max(n,source_cutoff-horizon))
        hi=bisect_left(positions,source_cutoff)
        return positions[max(lo,hi-128):hi]

    def features(self, query, start, target_count, base_prob, targets, *, source_cutoff=None):
        """Outputs (cache_p_target:[T,6], features:[T,6,4]).
        base_prob is the R25 probability distribution for each predicted token;
        never use the current target in gate features, only its loss label.
        """
        query=np.asarray(query,dtype=np.int32)
        targets=np.asarray(targets,dtype=np.int64)
        base=np.asarray(base_prob,dtype=np.float32)
        if base.shape[0]!=target_count or base.shape[1]<=int(max(self.tokens.max(),targets.max())):
            raise ValueError('vocabulary mismatch')
        if start<6 or start+target_count>len(query):raise ValueError('query bounds')
        if source_cutoff is not None and not(6<=source_cutoff<=len(self.tokens)):
            raise ValueError('donor cutoff')
        p_y=np.zeros((target_count,len(CHANNELS)),dtype=np.float32)
        feats=np.zeros((target_count,len(CHANNELS),4),dtype=np.float32)
        for t in range(target_count):
            q=start+t
            cutoff=q if source_cutoff is None else min(len(self.tokens),source_cutoff+t)
            for k,(n,horizon) in enumerate(CHANNELS):
                js=self.positions(query,q,n,horizon,cutoff)
                if not js:continue
                js=np.asarray(js,dtype=np.int32)
                ages=np.maximum(1,cutoff-js)
                weights=1/np.sqrt(1+ages/512)
                val=self.tokens[js]
                uniq,inv=np.unique(val,return_inverse=True)
                masses=np.bincount(inv,weights=weights,minlength=len(uniq)).astype(np.float32)
                total=masses.sum()
                if total<=0:continue
                masses/=total
                p_y[t,k]=float(masses[uniq==targets[t]].sum())
                feats[t,k,0]=math.log1p(len(js))/6
                feats[t,k,1]=float(masses.max())
                feats[t,k,2]=math.log1p(float(ages.min()))/math.log1p(horizon)
                feats[t,k,3]=float((masses*base[t,uniq]).sum())
        return p_y,feats

    def full_distributions(self, query, start, target_count, vocab, *, source_cutoff=None):
        """Causal vocabulary distributions for deployment/generation.
        Allocates only a caller-requested query chunk rather than document size.
        """
        query=np.asarray(query,dtype=np.int32)
        if start<6 or start+target_count>len(query):raise ValueError('query bounds')
        if source_cutoff is not None and not(6<=source_cutoff<=len(self.tokens)):
            raise ValueError('donor cutoff')
        out=np.zeros((target_count,len(CHANNELS),vocab),dtype=np.float32)
        for t in range(target_count):
            q=start+t
            cutoff=q if source_cutoff is None else min(len(self.tokens),source_cutoff+t)
            for k,(n,horizon) in enumerate(CHANNELS):
                js=self.positions(query,q,n,horizon,cutoff)
                if not js:continue
                js=np.asarray(js,dtype=np.int32)
                ages=np.maximum(1,cutoff-js)
                weights=1/np.sqrt(1+ages/512)
                counts=np.bincount(self.tokens[js],weights=weights,minlength=vocab)
                total=counts.sum()
                if total>0:out[t,k]=counts/total
        return out

class AdaptiveSpanMixer(nn.Module):
    """Residual mixture with one scalar and 6 channel scores.
    Initialized at identity; old R25 probabilities are bitwise retained.
    """
    def __init__(self,width=48):
        super().__init__()
        self.amplitude=nn.Parameter(torch.tensor(0.))
        self.gate=nn.Sequential(nn.LayerNorm(26),nn.Linear(26,width),nn.GELU(),nn.Linear(width,7))
        with torch.no_grad():
            self.gate[-1].weight.zero_();self.gate[-1].bias.zero_()

    def _blend(self,features):
        if features.ndim!=3 or features.shape[1:]!=(6,4):raise ValueError('features shape')
        # features are observed before each query; not dependent on ground truth
        evidence=features.reshape(len(features),24)
        present=features[:,:,0]>0
        # Aggregate fraction of matching higher-order context, age and confidence.
        pooled=torch.stack((present.float().mean(-1),
                            features[:,:,1].amax(-1)),dim=-1)
        scores=self.gate(torch.cat((evidence,pooled),dim=-1))
        w=F.softmax(scores[:,:6].masked_fill(~present,-1e4),dim=-1)*present.float()
        w=w/w.sum(-1,keepdim=True).clamp_min(1e-10)
        alpha=(.60*torch.clamp_min(torch.tanh(self.amplitude),0.)*
               torch.sigmoid(scores[:,6])*present.any(-1).float())
        return w,alpha

    def target_logprob(self,base_target_logprob,target_cache_prob,features):
        w,alpha=self._blend(features)
        cache_y=(w*target_cache_prob).sum(-1)
        p_y=(1-alpha)*torch.exp(base_target_logprob)+alpha*cache_y
        return torch.log(p_y.clamp_min(1e-30))

    def forward(self,base_logprobs,candidates,features):
        if candidates.ndim!=3 or candidates.shape[1:]!=(6,base_logprobs.shape[-1]):
            raise ValueError('candidate shape')
        w,alpha=self._blend(features)
        cache=torch.einsum('tn,tnv->tv',w,candidates)
        p=(1-alpha[:,None])*base_logprobs.exp()+alpha[:,None]*cache
        return torch.log(p.clamp_min(1e-30))
