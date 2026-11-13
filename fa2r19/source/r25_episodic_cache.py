"""R25 document-local causal n-gram mixture. A single add-on to unified FA2.

No oracle targets, target tokens, or future text are used to build a query's
candidate distribution; only completed tokens at positions j < query_position.
This module is initialized to an exact identity (amplitude=0).
"""
from bisect import bisect_left
from collections import defaultdict
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

NGRAMS=(0,1,2,3)
class CausalNGramIndex:
    def __init__(self, tokens, *, memory=32768):
        self.tokens=np.asarray(tokens,dtype=np.int64)
        self.memory=memory
        self.locations=[]
        for n in (1,2,3):
            d=defaultdict(list)
            for j in range(n,len(self.tokens)):
                d[tuple(int(x) for x in self.tokens[j-n:j])].append(j)
            self.locations.append(d)

    def prior_locations(self, n, query_tokens, query_position, source_cutoff):
        # The original source document and a potentially different query text.
        end=min(source_cutoff,len(self.tokens))
        start=max(n,end-self.memory)
        if n==0:
            return range(max(start,end-4096),end)
        if query_position<n:
            return ()
        key=tuple(int(x) for x in query_tokens[query_position-n:query_position])
        loc=self.locations[n-1].get(key,())
        lo=bisect_left(loc,start);hi=bisect_left(loc,end)
        return loc[max(lo,hi-(512 if n==1 else 256)):hi]

    def distributions(self, query_tokens, query_start, count, vocab, *, source_cutoff=None):
        query_tokens=np.asarray(query_tokens,dtype=np.int64)
        if query_start<3 or query_start+count>len(query_tokens):raise ValueError('query range')
        same_document=source_cutoff is None
        if same_document:source_cutoff=query_start
        if source_cutoff<3 or source_cutoff>len(self.tokens):raise ValueError('source cutoff')
        out=np.zeros((count,4,vocab),np.float32)
        feats=np.zeros((count,4,3),np.float32)
        for t in range(count):
            q=query_start+t
            # Same-document source can reveal tokens strictly before q,
            # including earlier targets from the current chunk.
            cutoff=q if same_document else source_cutoff+t
            cutoff=min(cutoff,len(self.tokens))
            for n in NGRAMS:
                loc=self.prior_locations(n,query_tokens,q,cutoff)
                if not len(loc):continue
                js=np.asarray(loc,dtype=np.int64)
                ages=np.maximum(cutoff-js,1)
                weights=np.reciprocal(np.sqrt(1.+ages/512.)).astype(np.float32)
                counts=np.bincount(self.tokens[js],weights=weights,minlength=vocab).astype(np.float32)
                z=counts.sum()
                if z<=0:continue
                out[t,n,:]=counts/z
                feats[t,n,0]=math.log1p(len(js))
                feats[t,n,1]=counts.max()/z
                feats[t,n,2]=math.log1p(float(ages.min()))/math.log(32769)
        return out,feats

class AdaptiveNGramMixer(nn.Module):
    def __init__(self, width=40):
        super().__init__()
        self.amplitude=nn.Parameter(torch.tensor(0.))
        self.gate=nn.Sequential(nn.LayerNorm(18),nn.Linear(18,width),nn.GELU(),nn.Linear(width,5))
        with torch.no_grad():
            self.gate[-1].weight.zero_()
            self.gate[-1].bias.zero_()

    def forward(self,base_logits,candidates,features,*,return_info=False):
        if candidates.ndim!=3 or candidates.shape[1]!=4 or candidates.shape[2]!=base_logits.shape[-1]:
            raise ValueError('candidate dimensionality')
        probs=F.softmax(base_logits,dim=-1)
        present=candidates.sum(-1)>0
        evidence=(candidates*probs[:,None,:]).sum(-1)
        x=torch.cat((features.reshape(-1,12),evidence,
                     -(probs*torch.log(probs.clamp_min(1e-12))).sum(-1,keepdim=True)/10,
                     probs.max(-1,keepdim=True).values),dim=-1)
        if x.shape[-1]!=18:raise ValueError('bad feature shape')
        out=self.gate(x)
        source=out[:,:4].masked_fill(~present,-1e4)
        mix=F.softmax(source,dim=-1)*present.float()
        mix=mix/mix.sum(-1,keepdim=True).clamp_min(1e-12)
        cache=torch.einsum('tk,tkv->tv',mix,candidates)
        has=present.any(-1)
        amp=torch.clamp_min(torch.tanh(self.amplitude),0.)*.70
        alpha=amp*torch.sigmoid(out[:,4])*has.float()
        p=(1-alpha[:,None])*probs+alpha[:,None]*cache
        result=torch.log(p.clamp_min(1e-30))
        if return_info:
            return result,{'alpha':alpha,'source_weights':mix,'coverage':present.float().mean(0)}
        return result
