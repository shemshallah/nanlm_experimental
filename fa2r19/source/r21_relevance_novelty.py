#!/usr/bin/env python3
"""FA2.2/R21: trainable relevance-and-novelty gate for inherited R18/R20 memory.

This is *one* PAL language model with an auxiliary, zero-residual memory gate.
The accepted R18 decoder, R18 memory and their optimizer histories are frozen
while testing the adapter. The gate is never promoted without held-out gains.

Query at t = encoded tokens[t-2],tokens[t-1]; source j <= t-min_gap.
No query feature uses the target tokens[t] or newer tokens. The decoder's
uncertainty is computed from its own pre-retrieval prediction at t.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

FEATURE_NAMES=('best_key_score','score_margin','value_agreement','novelty',
               'decoder_uncertainty','copy_probability','distance_prior','exact_prefix')

class RelevanceNoveltyGate(nn.Module):
    """Gated additive memory evidence; amplitude=0 gives exact logit identity."""
    def __init__(self,minimum_query_position=192):
        super().__init__()
        self.minimum_query_position=int(minimum_query_position)
        self.features=nn.Sequential(nn.LayerNorm(len(FEATURE_NAMES)),nn.Linear(len(FEATURE_NAMES),12),
                                    nn.Tanh(),nn.Linear(12,1))
        self.amplitude=nn.Parameter(torch.tensor(0.0))
        with torch.no_grad():
            self.features[-1].weight.zero_()
            self.features[-1].bias.zero_()

    def forward(self,base_logits,delta,features,*,query_position_offset=0):
        if (base_logits.shape!=delta.shape or features.shape!=(len(base_logits),len(FEATURE_NAMES))):
            raise ValueError('Gate shape mismatch')
        confidence=torch.sigmoid(self.features(features)).squeeze(-1)
        # Bounded signed amplitude: a learned negative sign is allowed, but is
        # explicitly measured. No target-conditioned weights are used at readout.
        gates=(2.0*torch.tanh(self.amplitude))*confidence
        # Allow memory to affect only sufficiently mature contexts. R20's
        # short-span regression motivated a fixed architectural restriction;
        # this cutoff is not inferred from the held-out test.
        query_pos=torch.arange(len(base_logits),device=base_logits.device) + int(query_position_offset)
        gates=gates*(query_pos>=self.minimum_query_position).to(gates.dtype)
        return base_logits+gates[:,None]*delta, gates

@torch.no_grad()
def causal_proposal(net,base_logits,tokens,*,memory=None,gap=None,block=48,content_only=True):
    """Frozen R18 Q/K read, gated with causal features derived at each t.

    Key and value are stored at position j; query t reads j<=t-gap.
    Returns proposal and 8 features, not learned activations or new weights.
    """
    n,v=base_logits.shape
    if tokens.ndim!=1 or len(tokens)!=n or tokens.min()<0 or tokens.max()>=v:
        raise ValueError('unaligned or out-of-vocab target stream')
    mem=net.max_memory if memory is None else int(memory)
    sep=net.min_gap if gap is None else int(gap)
    if sep<3 or mem<sep or block<1:raise ValueError('bad causal restrictions')
    delta=torch.zeros_like(base_logits)
    features=torch.zeros(n,len(FEATURE_NAMES),device=base_logits.device,dtype=base_logits.dtype)
    if n<=sep+2:return delta,features,{'eligible':0,'max_source_position':None}
    pair=torch.stack((tokens[:-2],tokens[1:-1]),dim=-1)
    keys=F.normalize(net.encode(pair,False),dim=-1)
    queries=F.normalize(net.encode(pair,True),dim=-1)
    key_j=torch.arange(2,n,device=tokens.device)
    valid_value=(tokens[key_j]!=32)&(tokens[key_j]!=10)&(tokens[key_j]!=13) if content_only else torch.ones_like(key_j,dtype=torch.bool)
    scale=net.log_temperature.exp().clamp(.1,50)
    available=0;max_source=None
    for st in range(0,n-2,block):
        pos=torch.arange(st+2,min(n,st+2+block),device=tokens.device)
        allowed=(key_j[None,:]<=pos[:,None]-sep)&(key_j[None,:]>=pos[:,None]-mem)&valid_value[None,:]
        allowed &= ((tokens[pos-1,None]!=32)&(tokens[pos-1,None]!=10)&(tokens[pos-1,None]!=13))
        good=allowed.any(1)
        if not bool(good.any()):continue
        dist=(pos[:,None]-key_j[None,:]).clamp_min(1)
        score=queries[st:st+len(pos)] @ keys.T * scale - net.recency*torch.log1p(dist.float())
        score=score.masked_fill(~allowed,-1e8)
        top,idx=score.topk(min(net.topk,len(key_j)),dim=-1)
        valid=allowed.gather(1,idx)
        top=top.masked_fill(~valid,-1e8)
        weights=F.softmax(top,dim=-1)*valid.float()
        chosen=key_j[idx]
        copied=tokens[chosen].long()
        delta[pos]=delta[pos].scatter_add(1,copied,weights.to(delta.dtype))
        prob=F.softmax(base_logits[pos],dim=-1)
        copy_mass=(weights*prob.gather(1,copied)).sum(1)
        top_value_prob=prob.gather(1,copied[:,0:1]).squeeze(1)
        agree=(weights*(copied==copied[:,0:1]).float()).sum(1)
        gap_score=(top[:,0]-top[:,1]).clamp(-5,5) if top.shape[1]>1 else top[:,0]*0
        exact=((tokens[chosen-2]==tokens[pos[:,None]-2]) &
               (tokens[chosen-1]==tokens[pos[:,None]-1])).float()
        p_entropy=-(prob*prob.clamp_min(1e-12).log()).sum(1)/math.log(v)
        proposals=torch.stack((top[:,0].clamp(-10,10),gap_score,agree,1-copy_mass,
             p_entropy,top_value_prob,torch.log1p((pos-chosen[:,0]).float())/math.log1p(mem),
             (weights*exact).sum(1)),dim=1)
        features[pos]=torch.where(good[:,None],proposals.to(features.dtype),torch.zeros_like(proposals))
        available+=int(good.sum())
        local_max=int(chosen[valid].max())
        max_source=local_max if max_source is None else max(max_source,local_max)
    return delta,features,{'eligible':available,'max_source_position':max_source,
        'min_gap':sep,'max_memory':mem,'content_only':bool(content_only)}
