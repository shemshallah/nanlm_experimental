#!/usr/bin/env python3
"""Experimental R20 causal soft-admission copy path for the single FA2.1 model.

No new LM is created. Q/K projections, shared subword embeddings, temperature,
recency bias and decoder are inherited from R18. This replaces R18's structural
"exact two-token match or no candidates" admission with causal top-k selection
from ALL earlier memory slots, bounded to max_memory and processed in blocks.
A new scalar gain must be initialized at zero to preserve previous logits.
No semantic competence is claimed until paraphrase, hard-negative and held-out
language-model gates pass. This module does not save or promote any weights.
"""
import math
import torch
from torch.nn import functional as F


def soft_candidate_copy(net, base_logits, tokens, semantic_gain, *, max_memory=None, gap=None, block=64):
    """Causal soft admission. For query t, retrieve only j<=t-gap.

    A slot at j has key (tokens[j-2], tokens[j-1]) and value tokens[j].
    A query at t uses only tokens[t-2:t]. No token >=t is used for query/key
    selection or copied values. Topk is nondifferentiable in its *indices* but
    the chosen Q/K compatibility scores remain differentiable.

    Returns updated logits and metadata. O(N^2 d) compute but O(block*N)
    temporary scores; production length-65k use requires an ANN/coarse index.
    """
    if base_logits.ndim!=2 or tokens.ndim!=1 or base_logits.shape[0]!=tokens.shape[0]:
        raise ValueError('Expected aligned [tokens,vocabulary] logits and token ids')
    n,V=base_logits.shape
    memory=net.max_memory if max_memory is None else max_memory
    min_gap=net.min_gap if gap is None else gap
    if min_gap<3 or memory<min_gap or block<1:
        raise ValueError('Invalid gap, memory, or block size')
    if n<=min_gap+2:
        return base_logits,{'enabled_positions':0,'admission':'all causal prefix pairs','max_source_position':None}
    # Pair index p corresponds to the memory value at j=p+2.
    pair=torch.stack((tokens[:-2],tokens[1:-1]),dim=-1)
    key=net.encode(pair,False)
    query=net.encode(pair,True)
    keys_j=torch.arange(2,n,device=tokens.device)
    positions=torch.arange(2,n,device=tokens.device)
    out_delta=torch.zeros_like(base_logits)
    used=0;max_used=None;selected=0
    scale=net.log_temperature.exp().clamp(.1,50)
    for start in range(0,len(positions),block):
        tp=positions[start:start+block]
        allowed=(keys_j[None,:]<=tp[:,None]-min_gap)&(keys_j[None,:]>=tp[:,None]-memory)
        allowed &= (tp[:,None]>=min_gap+2)
        allowed &= ~((tokens[tp-1,None]==32)|(tokens[tp-1,None]==10)|(tokens[tp-1,None]==13))
        has_any=allowed.any(dim=-1)
        if not bool(has_any.any()):
            continue
        distances=(tp[:,None]-keys_j[None,:]).clamp_min(1)
        scores=(query[start:start+len(tp)] @ key.transpose(0,1))*scale
        scores=scores-net.recency*torch.log1p(distances.float())
        scores=scores.masked_fill(~allowed,-1e9)
        k=min(net.topk,scores.shape[-1])
        top_scores,top_ix=scores.topk(k,dim=-1)
        chosen_valid=allowed.gather(1,top_ix)
        top_scores=top_scores.masked_fill(~chosen_valid,-1e9)
        weights=F.softmax(top_scores,dim=-1)*chosen_valid.float()
        chosen_positions=keys_j[top_ix]
        value=tokens[chosen_positions].long()
        if bool(((value<0)|(value>=V)).any()):
            raise ValueError('Some retrieved values are outside decoder vocabulary')
        addition=torch.zeros((len(tp),V),device=base_logits.device,dtype=base_logits.dtype)
        addition=addition.scatter_add(1,value,weights.to(base_logits.dtype))
        out_delta=out_delta.index_add(0,tp,addition)
        used+=int(has_any.sum())
        selected+=int(chosen_valid.sum())
        mx=chosen_positions[chosen_valid].max().item()
        max_used=mx if max_used is None else max(max_used,mx)
    return base_logits+semantic_gain*out_delta,{
        'enabled_positions':used,'selected_slots':selected,
        'admission':'all causal prefix pairs (no exact-key guard)',
        'max_source_position':max_used,'max_memory':int(memory),'min_gap':int(min_gap)}
