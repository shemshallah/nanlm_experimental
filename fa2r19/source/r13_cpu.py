#!/usr/bin/env python3
"""PAL R13 CPU fast path. Reuses the *same* R12 parameters and checkpoint.

Equivalent R12 recurrence, but each historical K/V is projected only once;
per-token cross-entropies are fused into a single batched operation.
No new weights, speculative target features, or additional prediction branch.
"""
import math
import torch
from torch.nn import functional as F


def run_window_fast(model, bridge, adapter, h, targets, tags, training=False,
                    disable_attention=False, disable_ffn=False, inspect_prefix=0, return_logits=False):
    B,T=targets.shape
    previous=torch.full((B,),model.memory.start_piece,device=targets.device,dtype=torch.long)
    state=h[:,0];last=tags[:,0];retain=bridge.retain_logit.sigmoid()
    khist=[];vhist=[];logits_seq=[];logits_prefix=[]
    for t in range(T):
        valid=targets[:,t]!=-100
        ix=tags[:,t].clamp(0,h.shape[1]-1)
        next_h=h.gather(1,ix[:,None,None].expand(-1,1,h.shape[-1]))[:,0]
        changed=(ix!=last)&valid
        state=torch.where(changed[:,None],retain*state+(1-retain)*bridge.context_gain*next_h,state)
        state=model.memory.subword_cell(model.memory.piece_emb(previous),state)
        out=state
        if adapter is not None:
            if khist and not disable_attention:
                past_k=torch.stack(khist[-adapter.capacity:],dim=1)
                past_v=torch.stack(vhist[-adapter.capacity:],dim=1)
                scores=(adapter.query(state).unsqueeze(1)*past_k).sum(-1)/math.sqrt(state.shape[-1])
                k=min(adapter.topk,scores.shape[1])
                inds=scores.topk(k,dim=1).indices
                selected=past_v.gather(1,inds[:,:,None].expand(-1,-1,state.shape[-1]))
                weights=F.softmax(scores.gather(1,inds),dim=-1)
                recalled=(selected*weights[:,:,None]).sum(1)
                out=out+adapter.attn_gain*adapter.attn_out(recalled)
            if not disable_ffn:
                out=out+adapter.ffn_gain*adapter.ffn(state)
            if not disable_attention:
                khist.append(adapter.key(state));vhist.append(adapter.value(state))
        logits=model.memory.subword_out(out)
        logits_seq.append(logits)
        if inspect_prefix and t<inspect_prefix: logits_prefix.append(logits.detach().clone())
        previous=torch.where(valid,targets[:,t],torch.zeros_like(previous))
        last=torch.where(valid,ix,last)
    # One CE kernel instead of a per-token kernel + many tiny scalar additions.
    logits=torch.stack(logits_seq,dim=1)
    if return_logits:return logits
    flat=logits.reshape(B*T,-1)
    labels=targets.reshape(-1)
    valid=labels!=-100
    total_n=int(valid.sum())
    loss=F.cross_entropy(flat,labels,ignore_index=-100,reduction='sum')
    if training:return loss/max(total_n,1),total_n
    space=valid&(labels==32);other=valid&(labels!=32)
    other_n=int(other.sum());space_n=int(space.sum())
    preds=flat.argmax(dim=-1)
    return {'total_loss':float(loss),'num_tokens':total_n,
            'other_loss':float(F.cross_entropy(flat[other],labels[other],reduction='sum')) if other_n else 0.,
            'nonspace_tokens':other_n,'space_tokens':space_n,
            'other_hits':int(((preds==labels)&other).sum()),
            'space_hits':int(((preds==32)&space).sum()),
            'prefix_logits':logits_prefix}