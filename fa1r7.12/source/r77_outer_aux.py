#!/usr/bin/env python3
"""Auxiliary causal associative-recall objective for existing R7.4 rank64 matrix.

No extra model or optimizer. The query and correct value are drawn from the
already-observed older FIFO; targets are NOT from the prediction position.
All candidate targets are historical, ensuring no present/future leakage.
"""
import torch
from torch.nn import functional as F

def choose_recall(history,nneg=24,generator=None):
    """Select a causal old (key,next-word) positive and old negative words."""
    b,steps=history.shape
    valid=(history[:,:-1]>1)&(history[:,1:]>1)
    # Uniformly select among eligible old bigrams (BOS is not a token event).
    scores=torch.rand(valid.shape,device=history.device,generator=generator).masked_fill(~valid,-1)
    idx=scores.argmax(1)
    active=valid.any(1)
    queries=history.gather(1,idx[:,None]).squeeze(1)
    positives=history.gather(1,(idx+1)[:,None]).squeeze(1)
    nvalid=history>1
    negative_scores=torch.rand(history.shape,device=history.device,generator=generator).masked_fill(~nvalid,-1)
    neg_idx=negative_scores.topk(min(nneg,steps),dim=1).indices
    negatives=history.gather(1,neg_idx)
    # No trivially matching 'negative' equals the correct positive; otherwise
    # the contrastive task has irreducibly ambiguous targets.
    mask=(negatives>1)&(negatives!=positives[:,None])
    return queries,positives,negatives,active,mask

def contrastive_loss(model,state,history,negatives=24,temperature=0.15,generator=None,swapped=False):
    q,y,neg,active,mask=choose_recall(history,negatives,generator)
    if not bool(active.any()):return state.sum()*0
    E=model.trunk.E.detach()
    qemb=F.embedding(q.long(),E)
    if swapped:state=state.roll(1,0)
    read=model.r74.outer.read_rails(state,qemb)
    candidates=torch.cat((y[:,None],neg),dim=1).long()
    vectors=F.normalize(F.embedding(candidates,E),dim=-1)
    scores=(F.normalize(read,dim=-1)[:,None,:]*vectors).sum(-1)/temperature
    scores[:,1:]=scores[:,1:].masked_fill(~mask,-1e4)
    return F.cross_entropy(scores[active],torch.zeros(int(active.sum()),device=scores.device,dtype=torch.long))
