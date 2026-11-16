"""R54: strictly causal, exact-pair successor-byte memory for the unified FA2 LM.

At position t an address match at u<t contributes only the already observed
input byte at u+1<=t. No target label or future token enters the memory.
"""
import math

import torch
from torch import nn


def retrieve_pair_successors(pair_ids, observed, horizon=32, vocab=257,
                             query_map=None, value_control="exact"):
    """Return B,T,V next-byte histograms, B,T edge coverage, B,T hit positions.

    The wrong-value control retains *identical address matches and coverage*,
    rotating only the already seen successor values inside each causal prefix.
    """
    if pair_ids.ndim != 3 or observed.ndim != 2 or pair_ids.shape[:2] != observed.shape:
        raise ValueError("Need matching B,T,E pair IDs and B,T observed bytes")
    if pair_ids.dtype != torch.long or observed.dtype != torch.long:
        raise ValueError("Pair addresses and observed bytes must use torch.long")
    if pair_ids.device != observed.device:
        raise ValueError("Addresses and bytes must use the same device")
    if horizon < 1 or vocab < 2 or value_control not in ("exact", "rotated"):
        raise ValueError("Invalid memory horizon, vocabulary or value control")
    if query_map is not None and (query_map.shape != (32640,) or
                                  query_map.dtype != torch.long):
        raise ValueError("Query map must be 32640 integer entries")
    if torch.any((pair_ids < 0) | (pair_ids >= 32640)) or torch.any(
            (observed < 0) | (observed >= vocab)):
        raise ValueError("Addresses or observed bytes out of range")
    batch, length, edges = pair_ids.shape
    query_map = query_map.to(pair_ids.device) if query_map is not None else None
    histograms = [torch.zeros(batch, vocab, device=pair_ids.device)]
    coverage = [torch.zeros(batch, device=pair_ids.device)]
    positions = [torch.zeros(batch, device=pair_ids.device)]
    for t in range(1, length):
        first = max(0, t-horizon)
        keys = pair_ids[:,first:t,:]
        query = pair_ids[:,t,:]
        if query_map is not None:
            query = query_map[query]  # only queries remapped, never stored keys
        match = query[:,:,None,None].eq(keys[:,None,:,:])  # B,E,H,E
        hit_counts = match.sum((2,3))                       # B,E
        active = hit_counts > 0
        by_time = match.sum(-1).float() / hit_counts.clamp_min(1)[:,:,None]
        historical_weight = by_time.sum(1) / active.sum(1).clamp_min(1)[:,None]
        # For u in [first,t), the successor x[u+1] is present among inputs
        # x[0:t+1]. The target x[t+1] is never read, even when u=t-1.
        successors = observed[:,first+1:t+1]
        if value_control == "rotated":
            successors = successors.roll(1,dims=1)  # same observed multiset
        hist = torch.zeros(batch,vocab,device=pair_ids.device)
        hist.scatter_add_(1,successors,historical_weight)
        histograms.append(hist)
        coverage.append(active.float().mean(1))
        positions.append(match.any(dim=(1,3)).float().sum(1))
    return (torch.stack(histograms,1),
            torch.stack(coverage,1),
            torch.stack(positions,1))


class PairSuccessorTrust(nn.Module):
    """Zero-initialized four-parameter trust gate; inherited byte head intact."""
    def __init__(self, horizon=32, smoothing=0.10):
        super().__init__()
        if horizon < 1 or not 0.0 < smoothing < 1.0:
            raise ValueError("Invalid local memory settings")
        self.weight = nn.Parameter(torch.zeros(4))
        self.horizon = horizon
        self.smoothing = smoothing
        self.query_map = None
        self.value_control = "exact"

    def forward(self, logits, pairs, observed):
        if logits.ndim != 3 or logits.shape[:2] != pairs.shape[:2]:
            raise ValueError("Logits and geometric address windows must align")
        vocab = logits.size(-1)
        with torch.no_grad():
            copy, coverage, positions = retrieve_pair_successors(
                pairs,observed,self.horizon,vocab,self.query_map,self.value_control)
            previous = logits.detach().float().softmax(-1)
            log_evidence = (copy*(1-self.smoothing)+self.smoothing/vocab).log()
            log_evidence = log_evidence + math.log(vocab)
            entropy = -(copy*copy.clamp_min(1e-12).log()).sum(-1)/math.log(vocab)
            agreement = (copy*previous).sum(-1)
            features = torch.stack((
                coverage, torch.log1p(positions)/math.log1p(self.horizon),
                1-entropy, agreement),-1)
        trust = torch.tanh((features.to(self.weight.dtype)*self.weight).sum(-1))
        strength = trust * coverage
        corrected = logits + (strength.unsqueeze(-1)*log_evidence).to(logits.dtype)
        stats = {"coverage":coverage.detach(),
                 "matched_positions":positions.detach(),
                 "trust":strength.detach()}
        return corrected,stats
