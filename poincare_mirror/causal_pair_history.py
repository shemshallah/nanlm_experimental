"""Exact C(256,2)-addressed, strictly-past episodic retrieval for FA2.

No parameters or persistent state: training can differentiate into the existing
pair representations and old optimizer history. A zero-gain caller keeps every
inherited checkpoint logit bit-for-bit identical.
"""
import torch


def retrieve_past_pair_values(pair_ids, values, horizon=16, query_map=None):
    """Return mean of strictly earlier *matching-address* edge values.

    pair_ids: (batch,time,28) exact IDs in [0,32640).
    values:   (batch,time,28,rank), already computed from causal inputs.
    horizon:  positive bounded number of prior tokens. Never includes t.
    query_map: optional 32640-way permutation applied ONLY on read, not write.
               This deliberately breaks key agreement for a negative control.

    Every current edge is a query; edges with at least one prior hit contribute
    equally to the pooled read. No hit => exactly zero (including at t=0).
    """
    if pair_ids.ndim != 3 or values.ndim != 4 or pair_ids.shape != values.shape[:3]:
        raise ValueError('Expected B,T,E addresses matching B,T,E,R values')
    if pair_ids.dtype != torch.long:
        raise ValueError('Exact pair IDs must use torch.long')
    if horizon < 1:
        raise ValueError('A positive temporal horizon is required')
    if query_map is not None and (query_map.shape != (32640,) or
                                  query_map.dtype != torch.long):
        raise ValueError('Query-only permutation must have 32640 integer entries')
    b, length, edges = pair_ids.shape
    rank = values.shape[-1]
    query_map_on_device = (query_map.to(device=pair_ids.device)
                           if query_map is not None else None)
    output = [values.new_zeros((b, rank))]
    count_hits = [values.new_zeros((b,))]
    for t in range(1, length):
        first = max(0,t-horizon)
        prior_ids = pair_ids[:, first:t, :]
        prior_values = values[:, first:t, :, :]
        query = pair_ids[:,t,:]
        if query_map_on_device is not None:
            query = query_map_on_device[query]
        match = query[:,:,None,None] == prior_ids[:,None,:,:]
        counts = match.sum(dim=(2,3))  # B,E current pair slots
        pair_sum = torch.einsum('bqhe,bher->bqr', match.to(values.dtype), prior_values)
        per_query = pair_sum / counts.clamp_min(1).unsqueeze(-1)
        has_hit = counts > 0
        read = (per_query*has_hit.unsqueeze(-1)).sum(1) / (
            has_hit.sum(1).clamp_min(1).unsqueeze(-1))
        output.append(read)
        count_hits.append(has_hit.sum(1).to(values.dtype))
    return torch.stack(output,dim=1), torch.stack(count_hits,dim=1)
