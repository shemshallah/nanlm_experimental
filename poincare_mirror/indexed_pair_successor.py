"""R61: exact causal CPU index for FA2 mirror-pair successor memory.

This is an optional execution engine, not a separately trained predictor.
It reproduces the R54/R59/R60 window semantics while indexing old pair and
point addresses in O(horizon * edges) *stored records*. No target leakage.
"""
from collections import defaultdict, deque

import torch


def retrieve_indexed_pair_successors(pair_ids, observed, horizon, vocab,
                                     query_map=None, value_control="exact",
                                     key_mode="pair", query_byte_map=None):
    """CPU-only B,T,V histograms, B,T confidence coverage, B,T hit positions."""
    if pair_ids.device.type != "cpu" or observed.device.type != "cpu":
        raise ValueError("Indexed causal lookup is CPU-only; use vectorized for GPU")
    if key_mode not in ("pair","pair_byte","pair_bigram","head_byte","hierarchical"):
        raise ValueError("Unsupported indexed key mode")
    batch, length, edges = pair_ids.shape
    pair_rows=pair_ids.tolist()
    seen_rows=observed.tolist()
    remap_pair=query_map.tolist() if query_map is not None else None
    remap_byte=query_byte_map.tolist() if query_byte_map is not None else None
    vertices=(torch.triu_indices(256,256,1).t().tolist()
              if key_mode in ("head_byte","hierarchical") else None)

    all_hist,all_coverage,all_positions=[],[],[]
    for b in range(batch):
        seen=seen_rows[b]
        addresses=pair_rows[b]
        exact_index=defaultdict(deque)   # compound keys -> (time, edge)
        head_index=defaultdict(deque)    # (head, observed byte) -> (time, edge)
        scheduled=deque()              # bounded sliding causal horizon
        hist_rows,cover_rows,pos_rows=[],[],[]

        def make_key(address,at):
            if key_mode=="pair":
                return (address,)
            if key_mode=="pair_bigram":
                return (address,seen[at],seen[at-1]) if at>0 else None
            return (address,seen[at])

        for t in range(length):
            first=max(0,t-horizon)
            # Evict every historical key whose successor is outside the
            # admissible [first,t) causal window. Memory stays O(H*E).
            while scheduled and scheduled[0][0] < first:
                _, expired=scheduled.popleft()
                for exact_key,heads,entry in expired:
                    if exact_key is not None:
                        old=exact_index[exact_key].popleft()
                        if old != entry:
                            raise AssertionError("Out-of-order exact-key expiration")
                        if not exact_index[exact_key]:
                            del exact_index[exact_key]
                    for head_key in heads:
                        old=head_index[head_key].popleft()
                        if old != entry:
                            raise AssertionError("Out-of-order endpoint expiration")
                        if not head_index[head_key]:
                            del head_index[head_key]

            hist=[0.0]*vocab
            seen_times=set()
            matched_edges=[]
            for q_address in addresses[t]:
                lookup_address=(remap_pair[q_address] if remap_pair is not None
                                else q_address)
                q_byte=(remap_byte[seen[t]] if remap_byte is not None
                        else seen[t])
                if key_mode=="pair":
                    match=list(exact_index.get((lookup_address,),()))
                    confidence=1.0
                elif key_mode=="pair_bigram":
                    match=list(exact_index.get((lookup_address,q_byte,seen[t-1]),()))
                    confidence=1.0
                else:
                    exact=list(exact_index.get((lookup_address,q_byte),()))
                    if key_mode=="pair_byte":
                        match,confidence=exact,1.0
                    else:
                        left,right=vertices[lookup_address]
                        shared=set(head_index.get((left,q_byte),()))
                        shared.update(head_index.get((right,q_byte),()))
                        if key_mode=="hierarchical" and exact:
                            match,confidence=exact,1.0
                        else:
                            match,confidence=list(shared),0.25
                if match:
                    matched_edges.append((match,confidence))
                    for u,_ in match:
                        seen_times.add(u)
            if matched_edges:
                for matches,_ in matched_edges:
                    contribution=1.0/(len(matched_edges)*len(matches))
                    for u,_ in matches:
                        # Matches at u<t use observed x[u+1]<=x[t].
                        # Rotation changes only values, not matched support.
                        at=(t if u==first else u) if value_control=="rotated" else u+1
                        hist[seen[at]]+=contribution
                cover=sum(confidence for _,confidence in matched_edges)/edges
            else:
                cover=0.0
            hist_rows.append(hist)
            cover_rows.append(cover)
            pos_rows.append(float(len(seen_times)))

            # Register this *input* only after serving query t, so its
            # successor cannot be read before the next observed byte exists.
            events=[]
            for e,address in enumerate(addresses[t]):
                entry=(t,e)
                exact_key=make_key(address,t)
                if exact_key is not None:
                    exact_index[exact_key].append(entry)
                heads=[]
                if vertices is not None:
                    left,right=vertices[address]
                    heads=[(left,seen[t]),(right,seen[t])]
                    for head_key in heads:
                        head_index[head_key].append(entry)
                events.append((exact_key,heads,entry))
            scheduled.append((t,events))
        all_hist.append(hist_rows)
        all_coverage.append(cover_rows)
        all_positions.append(pos_rows)

    return (torch.tensor(all_hist,dtype=torch.float32,device=pair_ids.device),
            torch.tensor(all_coverage,dtype=torch.float32,device=pair_ids.device),
            torch.tensor(all_positions,dtype=torch.float32,device=pair_ids.device))
