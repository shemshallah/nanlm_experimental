#!/usr/bin/env python3
"""FA2.3 R22: causal document-persistent, multi-timescale R18 memory read.

Model preservation: R18 logits + frozen R21 gate remain unchanged when R22's
three *new* residual amplitudes are zero. Sources are ONLY preceding tokens of
THE SAME Gutenberg book, excluding the current evaluation window and all later
material. This is retrieval capacity, not proof of semantic understanding.

Dense: <= 2,048-token old context; middle: 2,048..8,192 (stride 8);
far: 8,192..32,768 (stride 32). Missed exact two-byte prefixes are rescued
by a bounded full-prefix hash index. O(window * sampled memory * d) scoring,
NOT end-to-end transformer attention and not yet 32K recurrent training.
"""
from collections import defaultdict
import math
import torch
from torch import nn
from torch.nn import functional as F
from r21_relevance_novelty import RelevanceNoveltyGate, FEATURE_NAMES

BANDS = ('recent', 'middle', 'far')
BOUNDS = (65, 2048, 8192, 32768)
STRIDES = (1, 8, 32)


class R22Residual(nn.Module):
    """Three separately calibrated gated reads; exact R21 identity at init."""
    def __init__(self, minimum_query_position=192):
        super().__init__()
        self.branches = nn.ModuleDict({name: RelevanceNoveltyGate(minimum_query_position)
                                       for name in BANDS})

    def forward(self, r21_logits, proposals, features, *, query_position_offset=0):
        result = r21_logits
        gate_values = {}
        for name in BANDS:
            result, weights = self.branches[name](result, proposals[name], features[name],
                                                  query_position_offset=query_position_offset)
            gate_values[name] = weights
        return result, gate_values


@torch.no_grad()
def hierarchical_proposals(net, logits, book_tokens, window_start, *,
                            max_distance=32768, gap=65, topk=8, rescue=3):
    """Return causal candidate distributions and 8 R21-compatible features.

    logits are [T,V] for the local window at token offset window_start.
    book_tokens is whole same-book token sequence, used ONLY through index
    ``window_start + t - 1`` for a query t; candidate *values* must be <
    window_start and at least gap positions behind the query. No targets in
    the window are copied from the current/future positions.
    """
    n, vocab = logits.shape
    tok = torch.as_tensor(book_tokens, dtype=torch.long, device=logits.device)
    if logits.ndim != 2 or tok.ndim != 1 or n < 1 or not (0 <= window_start <= len(tok) - n):
        raise ValueError('bad window alignment or dimensions')
    if gap < 3 or max_distance < gap or topk < 1 or rescue < 0:
        raise ValueError('bad causal parameters')
    if torch.any(tok < 0) or torch.any(tok >= vocab):
        raise ValueError('invalid token IDs for inherited byte decoder')
    zeros = {name: torch.zeros_like(logits) for name in BANDS}
    ft = {name: logits.new_zeros((n, len(FEATURE_NAMES))) for name in BANDS}
    statistics = {name: {'eligible': 0, 'read_positions': 0, 'sampled_positions': 0,
                          'rescue_positions': 0, 'maximum_read_distance': 0}
                  for name in BANDS}
    lower = max(2, window_start - max_distance)
    # Only older-before-window memory may become candidate values.
    if window_start <= 2 or window_start <= lower:
        return zeros, ft, statistics
    source = torch.arange(lower, window_start, device=tok.device)
    before_window = (window_start - source)
    # Hash full preceding prefix, not just downsampled slots. Bounded rescue
    # brings exact-prefix matches back into the candidate set across all bands.
    prefix_map = defaultdict(list)
    for j in range(lower, window_start):
        if int(tok[j]) not in (10, 13, 32):
            prefix_map[(int(tok[j-2]), int(tok[j-1]))].append(j)
    queries_at = torch.arange(window_start, window_start + n, device=tok.device)
    query_pairs = torch.stack((tok[queries_at - 2], tok[queries_at - 1]), -1)
    rescue_set = set()
    for t in range(n):
        q = (int(query_pairs[t, 0]), int(query_pairs[t, 1]))
        if q[1] in (10, 13, 32):
            continue
        occur = prefix_map.get(q, ())
        for a, b in zip(BOUNDS[:-1], BOUNDS[1:]):
            seen = 0
            for j in reversed(occur):
                d = window_start + t - j
                if d >= b: continue
                if d >= max(gap, a):
                    rescue_set.add(j)
                    seen += 1
                    if seen >= rescue: break
    # Coarse compression fixed before seeing the query target; admissibility
    # is checked again per query at its exact distance.
    pick = ((before_window <= 2048) |
            ((before_window <= 8192) & (source % 8 == 0)) |
            (source % 32 == 0))
    indices = set(source[pick].tolist())
    sampled_indices = set(indices)
    sample_num = len(indices)
    indices.update(rescue_set)
    if not indices:
        return zeros, ft, statistics
    positions = torch.tensor(sorted(indices), device=tok.device, dtype=torch.long)
    key_pairs = torch.stack((tok[positions - 2], tok[positions - 1]), -1)
    # Critical causal guard: a slot rescued by some *future* query is NOT
    # automatically visible to earlier queries. That would leak the future
    # query distribution into the earlier shortlist. Only sampled slots or
    # exact matches to THIS query may enter its scored candidates.
    sampled = torch.tensor([int(j) in sampled_indices for j in positions.tolist()],
                           device=tok.device,dtype=torch.bool)
    shortlisted = sampled[None, :] | ((key_pairs[:, 0][None, :] == query_pairs[:, 0:1]) &
                                     (key_pairs[:, 1][None, :] == query_pairs[:, 1:2]))
    keys = net.encode(key_pairs, False)
    queries = net.encode(query_pairs, True)
    differences = queries_at[:, None] - positions[None, :]
    score = queries @ keys.T * net.log_temperature.exp().clamp(.1, 50)
    score = score - net.recency * torch.log1p(differences.clamp_min(1).float())
    score = score.masked_fill((tok[queries_at-1] == 32)[:, None] |
                              (tok[queries_at-1] == 10)[:, None] |
                              (tok[queries_at-1] == 13)[:, None], -1e9)
    p = F.softmax(logits, dim=-1)
    entropy = -(p * p.clamp_min(1e-12).log()).sum(-1) / math.log(vocab)
    for name, start, stop in zip(BANDS, BOUNDS[:-1], BOUNDS[1:]):
        valid = (shortlisted & (differences >= max(start, gap)) & (differences < stop) &
                 (tok[positions] != 32)[None, :] &
                 (tok[positions] != 10)[None, :] &
                 (tok[positions] != 13)[None, :] &
                 (tok[queries_at - 1] != 32)[:, None] &
                 (tok[queries_at - 1] != 10)[:, None] &
                 (tok[queries_at - 1] != 13)[:, None])
        allowed_rows = valid.any(1)
        if not bool(allowed_rows.any()):
            continue
        s = score.masked_fill(~valid, -1e9)
        k = min(topk, s.shape[1])
        vals, ix = s.topk(k, dim=-1)
        chosen_ok = valid.gather(1, ix)
        vals = vals.masked_fill(~chosen_ok, -1e9)
        weights = F.softmax(vals, -1) * chosen_ok.float()
        chosen_j = positions[ix]
        copied = tok[chosen_j]
        zeros[name].scatter_add_(1, copied, weights.to(logits.dtype))
        mass = (weights * p.gather(1, copied)).sum(-1)
        agreement = (weights * (copied == copied[:, :1]).float()).sum(-1)
        exact = ((tok[chosen_j-2] == query_pairs[:, None, 0]) &
                 (tok[chosen_j-1] == query_pairs[:, None, 1])).float()
        best_score = torch.where(allowed_rows, vals[:, 0], torch.zeros_like(vals[:, 0]))
        margin = (vals[:, 0] - vals[:, 1]).clamp(-5, 5) if k > 1 else vals[:, 0] * 0
        features = torch.stack((best_score.clamp(-10, 10), margin, agreement,
                                1 - mass, entropy, p.gather(1, copied[:, 0:1]).squeeze(1),
                                torch.log1p((queries_at - chosen_j[:, 0]).float()) / math.log1p(max_distance),
                                (weights * exact).sum(1)), -1)
        ft[name] = torch.where(allowed_rows[:, None], features, torch.zeros_like(features))
        picked = chosen_ok & allowed_rows[:, None]
        statistics[name] = {
            'eligible': int(allowed_rows.sum()),
            'read_positions': int(picked.sum()),
            'sampled_positions': sample_num,
            'rescue_positions': len(rescue_set),
            'maximum_read_distance': int(differences.gather(1, ix)[picked].max()) if bool(picked.any()) else 0,
        }
    return zeros, ft, statistics
