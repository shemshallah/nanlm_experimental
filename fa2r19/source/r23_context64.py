#!/usr/bin/env python3
"""FA2 R23: trainable 4/8/16/32/64-token contextual retrieval over R22 history.

One inherited PAL decoder, not a new language model. A candidate at j has
key tokens [j-64:j] and a stored value span [j:j+8]. At query t, the key
comes from [t-64:t], never from the target at t. Every stored value is in a
previous document window; index admission is a deterministic function of the
PAST only. No target-conditioned retrieval or test-set threshold selection.
"""
from collections import defaultdict
import math

import torch
from torch import nn
from torch.nn import functional as F

LENGTHS = (4, 8, 16, 32, 64)
TIERS = ("recent", "middle", "far")
BANDS = ((65, 2048), (2048, 8192), (8192, 32769))
HORIZONS = (1, 2, 4, 8)


def _history(tokens, endpoint, width=64):
    """Left-padded strictly-prior sequence; -1 never indexes a real token."""
    row = [-1] * width
    history = [int(z) for z in tokens[max(0, endpoint - width):endpoint]]
    row[-len(history):] = history
    return row


def build_index(book_tokens, window_start, window_length, *,
                memory=32768, gap=65, value_length=8, per_band=12):
    """Bounded query-specific candidate index, with 32K persistent addresses.

    Only prior-window source values are eligible: j + value_length <= start.
    Candidates are selected without inspecting book_tokens[t] or its suffix.
    Last-token and 2-token buckets supply both hard lexical distractors and
    exact-prefix rescue; sparse anchors cover the uncompressed horizon.
    """
    tok = [int(v) for v in book_tokens]
    if not (64 <= window_start <= len(tok) - window_length):
        raise ValueError("bad query window")
    if memory < 8192 or gap < value_length or per_band < 3:
        raise ValueError("bad retrieval bounds")
    if any(v < 0 for v in tok):
        raise ValueError("invalid token IDs")
    limit = min(memory, 32768)
    sources = range(max(64, window_start - limit),
                    window_start - value_length + 1)
    exact = [defaultdict(list) for _ in TIERS]
    last = [defaultdict(list) for _ in TIERS]
    anchors = [[] for _ in TIERS]
    # Index is built from stored, completed document history only.
    for j in sources:
        d = window_start - j
        tier = next((b for b, (lo, hi) in enumerate(BANDS)
                     if lo <= d < hi), None)
        if tier is None:
            continue
        exact[tier][(tok[j - 2], tok[j - 1])].append(j)
        last[tier][tok[j - 1]].append(j)
        stride = (48, 256, 1024)[tier]
        if j % stride == 0:
            anchors[tier].append(j)
    C = per_band * len(TIERS)
    query = torch.tensor([_history(tok, window_start + t)
                          for t in range(window_length)], dtype=torch.long)
    keys = torch.full((window_length, C, 64), -1, dtype=torch.long)
    values = torch.zeros((window_length, C, value_length), dtype=torch.long)
    positions = torch.full((window_length, C), -1, dtype=torch.long)
    masks = torch.zeros((window_length, C), dtype=torch.bool)
    distances = torch.ones((window_length, C), dtype=torch.float32)
    bands = torch.zeros((window_length, C), dtype=torch.long)
    coverage = {name: 0 for name in TIERS}
    furthest = {name: 0 for name in TIERS}

    for t in range(window_length):
        pos = window_start + t
        pair = (tok[pos - 2], tok[pos - 1])
        prior = tok[pos - 1]
        for b, (low, high) in enumerate(BANDS):
            def admissible(j):
                d = pos - j
                return (max(gap, low) <= d < high and
                        j + value_length <= window_start and
                        j >= pos - limit)

            # Queries see only their own exact-prefix rescue candidates.
            # No candidate rescued by a later query enters this shortlist.
            selected = []
            pool = (reversed(exact[b].get(pair, [])[-16:]),
                    reversed(last[b].get(prior, [])[-32:]),
                    reversed(anchors[b]))
            for source in pool:
                for j in source:
                    if admissible(j) and j not in selected:
                        selected.append(j)
                        if len(selected) == per_band:
                            break
                if len(selected) == per_band:
                    break
            for k, j in enumerate(selected):
                c = b * per_band + k
                keys[t, c] = torch.tensor(_history(tok, j))
                values[t, c] = torch.tensor(tok[j:j + value_length])
                positions[t, c] = j
                distances[t, c] = pos - j
                bands[t, c] = b
                masks[t, c] = True
                coverage[TIERS[b]] += 1
                furthest[TIERS[b]] = max(furthest[TIERS[b]], pos - j)
    return {"query": query, "keys": keys, "values": values,
            "positions": positions, "mask": masks, "distance": distances,
            "tier": bands, "window_start": window_start,
            "coverage": coverage, "max_distance": furthest}


class ContextualSpanMemory(nn.Module):
    """Trainable context Q/K and 8-token value read, inherited embeddings frozen.

    The R23 residual amplitude is exactly zero at migration. All parameters in
    this module are new; accepted R18 and frozen R21/R22 state are untouched.
    """
    def __init__(self, frozen_piece_embeddings, projection=64,
                 minimum_query_position=192, topk=8):
        super().__init__()
        self.register_buffer("pieces", frozen_piece_embeddings.detach().clone(),
                             persistent=False)
        d = frozen_piece_embeddings.shape[-1]
        self.q_attention = nn.Linear(d, 1, bias=False)
        self.k_attention = nn.Linear(d, 1, bias=False)
        self.q_mix = nn.Parameter(torch.zeros(len(LENGTHS)))
        self.k_mix = nn.Parameter(torch.zeros(len(LENGTHS)))
        self.query_net = nn.Sequential(nn.LayerNorm(2 * d),
                                       nn.Linear(2 * d, projection * 2),
                                       nn.GELU(), nn.Linear(projection * 2, projection))
        self.key_net = nn.Sequential(nn.LayerNorm(2 * d),
                                     nn.Linear(2 * d, projection * 2),
                                     nn.GELU(), nn.Linear(projection * 2, projection))
        self.log_temperature = nn.Parameter(torch.tensor(1.0))
        self.recency = nn.Parameter(torch.tensor(0.0))
        self.tier_bias = nn.Parameter(torch.zeros(len(TIERS)))
        self.gate = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 12),
                                  nn.Tanh(), nn.Linear(12, 1))
        self.amplitude = nn.Parameter(torch.tensor(0.0))
        self.minimum_query_position = int(minimum_query_position)
        self.topk = int(topk)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def encode(self, rows, query):
        if rows.shape[-1] != 64:
            raise ValueError("context must be 64 tokens, left padded with -1")
        shape = rows.shape[:-1]
        rows = rows.reshape(-1, 64).long()
        if bool((rows >= self.pieces.shape[0]).any()):
            raise ValueError("out-of-vocabulary memory token")
        valid = rows.ne(-1)
        embedded = F.embedding(rows.clamp_min(0), self.pieces)
        att = self.q_attention if query else self.k_attention
        mix = F.softmax(self.q_mix if query else self.k_mix, dim=0)
        pooled = []
        for length in LENGTHS:
            e, mask = embedded[:, -length:], valid[:, -length:]
            avg = (e * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp_min(1).unsqueeze(-1)
            weights = att(e).squeeze(-1).masked_fill(~mask, -1e4)
            weights = F.softmax(weights, dim=-1) * mask.float()
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
            weighted = (e * weights.unsqueeze(-1)).sum(1)
            pooled.append(torch.cat((avg, weighted), dim=-1))
        pooled = (torch.stack(pooled, dim=1) * mix[None, :, None]).sum(1)
        network = self.query_net if query else self.key_net
        return F.normalize(network(pooled), dim=-1, eps=1e-6).reshape(*shape, -1)

    def score(self, memory):
        q = self.encode(memory["query"], True)
        k = self.encode(memory["keys"], False)
        scores = (q[:, None, :] * k).sum(-1) * self.log_temperature.exp().clamp(0.1, 40)
        scores = scores - self.recency * torch.log1p(memory["distance"])
        scores = scores + self.tier_bias[memory["tier"]]
        return scores.masked_fill(~memory["mask"], -1e4)

    def read(self, base_logits, memory, *, scores=None):
        if scores is None:
            scores = self.score(memory)
        T, vocab = base_logits.shape
        if memory["query"].shape[0] != T:
            raise ValueError("unaligned memory window")
        k = min(self.topk, scores.shape[1])
        selected, ix = scores.topk(k, dim=-1)
        admissible = memory["mask"].gather(1, ix)
        weights = F.softmax(selected.masked_fill(~admissible, -1e4), -1)
        weights = weights * admissible.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        H = memory["values"].shape[-1]
        spans = memory["values"].gather(1, ix[:, :, None].expand(-1, -1, H))
        if bool((spans >= vocab).any()):
            raise ValueError("value outside decoder vocabulary")
        distributions = torch.zeros((T, H, vocab), device=base_logits.device,
                                     dtype=base_logits.dtype)
        for h in range(H):
            distributions[:, h].scatter_add_(1, spans[:, :, h],
                                               weights.to(base_logits.dtype))
        p = F.softmax(base_logits.detach(), dim=-1)
        picked = spans[:, :, 0]
        conf = torch.stack((
            selected[:, 0].clamp(-20, 20),
            (selected[:, 0] - selected[:, 1]).clamp(-10, 10)
            if k > 1 else torch.zeros_like(selected[:, 0]),
            (weights * p.gather(1, picked)).sum(-1),
            (weights > 0).sum(-1).float() / k,
        ), dim=-1)
        confidence = torch.sigmoid(self.gate(conf)).squeeze(-1)
        # Streamed chunks carry an absolute document cursor; conventional
        # R22 windows intentionally retain their historical local cutoff.
        offset = int(memory.get("query_position_offset", 0))
        eligible = (memory["mask"].any(-1) &
                    (torch.arange(T, device=scores.device) + offset >=
                     self.minimum_query_position))
        gain = (2 * torch.tanh(self.amplitude) *
                confidence * eligible.float())
        return base_logits + gain[:, None] * distributions[:, 0], {
            "scores": scores, "selected_values": spans, "weights": weights,
            "future_distributions": distributions, "gain": gain,
            "eligible": eligible,
        }

    def forward(self, base_logits, memory):
        return self.read(base_logits, memory)


def multipositive_value_loss(scores, memory, targets, horizons=HORIZONS):
    """Multi-positive InfoNCE: target labels are used ONLY in the loss.

    No target-dependent candidate admission, relevance score, or inference.
    Longer spans are supervised only when all horizon targets are available.
    """
    T = targets.shape[0]
    losses = []
    counts = {}
    vals = memory["values"]
    for h in horizons:
        if T < h or vals.shape[-1] < h:
            counts[str(h)] = 0
            continue
        labels = torch.stack([targets[t:t + h] for t in range(T - h + 1)])
        positives = (vals[:len(labels), :, :h] == labels[:, None, :]).all(-1)
        positives = positives & memory["mask"][:len(labels)]
        rows = positives.any(-1)
        counts[str(h)] = int(rows.sum())
        if bool(rows.any()):
            sub = scores[:len(labels)][rows]
            pos = sub.masked_fill(~positives[rows], -1e4)
            losses.append((torch.logsumexp(sub, -1) -
                           torch.logsumexp(pos, -1)).mean())
    return (torch.stack(losses).mean() if losses else
            scores.sum() * 0), counts


def multipositive_semantic_loss(scores, positives):
    """Supervise same-meaning paraphrases, plus same-entity/relation hard negatives."""
    if scores.shape != positives.shape or not bool(positives.any(-1).all()):
        raise ValueError("every semantic query requires a positive")
    return (torch.logsumexp(scores, -1) -
            torch.logsumexp(scores.masked_fill(~positives, -1e4), -1)).mean()
