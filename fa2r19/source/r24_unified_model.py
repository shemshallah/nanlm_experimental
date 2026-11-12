#!/usr/bin/env python3
"""FA2 R24: ONE PAL backbone + R18/R21/R22/R23 + causal mirror-MoE.

All paths share one forward pass, one optimizer and one candidate checkpoint.
The accepted R18 file is an immutable parent, not an experimental output.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from r18_long_memory import build_copy_queries, apply_copy_boost
from r21_relevance_novelty import causal_proposal
from r22_multiscale_memory import hierarchical_proposals
from r23_context64 import build_index


class SwiGLUExpert(nn.Module):
    def __init__(self, dim, width):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, width * 2)
        self.down = nn.Linear(width, dim)

    def forward(self, x):
        a, b = self.up(self.norm(x)).chunk(2, -1)
        return self.down(F.silu(a) * b)


class CausalMirrorMoE(nn.Module):
    """Token-local top-k routed experts, mirror features, safe zero residual.

    Routing and experts see only the current *causal* recurrent representation.
    Per-token evaluation uses selected experts only, not all expert projections.
    """
    def __init__(self, dim, width=384, experts=4, topk=2):
        super().__init__()
        if not (1 <= topk <= experts and width >= dim):
            raise ValueError("invalid expert configuration")
        self.router = nn.Linear(dim * 2, experts)
        self.experts = nn.ModuleList(SwiGLUExpert(dim, width) for _ in range(experts))
        self.mirror_gain = nn.Parameter(torch.tensor(0.0))
        self.amplitude = nn.Parameter(torch.tensor(0.0))
        self.topk = topk
        self.width = width

    def forward(self, states):
        dim = states.shape[-1]
        flat = states.reshape(-1, dim)
        reflect = flat.flip(-1)
        probabilities = F.softmax(self.router(torch.cat((flat, reflect), -1)), -1)
        chosen, indices = probabilities.topk(self.topk, -1)
        chosen = chosen / chosen.sum(-1, keepdim=True).clamp_min(1e-8)
        combined = torch.zeros_like(flat)
        mirror = torch.tanh(self.mirror_gain) * reflect
        for j, expert in enumerate(self.experts):
            rows, slots = (indices == j).nonzero(as_tuple=True)
            if rows.numel():
                sign = 1.0 if j % 2 == 0 else -1.0
                routed = expert(flat[rows] + sign * mirror[rows])
                combined.index_add_(0, rows, routed * chosen[rows, slots, None])
        balance = (probabilities.mean(0).square().sum() *
                   len(self.experts) - 1.0)
        return combined.reshape_as(states), balance


def scramble_completed_values(memory):
    """Same-history wrong key/value association control; no query target used."""
    copied = dict(memory)
    values = memory["values"].clone()
    for i in range(len(values)):
        eligible = memory["mask"][i].nonzero(as_tuple=True)[0]
        if eligible.numel() > 1:
            values[i, eligible] = memory["values"][i, eligible.roll(1)]
    copied["values"] = values
    return copied


class UnifiedFA2(nn.Module):
    def __init__(self, model, bridge, adapter, r18, r21, r22, r23, *,
                 local_history=512, expert_width=384, expert_count=4, expert_topk=2):
        super().__init__()
        if not (65 <= local_history <= 2048):
            raise ValueError("local history must be in [65,2048]")
        self.model, self.bridge, self.adapter = model, bridge, adapter
        self.r18, self.r21, self.r22, self.r23 = r18, r21, r22, r23
        dim = model.memory.piece_emb.weight.shape[-1]
        self.moe = CausalMirrorMoE(dim, expert_width, expert_count, expert_topk)
        self.local_history = int(local_history)

    def forward_chunk(self, book, segment, carry, *, counterfactual=False,
                      donor_book=None):
        begin, end, t0, t1 = segment
        examples = torch.as_tensor(book.examples[begin:end].astype(np.int64))[None]
        lexical = torch.as_tensor(book.context[begin:end].astype(np.float32))[None]
        tags = torch.as_tensor(book.token_tag[t0:t1].astype(np.int64))[None] - begin
        targets = torch.as_tensor(book.tokens[t0:t1].astype(np.int64))[None]
        if targets.numel() == 0 or tags.min() < 0 or tags.max() >= end - begin:
            raise ValueError("noncausal or empty word-aligned segment")
        hidden = self.model.core.hidden(
            examples.reshape(-1, 7), lexical.reshape(-1, 72)
        ).reshape(1, end - begin, -1)
        logits, states = carry.forward_chunk(
            self.model, self.bridge, self.adapter, hidden, targets, tags,
            book_id=book.book_id, token_offset=t0, chunk_tag_offset=begin,
            detach_after=False, return_states=True)
        deep, balance = self.moe(states)
        logits = logits[0] + torch.tanh(self.moe.amplitude) * (
            self.model.memory.subword_out(deep)[0])
        y = targets[0]

        # Local R18 and R21 have a real carried token prefix, not a reset at
        # every 32-word window. The prior logits are padding: only current
        # positions are supervised, and prior historical values are causal.
        from_pos = max(0, t0 - self.local_history)
        prefix = torch.as_tensor(book.tokens[from_pos:t0].astype(np.int64))
        context = torch.cat((prefix, y))
        prior_logits = logits.new_zeros((len(prefix), logits.shape[-1]))
        q18 = build_copy_queries(context, mem=self.r18.max_memory,
                                 gap=self.r18.min_gap)
        post18 = apply_copy_boost(
            self.r18, torch.cat((prior_logits, logits), 0), q18)[len(prefix):]
        r21_delta, f21, _ = causal_proposal(
            self.r18, torch.cat((prior_logits, post18), 0), context)
        post21, _ = self.r21(post18, r21_delta[len(prefix):],
                            f21[len(prefix):], query_position_offset=t0)
        proposed22, f22, _ = hierarchical_proposals(
            self.r18, post21, book.tokens, t0)
        post22, _ = self.r22(post21, proposed22, f22,
                            query_position_offset=t0)
        details = None
        memory = None
        actual = post22
        permuted = post22
        unrelated = None
        if t0 >= 64:
            memory = build_index(book.tokens, t0, t1 - t0)
            memory["query_position_offset"] = t0
            actual, details = self.r23(
                post22, memory, piece_table=self.model.memory.piece_emb.weight)
            if counterfactual:
                permuted, _ = self.r23(
                    post22, scramble_completed_values(memory),
                    piece_table=self.model.memory.piece_emb.weight)
                # Unrelated book MUST use the current book's exact causal
                # query. Its values and index positions come exclusively
                # from the earlier completed portion of a different book.
                if donor_book is not None and donor_book.book_id != book.book_id:
                    donor_start = min(t0, len(donor_book.tokens) - (t1 - t0))
                    if donor_start >= 129:
                        alternate = build_index(
                            donor_book.tokens, donor_start, t1 - t0,
                            query_book_tokens=book.tokens, query_start=t0)
                        alternate["query_position_offset"] = t0
                        unrelated, _ = self.r23(
                            post22, alternate,
                            piece_table=self.model.memory.piece_emb.weight)
        byte_start = int(book.byte_starts[begin])
        byte_end = (int(book.byte_starts[end]) if end < len(book.byte_starts)
                    else int(book.raw_len))
        if byte_end <= byte_start:
            raise ValueError("nonpositive source-verified byte span")
        return {"logits": actual, "without_r23": post22,
                "scrambled_values": permuted, "unrelated_history": unrelated,
                "target": y,
                "bytes": byte_end - byte_start, "memory_detail": details,
                "memory_index": memory,
                "moe_balance": balance, "start": t0}
