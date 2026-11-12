#!/usr/bin/env python3
"""FA2 R23: continuous PAL subword recurrence across document training chunks.

Replicates R13's recurrence/FFN/causal local attention, but carries the
state, previous token and attention K/V across consecutive chunks of ONE book.
Truncated BPTT is optional at explicit boundaries. The inherited checkpoint
is never changed by this helper. Distinguish forward-state persistence over
32K from full 32K backpropagation (not provided by truncated mode).
"""
import math

import torch
from torch.nn import functional as F


class ContinuousPALState:
    def __init__(self, book_id=None):
        self.reset(book_id)

    def reset(self, book_id):
        self.book_id = book_id
        self.cursor = 0
        self.state = None
        self.previous = None
        self.last_tag = None
        self.key_history = []
        self.value_history = []

    def detach(self):
        """Bound gradient history while keeping real recurrent state in memory."""
        if self.state is not None:
            self.state = self.state.detach()
        self.key_history = [x.detach() for x in self.key_history]
        self.value_history = [x.detach() for x in self.value_history]

    def forward_chunk(self, model, bridge, adapter, hidden, targets, tags, *,
                      book_id, token_offset, chunk_tag_offset,
                      detach_after=False):
        """Process contiguous token/word state using strictly previous targets.

        hidden: [B, words_in_this_chunk, D], computed from previous lexical
        context. tags: zero-based word indices local to this chunk, shifted
        here by chunk_tag_offset to maintain global word-change semantics.
        targets: [B, T] current next-token labels. No future target is read
        before its own prediction. Split only at word boundaries.
        """
        if self.book_id is None:
            self.book_id = book_id
        if book_id != self.book_id:
            raise ValueError("cross-book state leakage refused; call reset")
        if int(token_offset) != self.cursor:
            raise ValueError("noncontiguous document chunks refused")
        if hidden.ndim != 3 or targets.ndim != 2 or tags.shape != targets.shape:
            raise ValueError("invalid chunk tensors")
        B, T = targets.shape
        if T == 0 or hidden.shape[0] != B or bool((targets < 0).any()):
            raise ValueError("empty/padded stream not supported; chunk full books")
        if int(tags.min()) < 0 or int(tags.max()) >= hidden.shape[1]:
            raise ValueError("tag outside chunk words")
        # A single minibatch is assumed stable over the entire book.
        if self.previous is None:
            self.previous = torch.full((B,), model.memory.start_piece,
                                       dtype=torch.long, device=targets.device)
            self.state = hidden[:, 0]
            self.last_tag = tags[:, 0] + chunk_tag_offset
        if self.state.shape[0] != B or self.state.shape[-1] != hidden.shape[-1]:
            raise ValueError("batch/hidden dimensions changed mid-document")
        retain = torch.sigmoid(bridge.retain_logit)
        logits = []
        for t in range(T):
            local_tag = tags[:, t]
            global_tag = local_tag + chunk_tag_offset
            next_word = hidden.gather(
                1, local_tag[:, None, None].expand(-1, 1, hidden.shape[-1]))[:, 0]
            changed = global_tag != self.last_tag
            self.state = torch.where(
                changed[:, None],
                retain * self.state + (1.0 - retain) * bridge.context_gain * next_word,
                self.state)
            # Same next-piece causal recurrence as R13; previous is the last
            # *observed* piece from the prior position, never targets[:,t].
            self.state = model.memory.subword_cell(
                model.memory.piece_emb(self.previous), self.state)
            out = self.state
            if adapter is not None:
                if self.key_history:
                    keys = torch.stack(self.key_history, dim=1)
                    values = torch.stack(self.value_history, dim=1)
                    scores = (adapter.query(self.state)[:, None, :] * keys).sum(-1) / math.sqrt(self.state.shape[-1])
                    top = scores.topk(min(adapter.topk, scores.shape[1]), dim=-1)
                    selected = values.gather(1, top.indices[:, :, None].expand(-1, -1, self.state.shape[-1]))
                    recalled = (selected * F.softmax(top.values, dim=-1)[:, :, None]).sum(1)
                    out = out + adapter.attn_gain * adapter.attn_out(recalled)
                out = out + adapter.ffn_gain * adapter.ffn(self.state)
                self.key_history.append(adapter.key(self.state))
                self.value_history.append(adapter.value(self.state))
                if len(self.key_history) > adapter.capacity:
                    del self.key_history[0]
                    del self.value_history[0]
            logits.append(model.memory.subword_out(out))
            self.previous = targets[:, t]
            self.last_tag = global_tag
        self.cursor += T
        output = torch.stack(logits, dim=1)
        if detach_after:
            self.detach()
        return output

    def snapshot(self):
        """Diagnostic only. Never serialize model weights into this state."""
        return {"book_id": self.book_id, "token_cursor": self.cursor,
                "attention_cached": len(self.key_history),
                "has_recurrent_state": self.state is not None}


def contiguous_book_picks(book, chunk_span):
    """Successive word-boundary chunks for uninterrupted document streams.

    Return (start_chunk, end_chunk, start_token, end_token), not randomized
    noncontiguous windows. The last item contains the document remainder.
    """
    if chunk_span < 1:
        raise ValueError("chunk_span must be positive")
    n = len(book.chunks)
    for start in range(0, n, chunk_span):
        end = min(n, start + chunk_span)
        token_start = int(book.chunks[start])
        token_end = int(book.chunks[end]) if end < n else len(book.tokens)
        if token_end > token_start:
            yield start, end, token_start, token_end
