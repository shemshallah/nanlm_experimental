#!/usr/bin/env python3
"""Continuous PAL recurrence matches R13 single-window predictions exactly."""
import json

import torch
from torch import nn

from r13_cpu import run_window_fast
from r23_stream_carry import ContinuousPALState, contiguous_book_picks


class DummyMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.start_piece = 1
        self.piece_emb = nn.Embedding(64, 12)
        self.subword_cell = nn.GRUCell(12, 12)
        self.subword_out = nn.Linear(12, 64)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.memory = DummyMemory()


class DummyBridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.retain_logit = nn.Parameter(torch.tensor(0.4))
        self.context_gain = nn.Parameter(torch.tensor(0.5))


class DummyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.capacity, self.topk = 8, 4
        self.query = nn.Linear(12, 12)
        self.key = nn.Linear(12, 12)
        self.value = nn.Linear(12, 12)
        self.attn_out = nn.Linear(12, 12)
        self.attn_gain = nn.Parameter(torch.tensor(0.8))
        self.ffn = nn.Sequential(nn.Linear(12, 24), nn.Tanh(), nn.Linear(24, 12))
        self.ffn_gain = nn.Parameter(torch.tensor(0.5))


def run():
    torch.set_num_threads(2)
    torch.manual_seed(23777)
    model, bridge, adapter = DummyModel(), DummyBridge(), DummyAdapter()
    h = torch.randn(1, 6, 12)
    tags = torch.arange(6).repeat_interleave(3).reshape(1, 18)
    y = torch.randint(2, 64, (1, 18))
    expected = run_window_fast(model, bridge, adapter, h, y, tags,
                               training=True, return_logits=True)
    all_at_once = ContinuousPALState()
    actual = all_at_once.forward_chunk(model, bridge, adapter, h, y, tags,
                                       book_id="book-a", token_offset=0,
                                       chunk_tag_offset=0)
    stream = ContinuousPALState("book-a")
    first = stream.forward_chunk(model, bridge, adapter, h[:, :3], y[:, :9],
                                 tags[:, :9], book_id="book-a",
                                 token_offset=0, chunk_tag_offset=0)
    second = stream.forward_chunk(model, bridge, adapter, h[:, 3:], y[:, 9:],
                                  tags[:, 9:] - 3, book_id="book-a",
                                  token_offset=9, chunk_tag_offset=3)
    joined = torch.cat((first, second), dim=1)
    check = {}
    check["matches_existing_r13_single_window"] = torch.equal(actual, expected)
    check["split_and_unsplit_logits_identical"] = torch.equal(joined, actual)
    check["long_term_previous_token_carried"] = torch.equal(stream.previous, y[:, -1])
    check["global_token_cursor_continues"] = stream.cursor == 18
    check["bounded_attention_cache"] = stream.snapshot()["attention_cached"] == 8
    try:
        stream.forward_chunk(model, bridge, adapter, h[:, :1], y[:, :1],
                             tags[:, :1], book_id="book-b", token_offset=18,
                             chunk_tag_offset=0)
        check["cross_document_isolation"] = False
    except ValueError:
        check["cross_document_isolation"] = True
    try:
        stream.forward_chunk(model, bridge, adapter, h[:, :1], y[:, :1],
                             tags[:, :1], book_id="book-a", token_offset=0,
                             chunk_tag_offset=0)
        check["noncontiguous_chunks_rejected"] = False
    except ValueError:
        check["noncontiguous_chunks_rejected"] = True
    stream.detach()
    check["detached_grad_history_but_state_retained"] = (stream.state is not None and
                                                         stream.state.grad_fn is None and
                                                         stream.cursor == 18)
    class Book:
        chunks = list(range(0, 22, 2))
        tokens = list(range(25))
    pieces = list(contiguous_book_picks(Book(), 3))
    check["contiguous_token_boundaries"] = (pieces[0] == (0, 3, 0, 6) and
                                            pieces[1] == (3, 6, 6, 12) and
                                            pieces[-1] == (9, 11, 18, 25))
    assert all(check.values()), json.dumps(check, indent=2)
    print(json.dumps({"checks": check, "passed": len(check),
                      "book_state": stream.snapshot()}, indent=2))


if __name__ == "__main__":
    run()
