#!/usr/bin/env python3
"""Fast source-only FA2 R24 regressions; no R19 binary or corpus required."""
import copy
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from r18_long_memory import R18LongMemory
from r21_relevance_novelty import RelevanceNoveltyGate
from r22_multiscale_memory import R22Residual
from r23_context64 import ContextualSpanMemory, multipositive_value_loss, build_index
from r23_stream_carry import ContinuousPALState
from r24_unified_model import CausalMirrorMoE, UnifiedFA2, scramble_completed_values
from r24_unified_cpu import attach_historical_group


class FakeCore(nn.Module):
    def __init__(self, dim=12):
        super().__init__()
        self.word = nn.Embedding(256, dim)
        self.context = nn.Linear(72, dim)

    def hidden(self, words, lexical):
        return torch.tanh(self.word(words[:, 0].long()) + self.context(lexical))


class FakeMemory(nn.Module):
    def __init__(self, dim=12):
        super().__init__()
        self.start_piece = 1
        self.piece_emb = nn.Embedding(256, dim)
        self.subword_cell = nn.GRUCell(dim, dim)
        self.subword_out = nn.Linear(dim, 256)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = FakeCore()
        self.memory = FakeMemory()


class FakeBridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.retain_logit = nn.Parameter(torch.tensor(.4))
        self.context_gain = nn.Parameter(torch.tensor(.5))


class FakeAdapter(nn.Module):
    def __init__(self, dim=12):
        super().__init__()
        self.capacity, self.topk = 8, 4
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.attn_out = nn.Linear(dim, dim)
        self.attn_gain = nn.Parameter(torch.tensor(.8))
        self.ffn = nn.Sequential(nn.Linear(dim, 24), nn.Tanh(),
                                 nn.Linear(24, dim))
        self.ffn_gain = nn.Parameter(torch.tensor(.5))


class FakeBook:
    def __init__(self, book_id="source-book"):
        self.book_id = book_id
        self.tokens = (np.arange(300, dtype=np.int64) * 17 % 201 + 2)
        self.token_tag = np.repeat(np.arange(100), 3)
        self.examples = (np.arange(700).reshape(100, 7) % 255).astype(np.int64)
        self.context = np.random.default_rng(24024).normal(
            size=(100, 72)).astype(np.float32)
        self.byte_starts = np.arange(100, dtype=np.int64) * 3
        self.raw_len = 300


def run():
    torch.set_num_threads(2)
    torch.manual_seed(24024)
    checks = {}
    base = FakeModel()
    r18 = R18LongMemory(base.memory.piece_emb.weight, dim=12, key_dim=12)
    r21, r22 = RelevanceNoveltyGate(), R22Residual()
    r23 = ContextualSpanMemory(base.memory.piece_emb.weight, projection=12)
    unified = UnifiedFA2(
        base, FakeBridge(), FakeAdapter(), r18, r21, r22, r23,
        local_history=128, expert_width=24, expert_count=4, expert_topk=2)
    book = FakeBook()
    carry = ContinuousPALState(book.book_id)
    first = unified.forward_chunk(book, (0, 70, 0, 210), carry)
    second = unified.forward_chunk(book, (70, 100, 210, 300), carry,
                                   counterfactual=True)
    checks["same_document_state_carried"] = carry.cursor == 300
    checks["memory_index_causal"] = bool(
        (second["memory_index"]["positions"][
            second["memory_index"]["mask"]] <= 210 - 8).all())
    checks["zero_r23_exact_identity"] = torch.equal(
        second["logits"], second["without_r23"])
    checks["zero_moe_safe"] = float(unified.moe.amplitude) == 0
    checks["scrambled_control_exact_identity_at_zero"] = torch.equal(
        second["logits"], second["scrambled_values"])

    donor = FakeBook("unrelated-book")
    donor.tokens = np.roll(donor.tokens, 21)
    alternate = build_index(
        donor.tokens, 210, 90,
        query_book_tokens=book.tokens, query_start=210)
    checks["cross_book_control_uses_actual_query"] = torch.equal(
        alternate["query"], second["memory_index"]["query"])
    checks["cross_book_values_source_only"] = bool(
        (alternate["positions"][alternate["mask"]] <= 210 - 8).all())
    checks["three_tier_metadata"] = set(
        second["memory_index"]["coverage"]) == {"recent", "middle", "far"}
    checks["finite_causal_logits"] = bool(
        torch.isfinite(first["logits"]).all() and
        torch.isfinite(second["logits"]).all())
    checks["single_live_book_cursor"] = second["start"] == 210
    ce = F.cross_entropy(second["logits"], second["target"])
    span, count = multipositive_value_loss(
        second["memory_detail"]["scores"],
        second["memory_index"], second["target"])
    (ce + .1 * span + .002 * second["moe_balance"]).backward()
    checks["core_receives_lm_gradient"] = bool(
        base.memory.subword_cell.weight_ih.grad.abs().sum() > 0)
    checks["moe_residual_gate_receives_gradient"] = bool(
        unified.moe.amplitude.grad.abs().sum() > 0)
    checks["finite_joint_gradients"] = all(
        p.grad is None or bool(torch.isfinite(p.grad).all())
        for p in unified.parameters())
    checks["all_modules_in_one_graph"] = len(list(unified.parameters())) > 40

    base.memory.piece_emb.weight.grad = None
    candidate = torch.tensor([[2 + t for t in range(64)]])
    unified.r23.encode(
        candidate, True, piece_table=base.memory.piece_emb.weight
    )[:, 0].sum().backward()
    checks["live_tied_embedding_gradient"] = bool(
        base.memory.piece_emb.weight.grad.abs().sum() > 0)

    delta = torch.ones(3, 256)
    features = torch.zeros(3, 8)
    with torch.no_grad():
        r21.amplitude.fill_(.8)
    original, _ = r21(torch.zeros_like(delta), delta, features)
    shifted, _ = r21(torch.zeros_like(delta), delta, features,
                     query_position_offset=256)
    checks["r21_absolute_position_gate"] = bool(
        torch.equal(original, torch.zeros_like(delta)) and shifted.abs().sum() > 0)
    with torch.no_grad():
        for sub in r22.branches.values():
            sub.amplitude.fill_(.2)
    zeros = {k: torch.zeros_like(delta) for k in r22.branches}
    ones = {k: torch.ones_like(delta) for k in r22.branches}
    feats = {k: features.clone() for k in r22.branches}
    old22, _ = r22(torch.zeros_like(delta), ones, feats)
    new22, _ = r22(torch.zeros_like(delta), ones, feats,
                    query_position_offset=256)
    checks["r22_all_tiers_absolute_position"] = bool(
        torch.equal(old22, torch.zeros_like(delta)) and new22.abs().sum() > 0)

    # Restore a real AdamW moment tensor into the unified optimizer group.
    linear = nn.Linear(3, 2)
    previous_opt = torch.optim.AdamW(linear.parameters(), lr=.01)
    previous_opt.zero_grad(set_to_none=True)
    linear(torch.ones(1, 3)).sum().backward()
    previous_opt.step()
    history = {"optimizer": copy.deepcopy(previous_opt.state_dict())}
    another = nn.Linear(3, 2)
    first_group = torch.optim.AdamW(another.parameters(), lr=.01)
    revived = nn.Linear(3, 2)
    count_restored = attach_historical_group(
        first_group, revived, history, .01, "test adapter")
    checks["adam_moments_not_reset"] = bool(
        count_restored == 2 and len(first_group.param_groups) == 2 and
        all("exp_avg" in first_group.state[p] for p in revived.parameters()))
    checks["mirror_expert_top2"] = unified.moe.topk == 2
    try:
        wrong = ContinuousPALState("other-book")
        unified.forward_chunk(book, (70, 100, 210, 300), wrong)
        checks["reject_skipped_book_prefix"] = False
    except ValueError:
        checks["reject_skipped_book_prefix"] = True
    assert all(checks.values()), json.dumps(checks, indent=2)
    print(json.dumps({"FA2_R24_source_checks": checks, "passed": len(checks),
                      "value_horizons": count}, indent=2))


if __name__ == "__main__":
    run()
