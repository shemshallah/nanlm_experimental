#!/usr/bin/env python3
"""FA1 R7.53: additive causal successor distribution, directly from R7.49.

No new language model, no replacement checkpoint. Every mixture weight and
reliability feature is computed before the supervised next word. The label is
used only to read a probability from an already-normalized distribution.
"""
from __future__ import annotations

import contextlib
import torch
from torch import nn

from fa1_r744_causal_word_backoff import UnifiedR744Backoff
from fa1_r748_mirror_pair_values import PAIR_COUNT, pair_indices
from fa1_r749_address_reliability import ReliableAddressedMirrorBackoff

ROUND = "FA1_R7_53_CAUSAL_SUCCESSOR_MIXTURE_V1"
PARENT = "FA1_R7_49_CAUSAL_PAIR_ADDRESS_RELIABILITY_V1"
CONTROLS = ("full", "disabled", "word_only", "wrong_addresses",
            "wrong_successors", "no_apex")


def causal_successor_evidence(history, history_mask, head_lookup, targets,
                              control="full"):
    """Return target probabilities and pre-target support, with no future reads.

    history[:, -1] is the last *observed* word, never the target. Only pairs
    (history[t], history[t+1]) for t < last_history_index are considered.
    Short documents are padded on the left, with a separate causal mask.
    """
    if control not in CONTROLS:
        raise ValueError("unknown R7.53 control")
    if history.ndim != 2 or history.size(1) < 2:
        raise ValueError("successor history must be [B,H], H >= 2")
    if history_mask.shape != history.shape:
        raise ValueError("successor mask mismatch")
    if targets.shape != (history.size(0),):
        raise ValueError("one supervised target per history")
    if bool(((history < 0) | (history >= head_lookup.size(0))).any()):
        raise ValueError("history word ID out of vocabulary")
    valid = history_mask.bool()
    query = history[:, -1]
    predecessors = history[:, :-1]
    successors = history[:, 1:]
    usable = (valid[:, :-1] & valid[:, 1:] & valid[:, -1, None]
              & (query[:, None] != 1))
    if control == "wrong_successors":
        # Preserve all historical matching addresses, scramble ONLY values.
        successors = (successors + 7919) % head_lookup.size(0)
    # Relative age counts only causal, unmasked predecessor->successor pairs.
    # Adding extra BOS padding cannot change an otherwise identical history.
    rank = usable.float().cumsum(dim=1) - 1.
    span = (usable.float().sum(dim=1, keepdim=True) - 1.).clamp_min(1.)
    age = .25 + .75 * rank.clamp_min(0.) / span
    exact = usable & (predecessors == query[:, None])
    q_heads = head_lookup[query.long()]
    p_heads = head_lookup[predecessors.long()]
    query_pair, q_valid = pair_indices(q_heads[:, 0], q_heads[:, 1])
    prior_pair, p_valid = pair_indices(p_heads[..., 0], p_heads[..., 1])
    if control == "wrong_addresses":
        prior_pair = (prior_pair + 7919) % PAIR_COUNT
    q_sign = q_heads[:, 0] < q_heads[:, 1]
    p_sign = p_heads[..., 0] < p_heads[..., 1]
    geometric = (usable & (predecessors != query[:, None])
                 & q_valid[:, None] & p_valid
                 & (prior_pair == query_pair[:, None])
                 & (p_sign == q_sign[:, None]))
    # Distinct nonlexical predecessor count prevents one repeated collision
    # from masquerading as independent corroboration.
    ordered = torch.where(geometric, predecessors, -1).sort(dim=1).values
    previous = torch.cat((torch.full_like(ordered[:, :1], -1),
                          ordered[:, :-1]), dim=1)
    distinct = ((ordered >= 0) & (ordered != previous)).sum(dim=1)
    wmass = exact.float() * age
    gmass = geometric.float() * age
    wtotal = wmass.sum(dim=1)
    gtotal = gmass.sum(dim=1)
    # The next-word target enters only these FINAL probability gathers.
    matching = (successors == targets[:, None]).float()
    qw = (wmass * matching).sum(dim=1) / wtotal.clamp_min(1.e-20)
    qg = (gmass * matching).sum(dim=1) / gtotal.clamp_min(1.e-20)
    word_support = exact.sum(dim=1)
    geo_support = geometric.sum(dim=1)
    eligible = (geo_support >= 2) & (distinct >= 2)
    return qw, qg, word_support, geo_support, distinct, eligible


def compose_inherited_distribution(old_alpha, old_q, word_q, geo_q,
                                   word_gain, geo_gain):
    """Algebraically compose with R7.49's existing normalized copy mixture.

    Inherited: (1-a0) P_core + a0 Q_old.
    Successor: (1-w-g) P_R749 + w Q_word + g Q_geo.
    The returned pair is equivalent to the latter, using the original
    UnifiedR744Backoff.joint_nll implementation unchanged.
    """
    epsilon = word_gain + geo_gain
    if bool((epsilon < 0).any()) or bool((epsilon > .25 + 1.e-7).any()):
        raise ValueError("R7.53 successor weights must be in [0, .25]")
    if not bool((epsilon > 0).any()):
        return old_alpha, old_q  # EXACT zero-gain inherited parity.
    combined_alpha = old_alpha + (1. - old_alpha) * epsilon
    numerator = ((1. - epsilon) * old_alpha * old_q
                 + word_gain * word_q + geo_gain * geo_q)
    combined_q = torch.where(
        epsilon > 0, numerator / combined_alpha.clamp_min(1.e-20), old_q)
    return combined_alpha, combined_q


class CausalSuccessorBackoff(ReliableAddressedMirrorBackoff):
    def forward(self, data, trace, targets, embeddings=None):
        old_alpha, old_q, telemetry = super().forward(
            data, trace, targets, embeddings)
        mode = getattr(self, "_r753_control", "full")
        if mode not in CONTROLS:
            raise ValueError("unknown successor intervention")
        # The inherited R7.49 backoff still sees its ORIGINAL 32-word input.
        # A distinct history field carries 512..32K causal successor evidence.
        history = data.get("r753_successor_history_ids", data["word_ids"]).long()
        mask = data.get("r753_successor_history_mask")
        if mask is None:
            mask = history != 1
        qw, qg, wc, gc, distinct, eligible = causal_successor_evidence(
            history, mask, self.word_head_lookup, targets, mode)
        word_gain = (.125 * torch.tanh(self.r753_word_gain).clamp_min(0.)
                     * (wc > 0).float())
        summary = trace["descended_point_summary"].detach()
        if summary.shape != (history.size(0), 32):
            raise ValueError("successor reliability requires 32D apex descent")
        # Isolate the new successor gate's apex input; keep observed
        # support counts and inherited R7.49 forward unchanged.
        if mode == "no_apex":
            summary = torch.zeros_like(summary)
        features = torch.cat((
            summary, wc.float().log1p()[:, None],
            gc.float().log1p()[:, None],
            distinct.float().log1p()[:, None]), dim=1)
        trust = torch.sigmoid(self.r753_reliability(features).squeeze(-1))
        geo_gain = (.125 * torch.tanh(self.r753_geometry_gain).clamp_min(0.)
                    * trust * eligible.float())
        if mode in ("disabled",):
            word_gain = word_gain * 0.
            geo_gain = geo_gain * 0.
        if mode == "word_only":
            geo_gain = geo_gain * 0.
        alpha, q = compose_inherited_distribution(
            old_alpha, old_q, qw, qg, word_gain, geo_gain)
        telemetry.update({
            "r753_word_support": wc.detach(),
            "r753_geometric_support": gc.detach(),
            "r753_distinct_predecessors": distinct.detach(),
            "r753_geometric_eligible": eligible.detach(),
            "r753_word_weight": word_gain.detach(),
            "r753_geometric_weight": geo_gain.detach(),
            "r753_reliability": trust.detach(),
        })
        return alpha, q, telemetry


def attach_causal_successors(model, optimizer, learning_rate=3.e-4):
    """Append precisely one optimizer group to the 29-group R7.49 lineage."""
    if (type(model) is not UnifiedR744Backoff
            or type(model.word_backoff) is not ReliableAddressedMirrorBackoff
            or len(optimizer.param_groups) != 29):
        raise TypeError("R7.53 requires intact R7.49 model and 29 groups")
    old = {id(p) for p in model.parameters()}
    module = model.word_backoff
    module.__class__ = CausalSuccessorBackoff
    module.r753_word_gain = nn.Parameter(torch.zeros(()))
    module.r753_geometry_gain = nn.Parameter(torch.zeros(()))
    module.r753_reliability = nn.Linear(35, 1)
    nn.init.zeros_(module.r753_reliability.weight)
    nn.init.zeros_(module.r753_reliability.bias)
    added = [(name, p) for name, p in model.named_parameters()
             if id(p) not in old]
    expected = {"word_backoff.r753_word_gain",
                "word_backoff.r753_geometry_gain",
                "word_backoff.r753_reliability.weight",
                "word_backoff.r753_reliability.bias"}
    if {name for name, _ in added} != expected:
        raise AssertionError("unexpected successor parameter topology")
    optimizer._name_by_id.update({id(p): name for name, p in added})
    optimizer._group_names.append(
        {"algorithm": "adamw", "names": [name for name, _ in added]})
    optimizer._adamw.add_param_group(
        {"params": [p for _, p in added], "lr": learning_rate,
         "weight_decay": 0.})
    optimizer._source_group_counts[-1] += 1
    optimizer.param_groups = optimizer._legacy_groups + optimizer._adamw.param_groups
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if (len(ids) != len(set(ids))
            or set(ids) != {id(p) for p in model.parameters()}
            or len(optimizer.param_groups) != 30):
        raise AssertionError("R7.53 cannot discard an inherited Adam group")
    return model, optimizer


@contextlib.contextmanager
def successor_control(model, mode):
    if mode not in CONTROLS:
        raise ValueError("unknown successor control")
    module = model.word_backoff
    old = getattr(module, "_r753_control", "full")
    module._r753_control = mode
    try:
        yield
    finally:
        module._r753_control = old
