#!/usr/bin/env python3
"""FA1 R7.51: bounded bidirectional evidence at exact C(256,2) addresses.

One in-place extension of the complete R7.49 checkpoint. All features and
routing choices are pre-target; inherited model/optimizer state stays intact.
The non-address temperature is a matched calibration control. The two memory
banks store nonnegative support and opposition; their difference may have
either sign. These are NOT additional language models.
"""
from __future__ import annotations

import contextlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from fa1_r744_causal_word_backoff import UnifiedR744Backoff
from fa1_r748_mirror_pair_values import PAIR_COUNT, pair_indices
from fa1_r749_address_reliability import ReliableAddressedMirrorBackoff


ROUND = "FA1_R7_51_BIDIRECTIONAL_PAIR_EVIDENCE_V1"
CONTROLS = ("full", "disabled", "calibration_only", "constant_bank",
            "permute_pair_values", "fixed_confidence")


def causal_confidence_features(data, trace, candidate_ids):
    """[B,35] of pre-target apex, historical repetition, coverage, entropy."""
    summary = trace["descended_point_summary"]
    if summary.ndim != 2 or summary.shape != (candidate_ids.size(0), 32):
        raise ValueError("R7.51 requires the causal 32D apex descent")
    history = candidate_ids.long()
    if history.size(1) > 1:
        ordered = history.sort(dim=1).values
        repetition = (ordered[:, 1:] == ordered[:, :-1]).float().mean(1)
    else:
        repetition = torch.zeros(history.size(0), device=history.device)
    mask = data.get("diamond_backoff_recent_mask")
    if mask is not None:
        coverage = mask.float().mean(1)
    else:
        recent = data.get("diamond_backoff_recent_ids", data.get("word_ids"))
        coverage = ((recent != 1).float().mean(1) if recent is not None
                    else torch.ones_like(repetition))
    attention = trace.get("triangle_attention")
    if attention is not None and attention.ndim == 2 and attention.size(1) > 1:
        attn = attention.detach().float().clamp_min(1.e-12)
        entropy = -(attn * attn.log()).sum(1)
        concentration = (1. - entropy / math.log(attn.size(1))).clamp(0., 1.)
    else:
        concentration = torch.zeros_like(repetition)
    return torch.cat((summary, coverage[:, None],
                      repetition[:, None], concentration[:, None]), dim=1)


class BidirectionalPairEvidenceBackoff(ReliableAddressedMirrorBackoff):
    def pair_logits(self, data, trace, candidate_ids):
        logits, info = super().pair_logits(data, trace, candidate_ids)
        mode = getattr(self, "_r751_control", "full")
        if mode not in CONTROLS:
            raise ValueError("unknown R7.51 intervention")
        heads = self.word_head_lookup[candidate_ids.long()]
        address, valid = pair_indices(heads[..., 0], heads[..., 1])
        active = data["vertex_ids"].long()
        edges = torch.triu_indices(8, 8, 1, device=active.device)
        active_address, _ = pair_indices(active[:, edges[0]], active[:, edges[1]])
        exact = (address[..., None] == active_address[:, None, :]).any(-1) & valid

        # A pure score-temperature parameter can work even if NO per-address
        # information helps. Always compare against this fitted baseline.
        temperature = .5 * torch.tanh(self.mirror_r751_temperature)
        if mode == "disabled":
            temperature = temperature * 0.
        calibrated = logits + temperature * logits

        address_gain = .5 * torch.tanh(self.mirror_r751_address_gain)
        if mode in ("disabled", "calibration_only"):
            address_gain = address_gain * 0.
        lookup = ((address + 7919) % PAIR_COUNT
                  if mode == "permute_pair_values" else address)
        positive = F.softplus(self.mirror_r751_positive(lookup).squeeze(-1))
        negative = F.softplus(self.mirror_r751_negative(lookup).squeeze(-1))
        signed = positive - negative
        if mode == "constant_bank":
            signed = torch.zeros_like(signed)
        # The learned global signed offset is intentionally NOT an address
        # value. The constant-bank control includes it to test attribution.
        signed = torch.tanh(signed + self.mirror_r751_global_pair_bias)
        features = causal_confidence_features(data, trace, candidate_ids)
        trust = torch.sigmoid(self.mirror_r751_context(features).squeeze(-1))
        if mode == "fixed_confidence":
            trust = torch.full_like(trust, .5)
        delta = (address_gain * trust[:, None] * signed * exact.float())
        output = (calibrated + delta).masked_fill(~valid, 0.)
        info.update({
            "r751_exact_hit_fraction": exact.float().mean(1).detach(),
            "r751_copy_confidence": features[:, -2].detach(),
            "r751_context_trust": trust.detach(),
            "r751_signed_delta": delta.mean(1).detach(),
            "r751_temperature": temperature.detach(),
            "r751_address_gain": address_gain.detach(),
        })
        return output, info


def attach_bidirectional_pair_evidence(model, optimizer, learning_rate=2.e-4):
    if (type(model) is not UnifiedR744Backoff or
            type(model.word_backoff) is not ReliableAddressedMirrorBackoff or
            len(optimizer.param_groups) != 29):
        raise TypeError("R7.51 requires the complete 29-group R7.49 lineage")
    inherited = {id(p) for p in model.parameters()}
    module = model.word_backoff
    module.__class__ = BidirectionalPairEvidenceBackoff
    module.mirror_r751_positive = nn.Embedding(PAIR_COUNT, 1)
    module.mirror_r751_negative = nn.Embedding(PAIR_COUNT, 1)
    module.mirror_r751_context = nn.Linear(35, 1)
    module.mirror_r751_temperature = nn.Parameter(torch.zeros(()))
    module.mirror_r751_global_pair_bias = nn.Parameter(torch.zeros(()))
    module.mirror_r751_address_gain = nn.Parameter(torch.zeros(()))
    nn.init.zeros_(module.mirror_r751_positive.weight)
    nn.init.zeros_(module.mirror_r751_negative.weight)
    nn.init.zeros_(module.mirror_r751_context.weight)
    nn.init.zeros_(module.mirror_r751_context.bias)
    new = [(name, p) for name, p in model.named_parameters()
           if id(p) not in inherited]
    expected = {
        "word_backoff.mirror_r751_positive.weight",
        "word_backoff.mirror_r751_negative.weight",
        "word_backoff.mirror_r751_context.weight",
        "word_backoff.mirror_r751_context.bias",
        "word_backoff.mirror_r751_temperature",
        "word_backoff.mirror_r751_global_pair_bias",
        "word_backoff.mirror_r751_address_gain",
    }
    if {name for name, _ in new} != expected:
        raise AssertionError("unexpected R7.51 parameter topology")
    optimizer._name_by_id.update({id(p): name for name, p in new})
    optimizer._group_names.append({"algorithm": "adamw",
                                   "names": [name for name, _ in new]})
    optimizer._adamw.add_param_group({
        "params": [p for _, p in new], "lr": learning_rate,
        "weight_decay": 0.,
    })
    optimizer._source_group_counts[-1] += 1
    optimizer.param_groups = optimizer._legacy_groups + optimizer._adamw.param_groups
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if (len(ids) != len(set(ids)) or len(optimizer.param_groups) != 30 or
            set(ids) != {id(p) for p in model.parameters()}):
        raise AssertionError("R7.51 cannot discard any inherited Adam group")
    return model, optimizer


@contextlib.contextmanager
def bidirectional_control(model, mode="calibration_only"):
    if mode not in CONTROLS or mode == "full":
        raise ValueError("unknown R7.51 control")
    module = model.word_backoff
    old = getattr(module, "_r751_control", "full")
    module._r751_control = mode
    try:
        yield
    finally:
        module._r751_control = old
