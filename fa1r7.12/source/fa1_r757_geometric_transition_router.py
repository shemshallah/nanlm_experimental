#!/usr/bin/env python3
"""FA1 R7.57: exact TRAIN-only lexical/geometric transition router.

Attach to one existing R7.53 successor model and one optimizer. Frozen
TRAIN-only counts live inside the same checkpoint as its learned gate.
Leave-one-out TRAIN scoring removes the supervised transition itself.
"""
from __future__ import annotations

import contextlib
import numpy as np
import torch
from torch import nn

from r7_core import Corpus
from fa1_r744_causal_word_backoff import UnifiedR744Backoff
from fa1_r748_mirror_pair_values import pair_indices
from fa1_r753_causal_successor import (
    CausalSuccessorBackoff, compose_inherited_distribution)

ROUND = "FA1_R7_57_TRAIN_ONLY_SIGNED_PAIR_ROUTER_V1"
VOCAB = 32000
PAIR_COUNT = 32640
CONTROLS = ("full", "disabled", "lexical_matched", "wrong_pair_addresses",
            "wrong_pair_values", "no_apex")
BANK_FIELDS = ("word_keys", "word_counts", "word_totals",
               "signed_keys", "signed_counts", "signed_totals")


def signed_pair_numpy(heads):
    a = heads[:, 0].astype(np.int64)
    b = heads[:, 1].astype(np.int64)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    pair = lo * (511 - lo) // 2 + hi - lo - 1
    if bool(np.any(a == b)):
        raise ValueError("R7.57 requires distinct endpoint heads")
    return 2 * pair + (a < b).astype(np.int64)


def build_transition_banks(root, corpus_dir, head_lookup):
    """TRAIN data only, skipping cross-book edges, no DEV or TEST reads."""
    corpus = Corpus(root, "train", corpus_dir=corpus_dir)
    seq = corpus.seq
    valid = ((corpus.doc_start[:-1] == corpus.doc_start[1:]) &
             (seq[:-1] != 1) & (seq[1:] != 1))
    preceding, following = seq[:-1][valid], seq[1:][valid]
    lookup = head_lookup.cpu().numpy()
    addresses = {"word": preceding.astype(np.int64),
                 "signed": signed_pair_numpy(lookup[preceding])}
    output = {}
    for name, groups in (("word", VOCAB), ("signed", PAIR_COUNT * 2)):
        addr = addresses[name]
        packed = addr * VOCAB + following.astype(np.int64)
        keys, counts = np.unique(packed, return_counts=True)
        totals = np.bincount(addr, minlength=groups).astype(np.int64)
        if counts.sum() != totals.sum() or counts.sum() != len(preceding):
            raise AssertionError("TRAIN transition counts are incomplete")
        output[name + "_keys"] = torch.from_numpy(keys.copy()).long()
        output[name + "_counts"] = torch.from_numpy(counts.copy()).long()
        output[name + "_totals"] = torch.from_numpy(totals.copy()).long()
    return output


class TransitionRouter(nn.Module):
    def __init__(self, banks):
        super().__init__()
        for field in BANK_FIELDS:
            self.register_buffer(field, banks[field].clone().long())
        self.strength = nn.Parameter(torch.zeros(()))
        self.logits = nn.Linear(35, 3)
        nn.init.zeros_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)

    @staticmethod
    def lookup(keys, counts, totals, address, target, leave_one_out=False):
        idx = torch.searchsorted(keys, address * VOCAB + target.long())
        safe = idx.clamp_max(keys.numel() - 1)
        num = torch.where(keys[safe] == address * VOCAB + target.long(),
                          counts[safe], 0)
        den = totals[address]
        if leave_one_out:
            # Every current TRAIN edge was inserted once. Exclude exactly that
            # supervised edge, both from value count and address support.
            num = (num - 1).clamp_min(0)
            den = (den - 1).clamp_min(0)
        return num.float() / den.clamp_min(1).float(), den

    def forward(self, query, target, lookup, summary, *, control="full",
                leave_one_out=False):
        if control not in CONTROLS:
            raise ValueError("unknown R7.57 control")
        if summary.shape != (len(query), 32):
            raise ValueError("R7.57 must inherit 32D causal apex summary")
        wq, wc = self.lookup(self.word_keys, self.word_counts,
                             self.word_totals, query, target, leave_one_out)
        h = lookup[query]
        pair, valid = pair_indices(h[:, 0], h[:, 1])
        signed_addr = 2 * pair + (h[:, 0] < h[:, 1]).long()
        geo_addr = (signed_addr + 7919) % (PAIR_COUNT * 2) if control == "wrong_pair_addresses" else signed_addr
        geo_target = (target + 7919) % VOCAB if control == "wrong_pair_values" else target
        gq, gc = self.lookup(self.signed_keys, self.signed_counts,
                             self.signed_totals, geo_addr, geo_target,
                             leave_one_out and control not in
                             ("wrong_pair_addresses", "wrong_pair_values"))
        gq = gq * valid.float()
        if control == "no_apex":
            summary = torch.zeros_like(summary)
        features = torch.cat((
            summary.detach(), wc.float().log1p()[:, None],
            gc.float().log1p()[:, None],
            (gc.float() / wc.float().clamp_min(1.)).log1p()[:, None]), 1)
        shares = self.logits(features).softmax(-1)
        intensity = .24 * torch.tanh(self.strength).clamp_min(0.)
        lexical = intensity * shares[:, 1]
        geometric = intensity * shares[:, 2]
        if control == "lexical_matched":
            lexical = lexical + geometric
            geometric = geometric * 0.
        if control == "disabled":
            lexical = lexical * 0.
            geometric = geometric * 0.
        return wq, gq, lexical, geometric, wc, gc


class TrainTransitionBackoff(CausalSuccessorBackoff):
    def forward(self, data, trace, targets, embeddings=None):
        alpha, q, telemetry = super().forward(data, trace, targets, embeddings)
        history = data.get("r753_successor_history_ids", data["word_ids"])
        query = history[:, -1].long()
        mode = getattr(self, "_r757_control", "full")
        wq, gq, lw, gw, wc, gc = self.r757_router(
            query, targets, self.word_head_lookup,
            trace["descended_point_summary"], control=mode,
            leave_one_out=bool(data.get("r757_leave_one_out", False)))
        composed = compose_inherited_distribution(alpha, q, wq, gq, lw, gw)
        telemetry.update({"r757_word_gain": lw.detach(),
                          "r757_signed_pair_gain": gw.detach(),
                          "r757_word_support": wc.detach(),
                          "r757_pair_support": gc.detach()})
        return (*composed, telemetry)


def attach_transition_router(model, optimizer, banks, learning_rate=5.e-3):
    if (type(model) is not UnifiedR744Backoff or
            type(model.word_backoff) is not CausalSuccessorBackoff or
            len(optimizer.param_groups) != 30):
        raise TypeError("R7.57 requires original full 30-group R7.53")
    old = {id(p) for p in model.parameters()}
    module = model.word_backoff
    module.__class__ = TrainTransitionBackoff
    module.r757_router = TransitionRouter(banks)
    new = [(n, p) for n, p in model.named_parameters() if id(p) not in old]
    expected = {"word_backoff.r757_router.strength",
                "word_backoff.r757_router.logits.weight",
                "word_backoff.r757_router.logits.bias"}
    if {n for n, _ in new} != expected:
        raise AssertionError("R7.57 optimizer topology inconsistent")
    optimizer._name_by_id.update({id(p): n for n, p in new})
    optimizer._group_names.append({"algorithm": "adamw",
                                    "names": [n for n, _ in new]})
    optimizer._adamw.add_param_group({"params": [p for _, p in new],
                                       "lr": learning_rate,
                                       "weight_decay": 0.})
    optimizer._source_group_counts[-1] += 1
    optimizer.param_groups = optimizer._legacy_groups + optimizer._adamw.param_groups
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if len(ids) != len(set(ids)) or len(optimizer.param_groups) != 31:
        raise AssertionError("R7.57 lost optimizer history")
    return model, optimizer


@contextlib.contextmanager
def transition_control(model, mode):
    if mode not in CONTROLS:
        raise ValueError("unknown transition control")
    module = model.word_backoff
    old = getattr(module, "_r757_control", "full")
    module._r757_control = mode
    try:
        yield
    finally:
        module._r757_control = old
