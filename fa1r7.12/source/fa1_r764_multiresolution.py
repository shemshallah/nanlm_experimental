#!/usr/bin/env python3
"""FA1 R7.64: sparse causal multiresolution mirror memory, one unified model.

R7.63 conditioned on the exact recent word AND two 256-way heads.  That
65,536-way subaddress is sparse for most words.  This module complements it
with two 256x256 coarse crossed-head banks and two conditioned single-head
banks, all from TRAIN-only four-token windows.  The learned selector sees
support and orientation agreement but never the target.  A lexical null expert
ensures an exactly normalized backoff, including when every bank misses.
"""
from __future__ import annotations

import contextlib
import math

import numpy as np
import torch
from torch import nn

from r7_core import Corpus
from fa1_r757_geometric_transition_router import VOCAB
from fa1_r758_contextual_geometry import sparse_probability
from fa1_r763_temporal_cross import HEADS, TemporalCrossPrior

ROUND = 'FA1_R7_64_MULTISCALE_TEMPORAL_DIAMOND_V1'
PREFIXES = ('coarse_plus', 'coarse_minus', 'fine_plus', 'fine_minus')
BANK_FIELDS = tuple(f'{p}_{s}' for p in PREFIXES for s in
                    ('keys', 'counts', 'addresses', 'totals'))


def build_multires_banks(root, corpus_dir, heads):
    """No DEV/TEST reads. Exclude every window crossing a TRAIN book boundary."""
    corpus = Corpus(root, 'train', corpus_dir=corpus_dir)
    seq = np.asarray(corpus.seq, dtype=np.int64)
    start = np.asarray(corpus.doc_start)
    valid = ((start[:-3] == start[1:-2]) &
             (start[1:-2] == start[2:-1]) &
             (start[2:-1] == start[3:]) &
             (seq[:-3] != 1) & (seq[1:-2] != 1) &
             (seq[2:-1] != 1) & (seq[3:] != 1))
    third, older, recent, target = (
        seq[:-3][valid], seq[1:-2][valid], seq[2:-1][valid],
        seq[3:][valid])
    if target.size == 0:
        raise ValueError('No causal TRAIN four-word windows')
    h = heads.detach().cpu().numpy()
    h30 = h[third, 0].astype(np.int64)
    h31 = h[third, 1].astype(np.int64)
    h20 = h[older, 0].astype(np.int64)
    h21 = h[older, 1].astype(np.int64)
    address_specs = (
        ('coarse_plus', h30 * HEADS + h21),
        ('coarse_minus', h31 * HEADS + h20),
        ('fine_plus', recent * HEADS + h30),
        ('fine_minus', recent * HEADS + h21))
    result = {}
    for prefix, address in address_specs:
        keys, counts = np.unique(address * VOCAB + target,
                                 return_counts=True)
        addresses, totals = np.unique(address, return_counts=True)
        if int(counts.sum()) != len(target) or int(totals.sum()) != len(target):
            raise AssertionError('Multiresolution TRAIN mass mismatch')
        for key, arr in (('keys', keys), ('counts', counts),
                         ('addresses', addresses), ('totals', totals)):
            result[f'{prefix}_{key}'] = torch.from_numpy(
                np.asarray(arr, dtype=np.int64).copy())
    return result


class MultiresolutionPrior(TemporalCrossPrior):
    def forward(self, older, recent, target, heads, summary, exact_bank,
                word_bank, *, control='full', valid=None,
                leave_one_out=False, third_older=None, valid_third=None):
        parent_q, g, ec, geo_n, wc, k = super().forward(
            older, recent, target, heads, summary, exact_bank, word_bank,
            control=control, valid=valid, leave_one_out=leave_one_out,
            third_older=third_older, valid_third=valid_third)
        if not getattr(self, '_r764_enabled', True):
            return parent_q, g, ec, geo_n, wc, k
        if third_older is None:
            raise ValueError('R7.64 needs t-3 history')
        if valid is None:
            valid = torch.ones_like(recent, dtype=torch.bool)
        if valid_third is None:
            valid_third = valid
        eligible = valid & valid_third
        # Control must remove the actual third/second word, not corrupt the
        # target used in the inherited lexical and exact distributions.
        third_geo = ((third_older + 7919) % VOCAB
                     if control == 'wrong_older' else third_older)
        older_geo = ((older + 7919) % VOCAB
                     if control == 'wrong_older' else older)
        value = ((target + 7919) % VOCAB
                 if control == 'wrong_geometry_values' else target)
        h30 = heads[third_geo.long(), 0].long()
        h31 = heads[third_geo.long(), 1].long()
        h20 = heads[older_geo.long(), 0].long()
        h21 = heads[older_geo.long(), 1].long()
        addresses = (h30 * HEADS + h21,
                     h31 * HEADS + h20,
                     recent.long() * HEADS + h30,
                     recent.long() * HEADS + h21)
        loo = (eligible if leave_one_out and control not in
               ('wrong_older', 'wrong_geometry_values') else False)
        qs, ns = [], []
        for prefix, address in zip(PREFIXES, addresses):
            q, n = sparse_probability(
                getattr(self, prefix + '_keys'),
                getattr(self, prefix + '_counts'), address, value,
                getattr(self, prefix + '_totals'),
                total_keys=getattr(self, prefix + '_addresses'),
                leave_one_out=loo)
            qs.append(q)
            ns.append(n.float())
        qword, _ = word_bank.lookup(
            word_bank.word_keys, word_bank.word_counts,
            word_bank.word_totals, recent.long(), target,
            leave_one_out=leave_one_out)
        exact_address = older.long() * VOCAB + recent.long()
        qexact, _ = sparse_probability(
            exact_bank.exact_keys, exact_bank.exact_counts, exact_address,
            target, exact_bank.exact_totals,
            total_keys=exact_bank.exact_addresses,
            leave_one_out=valid if leave_one_out else False)
        count = torch.stack(ns, -1)
        empirical = torch.stack(qs, -1)
        tau = self.r764_log_tau.exp().clamp(1., 2048.)
        posterior = (count * empirical + tau * qword.unsqueeze(-1)) / (
            count + tau)
        features = torch.stack((
            count[:, 0].log1p(), count[:, 1].log1p(),
            count[:, 2].log1p(), count[:, 3].log1p(),
            ec.float().log1p(), wc.float().log1p(),
            (h30 == h21).float(), (h31 == h20).float()), -1)
        # Only causal/context/support features.  Selector independent of y;
        # the five-way convex combination remains normalized over vocabulary.
        selection = torch.softmax(self.r764_selector(features), -1)
        geometry = (selection[:, :4] * posterior).sum(-1) + (
            selection[:, 4] * qword)
        if control == 'lexical_matched':
            geometry = qword
        smoothed = (qexact * ec.float() + k * geometry) / (ec.float() + k)
        intensity = torch.tanh(self.r764_strength).clamp_min(0.)
        mix = intensity * eligible.float()
        if control == 'disabled':
            mix = mix * 0.
        return parent_q + mix * (smoothed - parent_q), g, ec, geo_n, wc, k


def attach_multiresolution(model, optimizer, banks, learning_rate=.001):
    if (type(model.word_backoff.r759_prior) is not TemporalCrossPrior
            or len(optimizer.param_groups) != 36):
        raise TypeError('R7.64 requires complete 36-group R7.63 checkpoint')
    previous = {id(p) for p in model.parameters()}
    prior = model.word_backoff.r759_prior
    prior.__class__ = MultiresolutionPrior
    for name in BANK_FIELDS:
        prior.register_buffer(name, banks[name].clone().long())
    prior.r764_strength = nn.Parameter(torch.zeros(()))
    prior.r764_log_tau = nn.Parameter(
        torch.tensor([math.log(64.)] * 4, dtype=torch.float32))
    prior.r764_selector = nn.Sequential(nn.Linear(8, 16), nn.Tanh(),
                                        nn.Linear(16, 5))
    nn.init.zeros_(prior.r764_selector[2].weight)
    with torch.no_grad():
        prior.r764_selector[2].bias.copy_(
            torch.tensor([0., 0., 0., 0., 2.]))
    added = [(n, p) for n, p in model.named_parameters() if id(p) not in previous]
    if len(added) != 6:
        raise AssertionError('R7.64 must add exactly six trainable tensors')
    optimizer._name_by_id.update({id(p): n for n, p in added})
    optimizer._group_names.append({
        'algorithm': 'adamw', 'names': [n for n, _ in added]})
    optimizer._adamw.add_param_group({
        'params': [p for _, p in added],
        'lr': learning_rate, 'weight_decay': 0.})
    optimizer._source_group_counts[-1] += 1
    optimizer.param_groups = (
        optimizer._legacy_groups + optimizer._adamw.param_groups)
    if len(optimizer.param_groups) != 37:
        raise AssertionError('R7.64 must have 37 optimizer groups')
    return model, optimizer


@contextlib.contextmanager
def multiresolution_control(model, enabled):
    prior = model.word_backoff.r759_prior
    old = getattr(prior, '_r764_enabled', True)
    prior._r764_enabled = enabled
    try:
        yield
    finally:
        prior._r764_enabled = old
