#!/usr/bin/env python3
"""FA1 R7.58: compositional temporal mirror memory, inheriting all R7.57 state."""
from __future__ import annotations

import contextlib
import numpy as np
import torch
from torch import nn

from r7_core import Corpus
from fa1_r744_causal_word_backoff import UnifiedR744Backoff
from fa1_r753_causal_successor import compose_inherited_distribution
from fa1_r757_geometric_transition_router import TrainTransitionBackoff, VOCAB

ROUND = 'FA1_R7_58_CAUSAL_COMPOSITIONAL_MIRROR_V1'
HEADS = 256
ADDRESSES = 65536
BANK_FIELDS = ('geo_keys', 'geo_counts', 'geo_totals',
               'exact_keys', 'exact_counts', 'exact_addresses', 'exact_totals')
CONTROLS = ('full', 'disabled', 'exact_matched', 'geometry_matched',
            'wrong_older', 'mirror_swapped', 'wrong_geo_values', 'no_apex')


def temporal_address(older, recent, lookup):
    """Oriented 256x256 relation from *two different* observed words."""
    return lookup[older.long(), 1].long() * HEADS + lookup[recent.long(), 0].long()


def build_context_banks(root, corpus_dir, lookup):
    """Sparse normalized TRAIN-only triple banks; never cross book boundaries."""
    corpus = Corpus(root, 'train', corpus_dir=corpus_dir)
    seq = np.asarray(corpus.seq, dtype=np.int64)
    starts = np.asarray(corpus.doc_start)
    valid = ((starts[:-2] == starts[1:-1]) &
             (starts[1:-1] == starts[2:]) &
             (seq[:-2] != 1) & (seq[1:-1] != 1) & (seq[2:] != 1))
    older, recent, target = seq[:-2][valid], seq[1:-1][valid], seq[2:][valid]
    if not len(older): raise ValueError('TRAIN contains no causal triples')
    heads = lookup.detach().cpu().numpy()
    geo = heads[older, 1].astype(np.int64) * HEADS + heads[recent, 0]
    exact = older * VOCAB + recent
    geo_keys, geo_counts = np.unique(geo * VOCAB + target, return_counts=True)
    exact_keys, exact_counts = np.unique(exact * VOCAB + target, return_counts=True)
    exact_addresses, exact_totals = np.unique(exact, return_counts=True)
    geo_totals = np.bincount(geo, minlength=ADDRESSES)
    if not (geo_counts.sum() == exact_counts.sum() == exact_totals.sum() ==
            geo_totals.sum() == len(target)):
        raise AssertionError('TRAIN triple count mass mismatch')
    values = locals()
    return {key: torch.from_numpy(np.asarray(values[key], dtype=np.int64).copy())
            for key in BANK_FIELDS}


def sparse_probability(keys, counts, address, target, totals, *,
                       total_keys=None, leave_one_out=False):
    packed = address.long() * VOCAB + target.long()
    idx = torch.searchsorted(keys, packed)
    safe = idx.clamp_max(keys.numel() - 1)
    numerator = torch.where((idx < keys.numel()) & (keys[safe] == packed),
                            counts[safe], torch.zeros_like(packed))
    if total_keys is None:
        denominator = totals[address.long()]
    else:
        at = torch.searchsorted(total_keys, address.long())
        sat = at.clamp_max(total_keys.numel() - 1)
        denominator = torch.where((at < total_keys.numel()) &
                                  (total_keys[sat] == address.long()),
                                  totals[sat], torch.zeros_like(packed))
    mask = (leave_one_out.bool() if torch.is_tensor(leave_one_out) else
            torch.full_like(packed, bool(leave_one_out), dtype=torch.bool))
    if bool(mask.any()):
        if not bool((numerator[mask] > 0).all()) or not bool((denominator[mask] > 0).all()):
            raise AssertionError('missing supervised TRAIN edge for leave-one-out')
        numerator -= mask.long()
        denominator -= mask.long()
    return numerator.float() / denominator.clamp_min(1).float(), denominator


class ContextualGeometry(nn.Module):
    def __init__(self, banks):
        super().__init__()
        for name in BANK_FIELDS:
            self.register_buffer(name, banks[name].clone().long())
        self.strength = nn.Parameter(torch.zeros(()))
        self.logits = nn.Linear(35, 3)
        nn.init.zeros_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)

    def forward(self, older, recent, target, lookup, summary, *,
                control='full', leave_one_out=False, valid=None):
        if control not in CONTROLS: raise ValueError('unrecognized control')
        if summary.shape != (len(recent), 32):
            raise ValueError('R7.58 requires 32D inherited apex')
        if valid is None: valid = torch.ones_like(recent, dtype=torch.bool)
        if valid.shape != recent.shape: raise ValueError('causal mask shape mismatch')
        exact_address = older.long() * VOCAB + recent.long()
        alt_older = ((older + 7919) % VOCAB if control == 'wrong_older' else older)
        geo_address = temporal_address(alt_older, recent, lookup)
        if control == 'mirror_swapped':
            geo_address = (geo_address % HEADS) * HEADS + geo_address // HEADS
        geo_target = ((target + 7919) % VOCAB if control == 'wrong_geo_values'
                      else target)
        geo_loo = (valid if leave_one_out and control not in
                   ('wrong_older', 'mirror_swapped', 'wrong_geo_values') else False)
        gq, gc = sparse_probability(self.geo_keys, self.geo_counts,
                                    geo_address, geo_target, self.geo_totals,
                                    leave_one_out=geo_loo)
        eq, ec = sparse_probability(self.exact_keys, self.exact_counts,
                                    exact_address, target, self.exact_totals,
                                    total_keys=self.exact_addresses,
                                    leave_one_out=valid if leave_one_out else False)
        gq, eq = gq * valid.float(), eq * valid.float()
        gc, ec = torch.where(valid, gc, 0), torch.where(valid, ec, 0)
        if control == 'no_apex': summary = torch.zeros_like(summary)
        features = torch.cat((summary.detach(), ec.float().log1p()[:, None],
                              gc.float().log1p()[:, None],
                              (gc.float()/ec.float().clamp_min(1)).log1p()[:, None]), 1)
        shares = self.logits(features).softmax(-1)
        intensity = .18 * torch.tanh(self.strength).clamp_min(0.)
        ew, gw = intensity * shares[:, 1] * valid.float(), intensity * shares[:, 2] * valid.float()
        if control == 'exact_matched': ew, gw = ew + gw, gw * 0.
        elif control == 'geometry_matched': gw, ew = ew + gw, ew * 0.
        elif control == 'disabled': ew, gw = ew * 0., gw * 0.
        return eq, gq, ew, gw, ec, gc


class ContextualGeometryBackoff(TrainTransitionBackoff):
    def forward(self, data, trace, targets, embeddings=None):
        alpha, q, telemetry = super().forward(data, trace, targets, embeddings)
        history = data.get('r753_successor_history_ids', data['word_ids'])
        mask = data.get('r753_successor_history_mask')
        if history.shape[1] < 2: raise ValueError('need two prior words')
        if mask is None: mask = history != 1
        valid = (mask[:, -2] & mask[:, -1] & (history[:, -2] != 1)
                 & (history[:, -1] != 1))
        eq, gq, ew, gw, ec, gc = self.r758_context(
            history[:, -2].long(), history[:, -1].long(), targets,
            self.word_head_lookup, trace['descended_point_summary'],
            control=getattr(self, '_r758_control', 'full'),
            leave_one_out=bool(data.get('r758_leave_one_out', False)), valid=valid)
        a, p = compose_inherited_distribution(alpha, q, eq, gq, ew, gw)
        telemetry.update({'r758_exact_gain': ew.detach(),
                          'r758_geometry_gain': gw.detach(),
                          'r758_exact_support': ec.detach(),
                          'r758_geometry_support': gc.detach()})
        return a, p, telemetry


def attach_contextual_geometry(model, optimizer, banks, learning_rate=.003):
    if (type(model) is not UnifiedR744Backoff or
            type(model.word_backoff) is not TrainTransitionBackoff or
            len(optimizer.param_groups) != 31):
        raise TypeError('R7.58 requires intact 31-group R7.57 model')
    original = {id(p) for p in model.parameters()}
    backoff = model.word_backoff
    backoff.__class__ = ContextualGeometryBackoff
    backoff.r758_context = ContextualGeometry(banks)
    added = [(n, p) for n, p in model.named_parameters() if id(p) not in original]
    expected = {'word_backoff.r758_context.strength',
                'word_backoff.r758_context.logits.weight',
                'word_backoff.r758_context.logits.bias'}
    if {n for n, _ in added} != expected: raise AssertionError('unexpected new parameters')
    optimizer._name_by_id.update({id(p): n for n, p in added})
    optimizer._group_names.append({'algorithm': 'adamw', 'names': [n for n, _ in added]})
    optimizer._adamw.add_param_group({'params': [p for _, p in added],
                                      'lr': learning_rate, 'weight_decay': 0.})
    optimizer._source_group_counts[-1] += 1
    optimizer.param_groups = optimizer._legacy_groups + optimizer._adamw.param_groups
    ids = [id(p) for g in optimizer.param_groups for p in g['params']]
    if len(optimizer.param_groups) != 32 or len(ids) != len(set(ids)):
        raise AssertionError('lost optimizer group/history')
    return model, optimizer


@contextlib.contextmanager
def contextual_control(model, mode):
    if mode not in CONTROLS: raise ValueError('unknown contextual control')
    backoff = model.word_backoff
    old = getattr(backoff, '_r758_control', 'full')
    backoff._r758_control = mode
    try: yield
    finally: backoff._r758_control = old
