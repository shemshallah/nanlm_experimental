#!/usr/bin/env python3
"""FA1 R7.28: causally train the existing R7.27 geometric address memory.

This is a continuation of ONE R7 model/optimizer, not a standalone geometry
model.  The learned full C(256,3) triangle-address embeddings, signed S+/S-
face codes, tetra-conditioned 256 sparse specialists and descending pair/point
path all receive next-word gradients.  Every score is compared on held-out
book-disjoint examples; no automatic checkpoint promotion is permitted.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from fa1_r723_train import forward, prepare, take
from fa1_r724_train import GEOMETRY_GATES, boot_books
from fa1_r725_controls import unrelated_book_context
from fa1_r727_checkpoint import digest, load_unified, save_unpromoted
from fa1_r727_eval import evaluate as vocabulary_evaluate, validate_book_corpus
from fa1_r732_matched_history import (matched_history_context,
                                       geometry_specific_history_gain,
                                       context_specific_margin)
from fa1_r733_far_history import HORIZONS, matched_far_history


@contextlib.contextmanager
def geometry_disabled(model):
    """Counterfactual: only disable new geometric channels, never the trunk."""
    names = list(GEOMETRY_GATES)
    if hasattr(model.diamond.geometry,'context_read_gain'):
        names.append('context_read_gain')
    if hasattr(model.diamond.geometry,'factor_up_gain'):
        names.extend(('factor_up_gain','factor_decoder_gain'))
    gates = [getattr(model.diamond.geometry, n) for n in names]
    before = [p.detach().clone() for p in gates]
    layer_gates=[]
    if hasattr(model.diamond,'hierarchy'):
        h=model.diamond.hierarchy
        # The layered context_gain is a *language* byte/subword/token/
        # sentence path, not a geometric read. Do not disable it in a
        # geometry-only ablation: that would confound the measured effect.
        layer_gates=[getattr(h,n) for n in ('point_gain','pair_gain',
                            'triangle_gain','word_gain','byte_geometry_gain')]
    original_layers=[p.detach().clone() for p in layer_gates]
    with torch.no_grad():
        for param in gates+layer_gates:
            param.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            for p, value in zip(gates, before):
                p.copy_(value)
            for p, value in zip(layer_gates, original_layers):
                p.copy_(value)


def layered_byte_losses(extra, data):
    """Per-word mean byte loss and true byte counts, excluding EOW from BPB."""
    logits=extra['byte_ar_logits']
    labels=data['byte_target_sequence']
    ce=F.cross_entropy(logits.transpose(1,2),labels,ignore_index=-100,reduction='none')
    valid=(labels>=0)&(labels<256)
    counts=valid.sum(-1)
    # Exclude EOW from the literal UTF-8 byte denominator, but score it
    # separately in the total conditional word likelihood.
    ce_bytes=(ce*valid).sum(-1)
    per_word=ce_bytes/counts.clamp_min(1)
    eos=(ce*(labels==256)).sum(-1)
    return per_word,counts,ce_bytes,eos,valid


def layered_byte_metrics(extra, data):
    per_word,counts,ce_bytes,eos,valid=layered_byte_losses(extra,data)
    scored=counts>0
    if not bool(scored.any()):
        raise ValueError('no valid UTF-8 bytes for R7.31 diagnostics')
    pred=extra['byte_ar_logits'].topk(5,dim=-1).indices
    labels=data['byte_target_sequence']
    correct=(pred[:,:,0]==labels)&valid
    top5=(pred==labels.unsqueeze(-1)).any(-1)&valid
    nbytes=int(counts.sum())
    return {'conditional_spelling_bits_per_byte':float(ce_bytes.sum()/counts.sum()/math.log(2)),
            'conditional_spelling_top1':float(correct.sum()/counts.sum()),
            'conditional_spelling_top5':float(top5.sum()/counts.sum()),
            'conditional_spelling_eow_nll_per_word':float(eos[scored].mean()),
            'conditional_spelling_bytes':nbytes,
            'conditional_spelling_words':int(scored.sum()),
            'conditional_spelling_bytes_per_word':nbytes/int(scored.sum()),
            'raw_contiguous_text_bits_per_byte':None,
            'metric_scope':'autoregressive UTF-8 spelling conditioned on past word context, EOW excluded from BPB'}


@contextlib.contextmanager
def contextual_read_disabled(model):
    """Disable only the new query-specific geometric read; keep prior S+/S-."""
    geometry=model.diamond.geometry
    if not hasattr(geometry,'context_read_gain'):
        yield
        return
    p=geometry.context_read_gain
    before=p.detach().clone()
    with torch.no_grad():p.zero_()
    try:yield
    finally:
        with torch.no_grad():p.copy_(before)


@contextlib.contextmanager
def layered_language_context_disabled(model):
    """Complementary control: ablate just the new language context loop."""
    if not hasattr(model.diamond, 'hierarchy'):
        yield
        return
    p = model.diamond.hierarchy.context_gain
    before = p.detach().clone()
    with torch.no_grad():p.zero_()
    try:yield
    finally:
        with torch.no_grad():p.copy_(before)


@contextlib.contextmanager
def factor_read_disabled(model):
    """Counterfactual: new R7.30 read off; inherited R7.29 path stays on."""
    geo=model.diamond.geometry
    if not hasattr(geo,'factor_up_gain'):
        yield
        return
    gates=(geo.factor_up_gain,geo.factor_decoder_gain)
    saved=[p.detach().clone() for p in gates]
    with torch.no_grad():
        for p in gates:p.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            for p,v in zip(gates,saved):p.copy_(v)


@torch.no_grad()
def geometry_diagnostic(model, data, seed, with_topk=False, base=None,
                        with_matched_far=False):
    model.eval()
    correct, extra = forward(model, data)
    # Hold the latest local context fixed: only older words/bytes, distant
    # sentence heads and far-memory channels are replaced by another book.
    # This is intentionally harder than a total-history replacement.
    matched = None
    if hasattr(model.diamond, 'hierarchy'):
        matched, matched_donors = matched_history_context(data, seed+101)
        matched_nll, matched_extra = forward(model, matched)
    with geometry_disabled(model):
        without_geometry, no_geo_extra = forward(model, data)
        if matched is not None:
            matched_off_nll, matched_off_extra = forward(model, matched)
    if hasattr(model.diamond, 'hierarchy'):
        with layered_language_context_disabled(model):
            no_layer_context, no_layer_context_extra = forward(model, data)
    else:
        no_layer_context, no_layer_context_extra = correct, extra
    if hasattr(model.diamond.geometry,'context_read_gain'):
        with contextual_read_disabled(model):
            without_read, no_read_extra = forward(model,data)
    else:
        without_read, no_read_extra = correct, extra
    if hasattr(model.diamond.geometry,'factor_up_gain'):
        with factor_read_disabled(model):
            without_factor,no_factor_extra=forward(model,data)
    else:
        without_factor,no_factor_extra=correct,extra
    foreign, donors = unrelated_book_context(data, seed)
    foreign_nll, foreign_extra = forward(model, foreign)
    no_apex, no_apex_extra = forward(model, data, 'no_apex')
    no_pair, no_pair_extra = forward(model, data, 'no_pair')
    far_horizons=[]
    if with_matched_far:
        for horizon,(nw,nb,ns) in enumerate(HORIZONS):
            counterfactual, far_donors = matched_far_history(data,seed+19+horizon,horizon)
            far_word, far_extra = forward(model,counterfactual)
            with geometry_disabled(model):
                far_geometry_off, far_off_extra = forward(model,counterfactual)
            far_horizons.append((horizon,nw,nb,ns,far_donors,far_word,
                                 far_extra,far_geometry_off,far_off_extra))
    old = model.joint_nll(data, diamond_enabled=False)[0]
    gain = without_geometry - correct
    history = foreign_nll - correct
    result = {
        'full_word_nll': float(correct.mean()),
        'inherited_word_nll': float(old.mean()),
        'geometry_off_word_nll': float(without_geometry.mean()),
        'geometry_word_gain': float(gain.mean()),
        'diamond_word_gain': float((old-correct).mean()),
        'correct_book_advantage_nats_per_word': float(history.mean()),
        'apex_word_gain': float((no_apex-correct).mean()),
        'pair_word_gain': float((no_pair-correct).mean()),
        'branch_word_nll': float(extra['branch_nll'].mean()),
        'branch_geometry_gain': float((no_geo_extra['branch_nll']-extra['branch_nll']).mean()),
        'branch_correct_book_advantage': float((foreign_extra['branch_nll']-extra['branch_nll']).mean()),
        'branch_apex_gain': float((no_apex_extra['branch_nll']-extra['branch_nll']).mean()),
        'branch_pair_gain': float((no_pair_extra['branch_nll']-extra['branch_nll']).mean()),
        'geometry_gate_values': {n: float(getattr(model.diamond.geometry,n).detach())
                                 for n in GEOMETRY_GATES},
        'diamond_bridge_gate': float(model.diamond.gate),
        'mean_bridge_weight': float(extra['bridge_weight'].mean()),
        'unique_triangle_addresses': int(extra['trace']['triangle_addresses'].unique().numel()),
        'unique_routed_apex_specialists': int(extra['trace']['selected_sectors'].unique().numel()),
        'foreign_book_donors': int(len(donors)),
    }
    if hasattr(model.diamond.geometry,'context_read_gain'):
        context_gain=without_read-correct
        result.update({
          'contextual_read_gate':float(model.diamond.geometry.context_read_gain.detach()),
          'contextual_read_word_gain':float(context_gain.mean()),
          'contextual_read_branch_gain':float((no_read_extra['branch_nll']-extra['branch_nll']).mean()),
          'contextual_read_book_gain':boot_books(context_gain,data['book_ids'],seed+4),
        })
    if hasattr(model.diamond.geometry,'factor_up_gain'):
        factor_gain=without_factor-correct
        branch_factor_gain=no_factor_extra['branch_nll']-extra['branch_nll']
        result.update({
          'factor_up_gate':float(model.diamond.geometry.factor_up_gain.detach()),
          'factor_decoder_gate':float(model.diamond.geometry.factor_decoder_gain.detach()),
          'factor_word_gain':float(factor_gain.mean()),
          'factor_branch_gain':float(branch_factor_gain.mean()),
          'factor_word_gain_bootstrap':boot_books(factor_gain,data['book_ids'],seed+5),
          'factor_branch_gain_bootstrap':boot_books(branch_factor_gain,data['book_ids'],seed+6),
          'branch_vs_mixture_gap':float((extra['branch_nll']-correct).mean()),
          'branch_beats_inherited_fraction':float((extra['branch_nll']<old).float().mean()),
        })
    if matched is not None:
        matched_geo = geometry_specific_history_gain(
            correct, matched_nll, without_geometry, matched_off_nll)
        matched_branch_geo = geometry_specific_history_gain(
            extra['branch_nll'], matched_extra['branch_nll'],
            no_geo_extra['branch_nll'], matched_off_extra['branch_nll'])
        result.update({
            'matched_recent_history_word_advantage':float((matched_nll-correct).mean()),
            'matched_geometry_word_advantage':float(matched_geo.mean()),
            'matched_geometry_branch_advantage':float(matched_branch_geo.mean()),
            'matched_geometry_word_book_bootstrap':boot_books(matched_geo,data['book_ids'],seed+111),
            'matched_geometry_branch_book_bootstrap':boot_books(matched_branch_geo,data['book_ids'],seed+112),
            'matched_foreign_book_donors':int(len(matched_donors)),
        })
    if 'byte_ar_logits' in extra:
        metrics=layered_byte_metrics(extra,data)
        full_per_word,counts,full_bytes,_,_=layered_byte_losses(extra,data)
        no_geo_per_word,_,no_geo_bytes,_,_=layered_byte_losses(no_geo_extra,data)
        no_apex_per_word,_,no_apex_bytes,_,_=layered_byte_losses(no_apex_extra,data)
        no_pair_per_word,_,no_pair_bytes,_,_=layered_byte_losses(no_pair_extra,data)
        delta=no_geo_per_word-full_per_word
        valid=counts>0
        metrics.update({
            # Global byte-level gains must weight long spellings by their
            # byte counts; book bootstrap still uses per-word observations.
            'conditional_byte_geometry_gain_nats_per_byte':float((no_geo_bytes-full_bytes).sum()/counts.sum()),
            'conditional_byte_apex_gain_nats_per_byte':float((no_apex_bytes-full_bytes).sum()/counts.sum()),
            'conditional_byte_pair_gain_nats_per_byte':float((no_pair_bytes-full_bytes).sum()/counts.sum()),
            'conditional_byte_geometry_gain_nats_per_word_normalized':float(delta[valid].mean()),
            'conditional_byte_geometry_gain_book_bootstrap':boot_books(delta[valid],data['book_ids'][valid],seed+7),
        })
        if matched is not None:
            matched_byte,_,matched_ce,_,_=layered_byte_losses(matched_extra,data)
            matched_off_byte,_,matched_off_ce,_,_=layered_byte_losses(matched_off_extra,data)
            matched_byte_gain=geometry_specific_history_gain(
                full_per_word,matched_byte,no_geo_per_word,matched_off_byte)
            matched_byte_gain_total=geometry_specific_history_gain(
                full_bytes,matched_ce,no_geo_bytes,matched_off_ce)
            metrics.update({
                'matched_history_byte_advantage_nats_per_byte':float(
                    (matched_ce-full_bytes).sum()/counts.sum()),
                'matched_geometry_byte_advantage_nats_per_byte':float(
                    matched_byte_gain_total.sum()/counts.sum()),
                'matched_geometry_byte_book_bootstrap':boot_books(
                    matched_byte_gain[valid],data['book_ids'][valid],seed+113),
            })
        result.update(metrics)
        if hasattr(model.diamond, 'hierarchy'):
            context_per_word,_,context_ce,_,_=layered_byte_losses(no_layer_context_extra,data)
            result.update({
                'layered_language_context_word_gain':float((no_layer_context-correct).mean()),
                'layered_language_context_byte_gain_nats_per_byte':float((context_ce-full_bytes).sum()/counts.sum()),
                'layered_language_context_byte_gain_nats_per_word_normalized':float((context_per_word-full_per_word)[valid].mean()),
            })
        if with_matched_far:
            byte_advantages=[]
            byte_specificities=[]
            result['matched_far_horizons']=[{} for _ in HORIZONS]
            for horizon,nw,nb,ns,far_donors,far_word,far_extra,far_geometry_off,far_off_extra in far_horizons:
                far_per_word,_,far_bytes,_,_=layered_byte_losses(far_extra,data)
                far_off_per_word,_,far_off_bytes,_,_=layered_byte_losses(far_off_extra,data)
                byte_adv=far_per_word-full_per_word
                byte_specific=(no_geo_per_word-full_per_word)-(far_off_per_word-far_per_word)
                byte_advantages.append(byte_adv)
                byte_specificities.append(byte_specific)
                result['matched_far_horizons'][horizon].update({
                    'byte_advantage_nats_per_byte':float((far_bytes-full_bytes).sum()/counts.sum()),
                    'byte_geometry_specificity_nats_per_byte':float(((no_geo_bytes-full_bytes)-(far_off_bytes-far_bytes)).sum()/counts.sum()),
                })
            # Per-example worst horizon: an easy donor/horizon cannot hide a
            # negative geometry contribution on another distant-history scale.
            byte_advantage=torch.stack(byte_advantages).min(0).values
            geo_specific=torch.stack(byte_specificities).min(0).values
            result.update({
                'matched_far_byte_history_advantage_nats_per_byte':float((byte_advantage*counts).sum()/counts.sum()),
                'matched_far_byte_geometry_specificity_nats_per_byte':float((geo_specific*counts).sum()/counts.sum()),
                'matched_far_byte_history_bootstrap':boot_books(byte_advantage[valid],data['book_ids'][valid],seed+23),
                'matched_far_byte_geometry_bootstrap':boot_books(geo_specific[valid],data['book_ids'][valid],seed+24),
            })
    if with_matched_far:
        word_advantages=[]
        word_specificities=[]
        horizon_stats=[]
        for horizon,nw,nb,ns,far_donors,far_word,far_extra,far_geometry_off,far_off_extra in far_horizons:
            word_adv=far_word-correct
            word_specific=(without_geometry-correct)-(far_geometry_off-far_word)
            word_advantages.append(word_adv)
            word_specificities.append(word_specific)
            horizon_stat=(result['matched_far_horizons'][horizon]
                          if 'matched_far_horizons' in result else {})
            horizon_stat.update({'horizon':horizon,'recent_words':nw,'recent_bytes':nb,
                'recent_sentences':ns,'word_advantage_nats_per_word':float(word_adv.mean()),
                'geometry_specificity_nats_per_word':float(word_specific.mean()),
                'donors':int(len(far_donors))})
            horizon_stats.append(horizon_stat)
        far_history_advantage=torch.stack(word_advantages).min(0).values
        far_geom_specific=torch.stack(word_specificities).min(0).values
        result.update({
            'matched_far_history_advantage_nats_per_word':float(far_history_advantage.mean()),
            'matched_far_geometry_specificity_nats_per_word':float(far_geom_specific.mean()),
            'matched_far_history_bootstrap':boot_books(far_history_advantage,data['book_ids'],seed+21),
            'matched_far_geometry_bootstrap':boot_books(far_geom_specific,data['book_ids'],seed+22),
            'matched_far_donor_count':sum(int(len(h[4])) for h in far_horizons),
            'matched_far_horizons':horizon_stats,
        })
    if 'book_ids' in data:
        result['book_geometry_gain'] = boot_books(gain, data['book_ids'], seed)
        result['book_correct_history'] = boot_books(history, data['book_ids'], seed+1)
    if with_topk:
        assert base is not None
        result['prediction_metrics'] = vocabulary_evaluate(model, base, data, seed+2)
    return result


def soft_margin(positive, negative, margin):
    """Contrast causal controls *per example*, so rare wins cannot mask failures."""
    return (F.softplus(8. * (margin - (negative-positive))) / 8.).mean()


def open_candidate_gates(model, bridge, meta, contextual_init=0.,
                         factor_up_init=0.,factor_decoder_init=0.):
    """A candidate-only gate warm start; the saved R7.27 parent is untouched."""
    with torch.no_grad():
        model.diamond.gate.fill_(math.atanh(bridge))
        geometry = model.diamond.geometry
        if abs(float(geometry.meta_gate)) < 1.e-9:
            geometry.meta_gate.fill_(math.atanh(meta))
        if hasattr(geometry,'context_read_gain'):
            geometry.context_read_gain.fill_(math.atanh(contextual_init))
        if hasattr(geometry,'factor_up_gain'):
            geometry.factor_up_gain.fill_(math.atanh(factor_up_init))
            geometry.factor_decoder_gain.fill_(math.atanh(factor_decoder_init))


def warm_start_only_if_migrated(model, args, loaded_meta):
    """Never reset learned gate values when resuming R7.31/R7.32."""
    resuming = (getattr(args, 'layered', False) and loaded_meta.get(
        'architecture_round') in ('FA1_R7_31_LAYERED_DIAMOND_V1',
                                  'FA1_R7_32_CALIBRATED_DIAMOND_V1'))
    if resuming:
        return False
    open_candidate_gates(model,args.bridge,args.meta,args.contextual_init,
                         getattr(args,'factor_up_init',0.),
                         getattr(args,'factor_decoder_init',0.))
    if getattr(args,'layered',False):
        with torch.no_grad():
            for name in ('context_gain','point_gain','pair_gain','triangle_gain',
                         'word_gain','byte_geometry_gain'):
                getattr(model.diamond.hierarchy,name).fill_(math.atanh(args.layered_gate_init))
    return True


def train_step(model, optimizer, sample, step, args):
    model.train()
    # Compute target-free geometry-off teacher before constructing the live
    # backward graph. Toggling gates after that graph is built would mutate
    # an autograd-saved tensor in-place and invalidate the gradient.
    no_geo_branch = None
    no_read_branch = None
    no_factor_branch = None
    no_geo_bytes = None
    matched = None
    no_geo_matched_branch = None
    no_geo_matched_bytes = None
    if (getattr(args,'layered',False) and step % args.ablate_every == 0 and
            (args.matched_geometry_weight or args.matched_byte_weight)):
        matched,_=matched_history_context(sample,args.seed+step+101)
    far_matched = None
    far_horizon_index = -1
    far_no_geo_branch = None
    far_no_geo_bytes = None
    if getattr(args,'matched_far_weight',0.) > 0:
        far_horizon_index=(step-1)//max(1,args.ablate_every)%len(HORIZONS)
        far_matched, _ = matched_far_history(sample,args.seed+step+13,far_horizon_index)
    if step % args.ablate_every == 0:
        with torch.no_grad():
            with geometry_disabled(model):
                _, no_geo = forward(model, sample)
                if 'byte_ar_logits' in no_geo:
                    no_geo_bytes=layered_byte_losses(no_geo,sample)[0].detach()
                if matched is not None:
                    _, no_matched = forward(model, matched)
                    no_geo_matched_branch=no_matched['branch_nll'].detach()
                    if 'byte_ar_logits' in no_matched:
                        no_geo_matched_bytes=layered_byte_losses(no_matched,sample)[0].detach()
                if far_matched is not None:
                    _, far_no_geo = forward(model,far_matched)
                    far_no_geo_branch=far_no_geo['branch_nll'].detach()
                    if 'byte_ar_logits' in far_no_geo:
                        far_no_geo_bytes=layered_byte_losses(far_no_geo,sample)[0].detach()
            if hasattr(model.diamond.geometry,'context_read_gain'):
                with contextual_read_disabled(model):
                    _, no_read = forward(model,sample)
                no_read_branch=no_read['branch_nll'].detach()
            if hasattr(model.diamond.geometry,'factor_up_gain'):
                with factor_read_disabled(model):
                    _,no_factor=forward(model,sample)
                no_factor_branch=no_factor['branch_nll'].detach()
        no_geo_branch = no_geo['branch_nll'].detach()
    correct, extra = forward(model, sample)
    branch = extra['branch_nll'].mean()
    byte = F.cross_entropy(extra['byte_logits'], sample['first_byte_target'])
    byte_ar=correct.new_zeros(())
    byte_geo_penalty=correct.new_zeros(())
    if 'byte_ar_logits' in extra:
        per_word,counts,ce_bytes,eos,_=layered_byte_losses(extra,sample)
        byte_ar=(ce_bytes+eos).sum()/((sample['byte_target_sequence']!=-100).sum().clamp_min(1))
        if no_geo_bytes is not None:
            valid=counts>0
            byte_geo_penalty=soft_margin(per_word[valid],no_geo_bytes[valid],args.geometry_margin)
    foreign, _ = unrelated_book_context(sample, args.seed + step)
    _, wrong_extra = forward(model, foreign)
    different_book = soft_margin(extra['branch_nll'], wrong_extra['branch_nll'], args.history_margin)
    matched_geo_penalty=correct.new_zeros(())
    matched_byte_penalty=correct.new_zeros(())
    if matched is not None:
        _, matched_extra = forward(model, matched)
        matched_geo_gain=geometry_specific_history_gain(
            extra['branch_nll'],matched_extra['branch_nll'],
            no_geo_branch,no_geo_matched_branch)
        matched_geo_penalty=context_specific_margin(matched_geo_gain,args.matched_margin)
        if 'byte_ar_logits' in extra:
            matched_byte=layered_byte_losses(matched_extra,sample)[0]
            byte_gain=geometry_specific_history_gain(
                per_word,matched_byte,no_geo_bytes,no_geo_matched_bytes)
            valid=counts>0
            if bool(valid.any()):
                matched_byte_penalty=context_specific_margin(byte_gain[valid],args.matched_margin)
    matched_word_loss=correct.new_zeros(())
    matched_byte_loss=correct.new_zeros(())
    matched_geo_specificity=correct.new_zeros(())
    matched_byte_specificity=correct.new_zeros(())
    if far_matched is not None:
        _, far_extra = forward(model,far_matched)
        matched_word_loss=soft_margin(extra['branch_nll'],far_extra['branch_nll'],args.history_margin)
        if 'byte_ar_logits' in far_extra:
            far_per_word=layered_byte_losses(far_extra,sample)[0]
            matched_byte_loss=soft_margin(per_word[counts>0],far_per_word[counts>0],args.history_margin)
        if far_no_geo_branch is not None:
            # Difference-in-differences: geometry must help *more* with the
            # true old history than with the swapped foreign-book history.
            correct_gain=no_geo_branch-extra['branch_nll']
            wrong_gain=far_no_geo_branch-far_extra['branch_nll']
            matched_geo_specificity=soft_margin(wrong_gain,correct_gain,args.geometry_margin)
            if far_no_geo_bytes is not None:
                byte_correct_gain=no_geo_bytes-per_word
                byte_wrong_gain=far_no_geo_bytes-far_per_word
                valid=counts>0
                matched_byte_specificity=soft_margin(byte_wrong_gain[valid],byte_correct_gain[valid],args.geometry_margin)
    # The two full-signal ablations force the memory to be stored in geometric
    # faces and used by the apex -> pair -> point read path.  The counterfactual
    # never modifies the target or the inherited R7 hidden state.
    geo_loss = correct.new_zeros(())
    apex_loss = correct.new_zeros(())
    pair_loss = correct.new_zeros(())
    read_loss = correct.new_zeros(())
    factor_loss = correct.new_zeros(())
    if no_geo_branch is not None:
        geo_loss = soft_margin(extra['branch_nll'], no_geo_branch, args.geometry_margin)
        if no_read_branch is not None:
            read_loss=soft_margin(extra['branch_nll'],no_read_branch,args.geometry_margin)
        if no_factor_branch is not None:
            factor_loss=soft_margin(extra['branch_nll'],no_factor_branch,args.geometry_margin)
        _, no_apex = forward(model, sample, 'no_apex')
        apex_loss = soft_margin(extra['branch_nll'], no_apex['branch_nll'], args.geometry_margin)
        _, no_pair = forward(model, sample, 'no_pair')
        pair_loss = soft_margin(extra['branch_nll'], no_pair['branch_nll'], args.geometry_margin)
    objective = (args.full_weight * correct.mean() + args.branch_weight * branch +
                 args.byte_weight * byte + args.history_weight * different_book +
                 args.geometry_weight * geo_loss + args.apex_weight * apex_loss +
                 args.pair_weight * pair_loss + args.contextual_weight * read_loss +
                 getattr(args,'factor_weight',0.) * factor_loss +
                 getattr(args,'byte_ar_weight',0.) * byte_ar +
                 getattr(args,'byte_geometry_weight',0.) * byte_geo_penalty +
                 getattr(args,'matched_geometry_weight',0.) * matched_geo_penalty +
                 getattr(args,'matched_byte_weight',0.) * matched_byte_penalty +
                 getattr(args,'matched_far_weight',0.) * matched_word_loss +
                 getattr(args,'matched_far_byte_weight',0.) * matched_byte_loss +
                 getattr(args,'matched_far_geometry_weight',0.) * matched_geo_specificity +
                 getattr(args,'matched_far_byte_geometry_weight',0.) * matched_byte_specificity)
    optimizer.zero_grad(set_to_none=True)
    objective.backward()
    face_grad = model.diamond.geometry.face_plus.weight.grad
    addressed = int((face_grad.abs().sum(-1) > 0).sum()) if face_grad is not None else 0
    torch.nn.utils.clip_grad_norm_(model.diamond.parameters(), args.clip)
    optimizer.step()
    return {'train_total': float(objective.detach()), 'train_full_nll': float(correct.mean().detach()),
            'train_branch_nll': float(branch.detach()), 'train_first_byte_nll': float(byte.detach()),
            'train_different_book_penalty': float(different_book.detach()),
            'train_factor_contrast':float(factor_loss.detach()),
            'train_layered_byte_ar_nll':float(byte_ar.detach()),
            'train_layered_byte_geometry_contrast':float(byte_geo_penalty.detach()),
            'train_matched_geometry_contrast':float(matched_geo_penalty.detach()),
            'train_matched_byte_contrast':float(matched_byte_penalty.detach()),
            'train_matched_far_word_contrast':float(matched_word_loss.detach()),
            'train_matched_far_byte_contrast':float(matched_byte_loss.detach()),
            'train_matched_far_geometry_specificity':float(matched_geo_specificity.detach()),
            'train_matched_far_byte_specificity':float(matched_byte_specificity.detach()),
            'train_matched_far_horizon':far_horizon_index,
            'train_face_plus_addresses_with_gradient': addressed}


def passes_gate(reference, candidate, min_gain):
    """Report strict development criteria; never promote based on training."""
    if candidate['full_word_nll'] >= reference['full_word_nll'] - min_gain:
        return False
    if candidate['geometry_word_gain'] <= 0 or candidate['branch_geometry_gain'] <= 0:
        return False
    if candidate['correct_book_advantage_nats_per_word'] <= 0:
        return False
    if candidate['apex_word_gain'] <= 0 or candidate['pair_word_gain'] <= 0:
        return False
    if candidate['book_geometry_gain']['book_bootstrap_ci95'][0] <= 0:
        return False
    if candidate['book_correct_history']['book_bootstrap_ci95'][0] <= 0:
        return False
    if 'contextual_read_word_gain' in candidate:
        if (candidate['contextual_read_word_gain'] <= 0 or
                candidate['contextual_read_branch_gain'] <= 0 or
                candidate['contextual_read_book_gain']['book_bootstrap_ci95'][0] <= 0):
            return False
    if 'factor_word_gain' in candidate:
        if (candidate['factor_word_gain']<=0 or candidate['factor_branch_gain']<=0 or
                candidate['factor_word_gain_bootstrap']['book_bootstrap_ci95'][0]<=0 or
                candidate['factor_branch_gain_bootstrap']['book_bootstrap_ci95'][0]<=0):
            return False
    if 'conditional_byte_geometry_gain_nats_per_byte' in candidate:
        if (candidate['conditional_byte_geometry_gain_nats_per_byte']<=0 or
                candidate['conditional_byte_apex_gain_nats_per_byte']<=0 or
                candidate['conditional_byte_pair_gain_nats_per_byte']<=0 or
                candidate['conditional_byte_geometry_gain_book_bootstrap']['book_bootstrap_ci95'][0]<=0):
            return False
        if candidate['conditional_spelling_bits_per_byte'] >= reference['conditional_spelling_bits_per_byte']:
            return False
        for key in ('conditional_spelling_top1','conditional_spelling_top5'):
            if key in reference and candidate[key] < reference[key]:
                return False
    if 'matched_geometry_word_advantage' in candidate:
        if (candidate['matched_geometry_word_advantage']<=0 or
                candidate['matched_geometry_branch_advantage']<=0 or
                candidate['matched_geometry_word_book_bootstrap']['book_bootstrap_ci95'][0]<=0 or
                candidate['matched_geometry_branch_book_bootstrap']['book_bootstrap_ci95'][0]<=0 or
                candidate['matched_geometry_byte_advantage_nats_per_byte']<=0 or
                candidate['matched_geometry_byte_book_bootstrap']['book_bootstrap_ci95'][0]<=0):
            return False
    if 'matched_far_history_advantage_nats_per_word' in candidate:
        for key in ('matched_far_history_advantage_nats_per_word',
                    'matched_far_geometry_specificity_nats_per_word'):
            if candidate[key] <= 0:
                return False
        for key in ('matched_far_history_bootstrap','matched_far_geometry_bootstrap'):
            if candidate[key]['book_bootstrap_ci95'][0] <= 0:
                return False
        if 'matched_far_byte_history_advantage_nats_per_byte' in candidate:
            for key in ('matched_far_byte_history_advantage_nats_per_byte',
                        'matched_far_byte_geometry_specificity_nats_per_byte'):
                if candidate[key] <= 0:
                    return False
            for key in ('matched_far_byte_history_bootstrap','matched_far_byte_geometry_bootstrap'):
                if candidate[key]['book_bootstrap_ci95'][0] <= 0:
                    return False
    return True


def candidate_priority(reference, candidate, min_gain):
    """Strictly admissible beats merely promising; both beat loss-only.

    This is a *selection* ranking, not a relaxation of the promotion gate.
    Ties within a tier use full word loss then conditional byte bits/byte.
    """
    strict = passes_gate(reference, candidate, min_gain)
    geometry_ready = (candidate['full_word_nll'] < reference['full_word_nll'] - min_gain)
    geometry_ready = geometry_ready and all(candidate[k] > 0 for k in (
        'geometry_word_gain', 'branch_geometry_gain',
        'correct_book_advantage_nats_per_word', 'apex_word_gain',
        'pair_word_gain'))
    if 'conditional_byte_geometry_gain_nats_per_byte' in candidate:
        geometry_ready = geometry_ready and all(candidate[k] > 0 for k in (
            'conditional_byte_geometry_gain_nats_per_byte',
            'conditional_byte_apex_gain_nats_per_byte',
            'conditional_byte_pair_gain_nats_per_byte'))
        geometry_ready = (geometry_ready and
            candidate['conditional_spelling_bits_per_byte'] <
            reference['conditional_spelling_bits_per_byte'])
    if 'matched_geometry_word_advantage' in candidate:
        # A candidate with context-insensitive geometry cannot earn the
        # intermediate geometry-ready tier merely by lowering word loss.
        geometry_ready = geometry_ready and all(candidate[k] > 0 for k in (
            'matched_geometry_word_advantage',
            'matched_geometry_branch_advantage',
            'matched_geometry_byte_advantage_nats_per_byte'))
    if 'matched_far_history_advantage_nats_per_word' in candidate:
        geometry_ready = geometry_ready and all(candidate[k] > 0 for k in (
            'matched_far_history_advantage_nats_per_word',
            'matched_far_geometry_specificity_nats_per_word',
            'matched_far_byte_history_advantage_nats_per_byte',
            'matched_far_byte_geometry_specificity_nats_per_byte'))
    tier = 2 if strict else 1 if geometry_ready else 0
    return (tier, -candidate['full_word_nll'],
            -candidate.get('conditional_spelling_bits_per_byte', float('inf')))


def run(args):
    t0 = time.monotonic()
    if any(getattr(args, n, 0.) > 0 for n in (
        'matched_far_weight','matched_far_byte_weight',
        'matched_far_geometry_weight','matched_far_byte_geometry_weight')):
        args.matched_far_history=True
    if getattr(args,'matched_far_history',False) and not getattr(args,'calibrated',False):
        raise ValueError('R7.33 matched far-history controls require --calibrated')
    if getattr(args,'matched_far_history',False) and any(getattr(args,n,0.)>0 for n in (
            'matched_far_byte_weight','matched_far_geometry_weight',
            'matched_far_byte_geometry_weight')) and getattr(args,'matched_far_weight',0.)<=0:
        raise ValueError('matched-far auxiliary weights require positive --matched-far-weight')
    if getattr(args, 'calibrated', False):
        args.layered = True  # same layered byte/sentence/geometry supervision
    if args.split != 'dev':
        raise ValueError('candidate selection is DEV-only; independent TEST is sealed')
    inventory = validate_book_corpus(args.corpus_dir, args.split)
    torch.set_num_threads(args.threads)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    parent_hash = digest(args.checkpoint)
    if getattr(args,'calibrated',False):
        from fa1_r732_geometry_calibration import load_calibrated
        model, optimizer, base, compositor, parent_meta = load_calibrated(args.checkpoint,args.root,args.threads)
    elif getattr(args,'layered',False):
        from fa1_r731_layered_decoder import load_layered
        model, optimizer, base, compositor, parent_meta = load_layered(args.checkpoint,args.root,args.threads)
    elif getattr(args,'factorized',False):
        from fa1_r730_factorized_geometry import load_factorized
        model, optimizer, base, compositor, parent_meta = load_factorized(args.checkpoint,args.root,args.threads)
    elif args.contextual:
        from fa1_r729_contextual_geometry import load_contextual
        model, optimizer, base, compositor, parent_meta = load_contextual(args.checkpoint, args.root, args.threads)
    else:
        model, optimizer, base, compositor, parent_meta = load_unified(args.checkpoint, args.root, args.threads)
    # Real R7.27 parent prediction before any gate warm start or gradient.
    dev = prepare(args, model, base, compositor, 'dev', args.dev, args.seed+2)
    if getattr(args,'layered',False):dev['layered_byte_supervision']=True
    original = geometry_diagnostic(model, dev, args.seed+3, True, base,
                                   with_matched_far=getattr(args,'matched_far_history',False))
    print('PARENT', json.dumps(original), flush=True)
    train = prepare(args, model, base, compositor, 'train', args.train, args.seed+1)
    if getattr(args,'layered',False):train['layered_byte_supervision']=True
    if len(torch.unique(train['book_ids'])) < 2:
        raise ValueError('training positions must span two or more books')
    # Preserve all non-Diamond weights/moments: only the already-present joint
    # geometric+apex path receives new gradients. All 13 optimizer groups and
    # historical accumulator algorithms remain intact.
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.diamond.parameters():
        p.requires_grad_(True)
    optimizer.param_groups[-1]['lr'] = args.lr
    face_plus_start = model.diamond.geometry.face_plus.weight.detach().clone()
    face_minus_start = model.diamond.geometry.face_minus.weight.detach().clone()
    triangle_code_start = model.diamond.triple_code.weight.detach().clone()
    # Source-migrated candidates get the initial gate warm-start. Fully trained
    # R7.31/R7.32 candidates retain the learned gates and Adam moments.
    preserve_learned_gates = not warm_start_only_if_migrated(model,args,parent_meta)
    opened = geometry_diagnostic(model, dev, args.seed+3,
                                 with_matched_far=getattr(args,'matched_far_history',False))
    print('CANDIDATE_GATE_OPEN', json.dumps(opened), flush=True)
    rng = np.random.default_rng(args.seed+9)
    history = []
    best_rank = None
    best_state = None
    best_step = 0
    for step in range(1, args.steps+1):
        ids = torch.tensor(rng.choice(args.train, min(args.minibatch,args.train),replace=False))
        sampled = take(train, ids)
        if sampled['book_ids'].unique().numel() < 2:
            donor = torch.where(train['book_ids'] != sampled['book_ids'][0])[0]
            ids[0] = donor[int(rng.integers(len(donor)))]
            sampled = take(train, ids)
        train_result = train_step(model, optimizer, sampled, step, args)
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            diagnostics = geometry_diagnostic(model, dev, args.seed+3,
                                               with_matched_far=getattr(args,'matched_far_history',False))
            diagnostics.update({'step':step,**train_result})
            diagnostics['development_gate_passed'] = passes_gate(original, diagnostics, args.min_gain)
            rank = candidate_priority(original, diagnostics, args.min_gain)
            diagnostics['candidate_selection_tier'] = rank[0]
            history.append(diagnostics)
            print('STEP', json.dumps(diagnostics), flush=True)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_step = step
                # Restore the selected state even when saving a .pt was not
                # requested; otherwise a no-save probe reports a selection
                # but actually scores the last (possibly worse) update.
                best_state = (copy.deepcopy(model.diamond.state_dict()),
                              copy.deepcopy(optimizer.state_dict()))
    if best_state is not None:
        model.diamond.load_state_dict(best_state[0])
        optimizer.load_state_dict(best_state[1])
    last = geometry_diagnostic(model, dev, args.seed+3, True, base,
                               with_matched_far=getattr(args,'matched_far_history',False))
    with torch.no_grad():
        plus_changed = (model.diamond.geometry.face_plus.weight != face_plus_start).any(dim=-1)
        minus_changed = (model.diamond.geometry.face_minus.weight != face_minus_start).any(dim=-1)
        old_code_changed = (model.diamond.triple_code.weight != triangle_code_start).any(dim=-1)
    footprint = {
        'signed_splus_triangle_codes_updated':int(plus_changed.sum()),
        'signed_sminus_triangle_codes_updated':int(minus_changed.sum()),
        'existing_triangle_code_rows_updated':int(old_code_changed.sum()),
        'total_possible_triangle_addresses':len(face_plus_start),
        'persistent_model_tensors_used':True,
    }
    # This is a scientific probe, not a checkpoint promotion. Its full result
    # stays in JSON, and future rounds may choose to save a guarded candidate.
    report = {
        'format':('FA1_R7_33_MATCHED_FAR_HISTORY_DEV_PROBE_V1' if getattr(args,'matched_far_history',False) else
                  'FA1_R7_32_CALIBRATED_DIAMOND_DEV_PROBE_V1' if getattr(args,'calibrated',False) else
                  'FA1_R7_31_LAYERED_DIAMOND_DEV_PROBE_V1' if getattr(args,'layered',False) else
                  'FA1_R7_30_FACTORIZED_GEOMETRY_DEV_PROBE_V1' if getattr(args,'factorized',False) else
                  'FA1_R7_29_CONTEXTUAL_GEOMETRY_DEV_PROBE_V1' if args.contextual
                  else 'FA1_R7_28_GEOMETRIC_MEMORY_DEV_PROBE_V1'),
        'source_checkpoint_sha256':parent_hash,
        'branch':'master',
        'contextual_geometric_read':bool(args.contextual or getattr(args,'factorized',False) or getattr(args,'layered',False)),
        'factorized_geometric_read':bool(getattr(args,'factorized',False) or getattr(args,'layered',False)),
        'layered_autoregressive_decoder':bool(getattr(args,'layered',False)),
        'resumed_learned_gates_unchanged':preserve_learned_gates,
        'matched_history_contrast':bool(getattr(args,'layered',False)),
        'calibrated_geometric_trust':bool(getattr(args,'calibrated',False)),
        'matched_far_history_controls':bool(getattr(args,'matched_far_history',False)),
        'accepted_parent_step':14416,
        'one_optimizer_groups':len(optimizer.param_groups),
        'source_checkpoint_never_overwritten': digest(args.checkpoint)==parent_hash,
        'checkpoint_promoted':False,
        'inherited_learned_gates_preserved':bool(preserve_learned_gates),
        'test_books_consulted':False,
        'corpus_inventory':inventory,
        'dev_positions':args.dev,'train_positions':args.train,
        'train_steps':args.steps,
        'chosen_dev_step':best_step,
        'chosen_dev_selection_tier':best_rank[0] if best_rank is not None else None,
        'chosen_dev_strict_gate_passed':bool(best_rank is not None and best_rank[0]==2),
        'candidate_selection_policy':'strict_gate > positive_geometry_and_history > loss_only; then word NLL, then spelling BPB',
        'best_word_nll_step':min(history,key=lambda x:x['full_word_nll'])['step'] if history else None,
        'best_spelling_bits_per_byte_step':min(history,key=lambda x:x.get('conditional_spelling_bits_per_byte',float('inf')))['step'] if history and getattr(args,'layered',False) else None,
        'baseline':original,'gate_open_only':opened,'history':history,
        'final_development_with_accuracy':last,
        'geometry_address_storage':footprint,
        'best_development':next((x for x in history if x['step']==best_step),None),
        'best_word_loss_development':min(history,key=lambda x:x['full_word_nll']) if history else None,
        'elapsed_seconds':time.monotonic()-t0,
    }
    if args.candidate_checkpoint:
        report['unpromoted_candidate_checkpoint'] = save_unpromoted(
            args.candidate_checkpoint, model, optimizer, args.root, args.checkpoint,
            source_metadata={'format':'FA1_R7_27_SINGLE_MODEL_SINGLE_HYBRID_OPTIMIZER_V1'},
            candidate_trained=True, training_round=('FA1 R7.33 matched old-history geometry contrast'
                                                     if getattr(args,'matched_far_history',False) else
                                                     'FA1 R7.32 calibrated geometric trust decoder'
                                                     if getattr(args,'calibrated',False) else
                                                     'FA1 R7.31 layered geometric autoregressive decoder'
                                                     if getattr(args,'layered',False) else
                                                     'FA1 R7.30 factorized geometric memory'
                                                     if getattr(args,'factorized',False) else
                                                     'FA1 R7.29 context-addressed geometric memory'
                                                     if args.contextual else 'FA1 R7.28 geometric memory'),
            training_updates=best_step, development_gate_passed=passes_gate(original,last,args.min_gain),
            architecture_round=('FA1_R7_32_CALIBRATED_DIAMOND_V1'
                                if getattr(args,'calibrated',False) else
                                'FA1_R7_31_LAYERED_DIAMOND_V1'
                                if getattr(args,'layered',False) else
                                'FA1_R7_30_FACTORIZED_GEOMETRY_V1'
                                if getattr(args,'factorized',False) else
                                'FA1_R7_29_CONTEXTUAL_GEOMETRY_V1' if args.contextual else ''))
    else:
        report['unpromoted_candidate_checkpoint'] = None
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print('REPORT',str(output),flush=True)
    return report


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--corpus-dir',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--candidate-checkpoint',default='',help='separate unpromoted successor; never accepted parent')
    p.add_argument('--contextual',action='store_true',help='R7.29 context-addressed signed triangle read')
    p.add_argument('--factorized',action='store_true',help='R7.30 continuation with point-factorized readout')
    p.add_argument('--layered',action='store_true',help='R7.31 continuation with full-word byte AR')
    p.add_argument('--calibrated',action='store_true',help='R7.32 continuation with identity-initialized word/byte geometric trust')
    p.add_argument('--matched-far-history',action='store_true',help='R7.33 matched recent-context/far-history DEV controls')
    p.add_argument('--matched-far-weight',type=float,default=0.,help='TRAIN correct-vs-foreign far-history word contrast')
    p.add_argument('--matched-far-byte-weight',type=float,default=0.,help='TRAIN conditional byte far-history contrast')
    p.add_argument('--matched-far-geometry-weight',type=float,default=0.,help='TRAIN word geometric difference-in-differences')
    p.add_argument('--matched-far-byte-geometry-weight',type=float,default=0.,help='TRAIN byte geometric difference-in-differences')
    p.add_argument('--layered-gate-init',type=float,default=.12)
    p.add_argument('--byte-ar-weight',type=float,default=.35)
    p.add_argument('--byte-geometry-weight',type=float,default=.2)
    p.add_argument('--matched-geometry-weight',type=float,default=.6,
                   help='R7.32 matched-old-history branch geometry contrast (R7.31 only)')
    p.add_argument('--matched-byte-weight',type=float,default=.25,
                   help='R7.32 matched-old-history byte geometry contrast (R7.31 only)')
    p.add_argument('--matched-margin',type=float,default=.005)
    p.add_argument('--factor-up-init',type=float,default=0.20)
    p.add_argument('--factor-decoder-init',type=float,default=0.35)
    p.add_argument('--factor-weight',type=float,default=0.75)
    p.add_argument('--contextual-init',type=float,default=0.15)
    p.add_argument('--contextual-weight',type=float,default=0.6)
    p.add_argument('--split',default='dev')
    p.add_argument('--train',type=int,default=256)
    p.add_argument('--dev',type=int,default=128)
    p.add_argument('--steps',type=int,default=32)
    p.add_argument('--minibatch',type=int,default=12)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--eval-every',type=int,default=8)
    p.add_argument('--ablate-every',type=int,default=4)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--lr',type=float,default=0.00015)
    p.add_argument('--bridge',type=float,default=0.35)
    p.add_argument('--meta',type=float,default=0.15)
    p.add_argument('--full-weight',type=float,default=1.)
    p.add_argument('--branch-weight',type=float,default=0.75)
    p.add_argument('--byte-weight',type=float,default=0.15)
    p.add_argument('--history-weight',type=float,default=0.5)
    p.add_argument('--geometry-weight',type=float,default=0.4)
    p.add_argument('--apex-weight',type=float,default=0.2)
    p.add_argument('--pair-weight',type=float,default=0.2)
    p.add_argument('--history-margin',type=float,default=0.003)
    p.add_argument('--geometry-margin',type=float,default=0.003)
    p.add_argument('--min-gain',type=float,default=0.0001)
    p.add_argument('--clip',type=float,default=1.)
    p.add_argument('--seed',type=int,default=728)
    return p.parse_args()


if __name__=='__main__':
    run(parse_args())
