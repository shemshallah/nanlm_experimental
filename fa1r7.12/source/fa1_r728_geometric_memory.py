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


@contextlib.contextmanager
def geometry_disabled(model):
    """Counterfactual: only disable new geometric channels, never the trunk."""
    names = list(GEOMETRY_GATES)
    if hasattr(model.diamond.geometry,'context_read_gain'):
        names.append('context_read_gain')
    gates = [getattr(model.diamond.geometry, n) for n in names]
    before = [p.detach().clone() for p in gates]
    with torch.no_grad():
        for param in gates:
            param.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            for p, value in zip(gates, before):
                p.copy_(value)


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


@torch.no_grad()
def geometry_diagnostic(model, data, seed, with_topk=False, base=None):
    model.eval()
    correct, extra = forward(model, data)
    with geometry_disabled(model):
        without_geometry, no_geo_extra = forward(model, data)
    if hasattr(model.diamond.geometry,'context_read_gain'):
        with contextual_read_disabled(model):
            without_read, no_read_extra = forward(model,data)
    else:
        without_read, no_read_extra = correct, extra
    foreign, donors = unrelated_book_context(data, seed)
    foreign_nll, foreign_extra = forward(model, foreign)
    no_apex, no_apex_extra = forward(model, data, 'no_apex')
    no_pair, no_pair_extra = forward(model, data, 'no_pair')
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


def open_candidate_gates(model, bridge, meta, contextual_init=0.):
    """A candidate-only gate warm start; the saved R7.27 parent is untouched."""
    with torch.no_grad():
        model.diamond.gate.fill_(math.atanh(bridge))
        geometry = model.diamond.geometry
        if abs(float(geometry.meta_gate)) < 1.e-9:
            geometry.meta_gate.fill_(math.atanh(meta))
        if hasattr(geometry,'context_read_gain'):
            geometry.context_read_gain.fill_(math.atanh(contextual_init))


def train_step(model, optimizer, sample, step, args):
    model.train()
    # Compute target-free geometry-off teacher before constructing the live
    # backward graph. Toggling gates after that graph is built would mutate
    # an autograd-saved tensor in-place and invalidate the gradient.
    no_geo_branch = None
    no_read_branch = None
    if step % args.ablate_every == 0:
        with torch.no_grad():
            with geometry_disabled(model):
                _, no_geo = forward(model, sample)
            if hasattr(model.diamond.geometry,'context_read_gain'):
                with contextual_read_disabled(model):
                    _, no_read = forward(model,sample)
                no_read_branch=no_read['branch_nll'].detach()
        no_geo_branch = no_geo['branch_nll'].detach()
    correct, extra = forward(model, sample)
    branch = extra['branch_nll'].mean()
    byte = F.cross_entropy(extra['byte_logits'], sample['first_byte_target'])
    foreign, _ = unrelated_book_context(sample, args.seed + step)
    _, wrong_extra = forward(model, foreign)
    different_book = soft_margin(extra['branch_nll'], wrong_extra['branch_nll'], args.history_margin)
    # The two full-signal ablations force the memory to be stored in geometric
    # faces and used by the apex -> pair -> point read path.  The counterfactual
    # never modifies the target or the inherited R7 hidden state.
    geo_loss = correct.new_zeros(())
    apex_loss = correct.new_zeros(())
    pair_loss = correct.new_zeros(())
    read_loss = correct.new_zeros(())
    if no_geo_branch is not None:
        geo_loss = soft_margin(extra['branch_nll'], no_geo_branch, args.geometry_margin)
        if no_read_branch is not None:
            read_loss=soft_margin(extra['branch_nll'],no_read_branch,args.geometry_margin)
        _, no_apex = forward(model, sample, 'no_apex')
        apex_loss = soft_margin(extra['branch_nll'], no_apex['branch_nll'], args.geometry_margin)
        _, no_pair = forward(model, sample, 'no_pair')
        pair_loss = soft_margin(extra['branch_nll'], no_pair['branch_nll'], args.geometry_margin)
    objective = (args.full_weight * correct.mean() + args.branch_weight * branch +
                 args.byte_weight * byte + args.history_weight * different_book +
                 args.geometry_weight * geo_loss + args.apex_weight * apex_loss +
                 args.pair_weight * pair_loss + args.contextual_weight * read_loss)
    optimizer.zero_grad(set_to_none=True)
    objective.backward()
    face_grad = model.diamond.geometry.face_plus.weight.grad
    addressed = int((face_grad.abs().sum(-1) > 0).sum()) if face_grad is not None else 0
    torch.nn.utils.clip_grad_norm_(model.diamond.parameters(), args.clip)
    optimizer.step()
    return {'train_total': float(objective.detach()), 'train_full_nll': float(correct.mean().detach()),
            'train_branch_nll': float(branch.detach()), 'train_first_byte_nll': float(byte.detach()),
            'train_different_book_penalty': float(different_book.detach()),
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
    return True


def run(args):
    t0 = time.monotonic()
    if args.split != 'dev':
        raise ValueError('candidate selection is DEV-only; independent TEST is sealed')
    inventory = validate_book_corpus(args.corpus_dir, args.split)
    torch.set_num_threads(args.threads)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    parent_hash = digest(args.checkpoint)
    if args.contextual:
        from fa1_r729_contextual_geometry import load_contextual
        model, optimizer, base, compositor, _ = load_contextual(args.checkpoint, args.root, args.threads)
    else:
        model, optimizer, base, compositor, _ = load_unified(args.checkpoint, args.root, args.threads)
    # Real R7.27 parent prediction before any gate warm start or gradient.
    dev = prepare(args, model, base, compositor, 'dev', args.dev, args.seed+2)
    original = geometry_diagnostic(model, dev, args.seed+3, True, base)
    print('PARENT', json.dumps(original), flush=True)
    train = prepare(args, model, base, compositor, 'train', args.train, args.seed+1)
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
    open_candidate_gates(model, args.bridge, args.meta, args.contextual_init)
    opened = geometry_diagnostic(model, dev, args.seed+3)
    print('CANDIDATE_GATE_OPEN', json.dumps(opened), flush=True)
    rng = np.random.default_rng(args.seed+9)
    history = []
    best_loss = float('inf')
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
            diagnostics = geometry_diagnostic(model, dev, args.seed+3)
            diagnostics.update({'step':step,**train_result})
            diagnostics['development_gate_passed'] = passes_gate(original, diagnostics, args.min_gain)
            history.append(diagnostics)
            print('STEP', json.dumps(diagnostics), flush=True)
            if diagnostics['full_word_nll'] < best_loss:
                best_loss = diagnostics['full_word_nll']
                best_step = step
                if args.candidate_checkpoint:
                    best_state = (copy.deepcopy(model.diamond.state_dict()),
                                  copy.deepcopy(optimizer.state_dict()))
    if best_state is not None:
        model.diamond.load_state_dict(best_state[0])
        optimizer.load_state_dict(best_state[1])
    last = geometry_diagnostic(model, dev, args.seed+3, True, base)
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
        'format':('FA1_R7_29_CONTEXTUAL_GEOMETRY_DEV_PROBE_V1' if args.contextual
                  else 'FA1_R7_28_GEOMETRIC_MEMORY_DEV_PROBE_V1'),
        'source_checkpoint_sha256':parent_hash,
        'branch':'master',
        'contextual_geometric_read':bool(args.contextual),
        'accepted_parent_step':14416,
        'one_optimizer_groups':len(optimizer.param_groups),
        'source_checkpoint_never_overwritten': digest(args.checkpoint)==parent_hash,
        'checkpoint_promoted':False,
        'test_books_consulted':False,
        'corpus_inventory':inventory,
        'dev_positions':args.dev,'train_positions':args.train,
        'train_steps':args.steps,
        'chosen_dev_step':best_step,
        'baseline':original,'gate_open_only':opened,'history':history,
        'final_development_with_accuracy':last,
        'geometry_address_storage':footprint,
        'best_development':min(history,key=lambda x:x['full_word_nll']) if history else None,
        'elapsed_seconds':time.monotonic()-t0,
    }
    if args.candidate_checkpoint:
        report['unpromoted_candidate_checkpoint'] = save_unpromoted(
            args.candidate_checkpoint, model, optimizer, args.root, args.checkpoint,
            source_metadata={'format':'FA1_R7_27_SINGLE_MODEL_SINGLE_HYBRID_OPTIMIZER_V1'},
            candidate_trained=True, training_round=('FA1 R7.29 context-addressed geometric memory'
                                                     if args.contextual else 'FA1 R7.28 geometric memory'),
            training_updates=best_step, development_gate_passed=passes_gate(original,last,args.min_gain),
            architecture_round=('FA1_R7_29_CONTEXTUAL_GEOMETRY_V1' if args.contextual else ''))
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
