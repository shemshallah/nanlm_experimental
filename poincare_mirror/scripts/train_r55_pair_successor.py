#!/usr/bin/env python3
"""FA2 R55: TRAIN-only four-parameter successor gate; paired book-disjoint DEV.

This is a continuation of ONE FA2 model and ONE optimizer state, not a donor
or a second predictor. TEST is never loaded. A candidate is not promoted.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poincare_frontier import (Config, MirrorLM, content_balanced_ce,
                               file_hash, load_meta, load_v25_compatible)
from compare_geometry_books import METRICS, book_starts
from probe_r47_selective_emission import book_comparison


PROTECTED_SHA = '74478d2f77b3c2da7978458f3660dffe8aad788938ce4a7ab6eae7cff2de0817'
ARMS = ('neutral', 'exact', 'wrong_address', 'wrong_successor')
COMPOUND_ARMS = ARMS + ('wrong_byte',)


def checked_corpus(directory, saved):
    meta = load_meta(directory)
    if meta != saved['data']:
        raise ValueError('Checkpoint and Gutenberg38 corpus metadata differ')
    if meta.get('unit') != 'UTF8_bytes_plus_eos' or meta.get('split_unit') != 'whole_book':
        raise ValueError('Requires verified whole-book byte/EOS splits')
    if file_hash(directory / 'split_manifest.json') != meta['split_manifest_sha256']:
        raise ValueError('Book split manifest SHA-256 mismatch')
    arrays = {}
    for split in ('train', 'valid'):  # Never open TEST during candidate selection.
        path = directory / (split + '.bin')
        expected = meta.get(split + '_sha256') or meta.get('bin_' + split + '_sha256')
        if not expected or file_hash(path) != expected:
            raise ValueError(split + ' corpus SHA-256 mismatch')
        arrays[split] = np.memmap(path, dtype=meta['token_dtype'], mode='r')
        if len(arrays[split]) != sum(meta[split + '_doc_lengths']):
            raise ValueError(split + ' book offsets mismatch')
    book_manifest = json.loads((directory / 'split_manifest.json').read_text())
    train_ids = {book['gutenberg_id'] for book in book_manifest['books']
                 if book['split'] == 'train'}
    dev_ids = {book['gutenberg_id'] for book in book_manifest['books']
               if book['split'] in ('dev', 'valid')}
    if len(train_ids) != meta['train_books'] or len(dev_ids) != meta['valid_books']:
        raise ValueError('Book split counts disagree with the manifest')
    if train_ids.intersection(dev_ids):
        raise ValueError('TRAIN and DEV share book identities')
    return meta, arrays


@torch.no_grad()
def cache_books(model, mem, lengths, context, windows, batch, seed):
    """Cache the frozen inherited logits, geometric keys and targets per book."""
    starts, total = book_starts(lengths, context, windows, seed)
    if total != len(mem):
        raise ValueError('Book offsets do not cover complete corpus')
    model.eval()
    geometry = model._r54_geometry()
    records = []
    for book in starts:
        items = []
        for first in range(0, len(book), batch):
            selected = book[first:first + batch]
            seq = np.stack([mem[int(p):int(p)+context+3] for p in selected])
            ids = torch.from_numpy(seq.astype(np.int64))
            observed = ids[:, :-3]
            logits, hidden, _ = model(observed)
            plus2, plus3 = model.future_logits(hidden)
            pairs = geometry.r54_last_pairs
            if pairs is None or pairs.shape[:2] != observed.shape:
                raise AssertionError('No causal geometric address capture')
            # Cache the inherited byte-head logits BEFORE successor correction.
            # R58 can thereby continue a *nonzero* R55 trust gate without
            # accidentally applying that inherited gate twice.
            base_logits = model.head(hidden).detach()
            items.append((base_logits, plus2.detach(), plus3.detach(),
                          pairs.clone().detach(), observed,
                          (ids[:, 1:-2], ids[:, 2:-1], ids[:, 3:])))
        records.append(items)
    return records


@contextmanager
def memory_arm(module, arm, wrong_queries):
    """Intervene only on the newest active rail; inherited rails stay exact."""
    joint = module.joint_weight is not None
    conditional = not joint and module.content_weight is not None
    if joint:
        previous = (module.joint_query_map,module.joint_value_control,
                    module.joint_query_byte_map)
        previous_weight = module.joint_weight.detach().clone()
        module.joint_query_map = wrong_queries if arm == 'wrong_address' else None
        module.joint_value_control = 'rotated' if arm == 'wrong_successor' else 'exact'
        module.joint_query_byte_map = (torch.arange(257,dtype=torch.long).roll(1)
                                       if arm == 'wrong_byte' else None)
    elif conditional:
        previous = (module.conditional_query_map,module.conditional_value_control)
        previous_weight = module.content_weight.detach().clone()
        module.conditional_query_map = wrong_queries if arm == 'wrong_address' else None
        module.conditional_value_control = 'rotated' if arm == 'wrong_successor' else 'exact'
    else:
        previous = (module.query_map,module.value_control)
        previous_weight = module.weight.detach().clone()
        module.query_map = wrong_queries if arm == 'wrong_address' else None
        module.value_control = 'rotated' if arm == 'wrong_successor' else 'exact'
    active_weight = (module.joint_weight if joint else
                     module.content_weight if conditional else module.weight)
    try:
        if arm == 'neutral':
            with torch.no_grad():
                active_weight.zero_()
        yield
    finally:
        if joint:
            (module.joint_query_map,module.joint_value_control,
             module.joint_query_byte_map) = previous
        elif conditional:
            module.conditional_query_map,module.conditional_value_control = previous
        else:
            module.query_map,module.value_control = previous
        if arm == 'neutral':
            with torch.no_grad():
                active_weight.copy_(previous_weight)

@torch.no_grad()
def score_books(module, records, arm, wrong_queries):
    output = []
    with memory_arm(module, arm, wrong_queries):
        for book in records:
            sums = np.zeros(len(METRICS), dtype=np.float64)
            counts = np.zeros(len(METRICS), dtype=np.int64)
            for logits, plus2, plus3, pairs, observed, targets in book:
                next_logits, _ = module(logits, pairs, observed)
                for horizon, (prediction, target) in enumerate(
                        zip((next_logits, plus2, plus3), targets)):
                    ce = F.cross_entropy(prediction.flatten(0, 1), target.flatten(),
                                         reduction='none').reshape_as(target)
                    top1 = prediction.argmax(-1).eq(target)
                    top5 = prediction.topk(5, -1).indices.eq(target.unsqueeze(-1)).any(-1)
                    i = horizon
                    sums[i] += ce.sum().item()
                    sums[3+i] += top1.sum().item()
                    sums[6+i] += top5.sum().item()
                    counts[[i,3+i,6+i]] += target.numel()
                    if horizon == 0:
                        for offset, mask in ((9, target.ne(32)), (12, target.eq(32))):
                            if mask.any():
                                sums[offset] += ce[mask].sum().item()
                                sums[offset+1] += top1[mask].sum().item()
                                sums[offset+2] += top5[mask].sum().item()
                                counts[offset:offset+3] += mask.sum().item()
            if np.any(counts == 0):
                raise ValueError('Book lacks space or non-space evaluation targets')
            output.append(dict(zip(METRICS, (sums / counts).tolist())))
    return output


def train_only_gate(module, optimizer, records, updates, lr, seed,
                    nonspace_weight=1.0):
    """Recycle frozen TRAIN windows; 194 inherited Adam slots receive no steps."""
    if len(optimizer.param_groups) != 1 or not 0 < lr < 1:
        raise ValueError('Requires one optimizer group and positive bounded LR')
    batches = [item for book in records for item in book]
    if not batches:
        raise ValueError('No verified TRAIN windows')
    generator = np.random.default_rng(seed)
    trained_parameter = (module.joint_weight if module.joint_weight is not None else
                         module.content_weight if module.content_weight is not None else
                         module.weight)
    original_lr = optimizer.param_groups[0]['lr']
    original_decay = optimizer.param_groups[0]['weight_decay']
    losses, examples = [], 0
    cursor, order = len(batches), None
    try:
        optimizer.param_groups[0]['lr'] = lr
        optimizer.param_groups[0]['weight_decay'] = 0.0
        for _ in range(updates):
            if cursor == len(batches):
                order = generator.permutation(len(batches))
                cursor = 0
            logits, _, _, pairs, observed, targets = batches[int(order[cursor])]
            cursor += 1
            optimizer.zero_grad(set_to_none=True)
            corrected, _ = module(logits, pairs, observed)
            loss = content_balanced_ce(corrected, targets[0], nonspace_weight)
            loss.backward()
            if (trained_parameter.grad is None or
                    not torch.isfinite(trained_parameter.grad).all()):
                raise FloatingPointError('Nonfinite trust gradient')
            optimizer.step()
            if not torch.isfinite(trained_parameter).all():
                raise FloatingPointError('Nonfinite trust weights')
            losses.append(float(loss.detach()))
            examples += targets[0].numel()
    finally:
        optimizer.param_groups[0]['lr'] = original_lr
        optimizer.param_groups[0]['weight_decay'] = original_decay
        optimizer.zero_grad(set_to_none=True)
    return {'first_ce':losses[0], 'last_ce':losses[-1],
            'mean_ce':float(np.mean(losses)), 'tokens_processed':examples,
            'updates':updates, 'learning_rate':lr,
            'nonspace_weight':nonspace_weight}


def audit_inheritance(saved, model, optimizer, new_gate, content_only=False,
                      joint_only=False):
    current = model.state_dict()
    errors = []
    for name, tensor in saved['model'].items():
        if not new_gate and not content_only and not joint_only and name == 'pair_successor.weight':
            continue  # The resumed gate, and only this gate, is allowed to learn.
        if name not in current or not torch.equal(tensor, current[name]):
            errors.append('Inherited tensor changed: ' + name)
    old = saved['optimizer']
    current_opt = optimizer.state_dict()
    if len(old['param_groups']) != 1 or len(current_opt['param_groups']) != 1:
        errors.append('Unified optimizer was split')
    old_ids = old['param_groups'][0]['params']
    new_ids = current_opt['param_groups'][0]['params']
    if new_ids[:len(old_ids)] != old_ids:
        errors.append('Inherited Adam parameter IDs reordered')
    if not new_gate and not content_only and not joint_only and len(new_ids) != len(old_ids):
        errors.append('Resumed gate changed parameter count')
    if (new_gate or content_only or joint_only) and len(new_ids) != len(old_ids)+1:
        errors.append('Exactly one new trust tensor expected')
    for pid, before in old['state'].items():
        if not new_gate and not content_only and not joint_only and pid == old_ids[-1]:
            continue  # Resumed gate Adam moments must advance, not stay frozen.
        after = current_opt['state'].get(pid)
        if after is None or any(k not in after or
            (not torch.equal(v, after[k]) if torch.is_tensor(v) else v != after[k])
            for k, v in before.items()):
            errors.append('Inherited Adam moment changed: ' + str(pid))
    if set(old['state']) - set(current_opt['state']):
        errors.append('Inherited Adam states disappeared')
    return {'pass':not errors, 'issues':errors,
            'inherited_tensors':len(saved['model']),
            'inherited_adam_states':len(old['state']),
            'retained_adam_states':sum(i in current_opt['state'] for i in old['state']),
            'new_trust_parameter':new_gate or content_only or joint_only,
            'conditional_specialist':content_only,
            'joint_specialist':joint_only,
            'gate_adam_state_present':bool(current_opt['state'].get(new_ids[-1]))}


def evaluate(arm_rows, bootstrap, seed, targets, raw_bytes, elapsed):
    baseline = arm_rows['neutral']
    comparisons = {}
    results = {}
    current_arms = COMPOUND_ARMS if 'wrong_byte' in arm_rows else ARMS
    for arm in current_arms:
        rows = arm_rows[arm]
        means = {metric:float(np.mean([r[metric] for r in rows])) for metric in METRICS}
        results[arm] = {'metrics':{**means,
            'bits_per_raw_byte':means['ce_1']*targets/raw_bytes/math.log(2),
            'bits_per_byte_or_eos_target':means['ce_1']/math.log(2)},
            'scored_targets_per_second':targets/max(elapsed[arm],1e-9)}
    # Bonferroni-correct across all predeclared candidate comparisons.
    controls = [('neutral','vs_parent'),
                ('wrong_address','vs_wrong_address'),
                ('wrong_successor','vs_wrong_successor')]
    if 'wrong_byte' in arm_rows:
        controls.append(('wrong_byte','vs_wrong_byte'))
    for reference, name in controls:
        comparisons[name] = book_comparison(arm_rows[reference],
                                             arm_rows['exact'],bootstrap,seed,len(controls))
    against_parent = comparisons['vs_parent']
    exact_address = comparisons['vs_wrong_address']
    exact_value = comparisons['vs_wrong_successor']
    required = [against_parent, exact_address, exact_value]
    if 'vs_wrong_byte' in comparisons:
        required.append(comparisons['vs_wrong_byte'])
    gate = (all(c['ce_1']['ci95_bonferroni'][0] > 0 and
                c['ce_nonspace_1']['ci95_bonferroni'][0] > 0
                for c in required) and
            against_parent['acc_nonspace_1']['mean_gain'] >= 0 and
            against_parent['top5_nonspace_1']['mean_gain'] >= 0 and
            all(against_parent[k]['mean_gain'] >= 0 for k in ('ce_2','ce_3')))
    return results, comparisons, bool(gate)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output-checkpoint', type=Path, required=True)
    p.add_argument('--output-report', type=Path, required=True)
    p.add_argument('--context', type=int, default=64)
    p.add_argument('--train-windows-per-book', type=int, default=8)
    p.add_argument('--dev-windows-per-book', type=int, default=8)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--updates', type=int, default=96)
    p.add_argument('--lr', type=float, default=.02)
    p.add_argument('--nonspace-weight', type=float, default=1.0,
                   help='R56 TRAIN-only content-byte weighting; 1 retains R55')
    p.add_argument('--conditional-content', action='store_true',
                   help='R58: append zero-init mass-preserving content expert')
    p.add_argument('--joint-content', action='store_true',
                   help='R59: append zero-init causal compound-key content rail')
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--expected-parent-sha256', default=None)
    args = p.parse_args()
    if min(args.context,args.train_windows_per_book,args.dev_windows_per_book,
           args.batch,args.updates,args.bootstrap,args.threads) < 1:
        p.error('Workload quantities must all be positive')
    if args.checkpoint.resolve() == args.output_checkpoint.resolve():
        p.error('Never replace the protected input checkpoint')
    if args.output_checkpoint.exists() or args.output_report.exists():
        p.error('Candidate output already exists; use a new path for continuation')
    torch.set_num_threads(args.threads)
    parent_sha = file_hash(args.checkpoint)
    saved = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    new_gate = not saved['config'].get('pair_successor_copy',False)
    if args.conditional_content and (new_gate or
                                     saved['config'].get('pair_successor_conditional',False)):
        raise ValueError('R58 needs existing non-conditional R55 successor gate')
    if args.joint_content and (args.conditional_content or
                              not saved['config'].get('pair_successor_conditional',False) or
                              saved['config'].get('pair_successor_joint',False)):
        raise ValueError('R59 needs existing non-joint R58 successor content rail')
    if new_gate and parent_sha != PROTECTED_SHA:
        raise ValueError('R55 initial migration requires exact protected step2993 SHA')
    if args.expected_parent_sha256 and parent_sha != args.expected_parent_sha256:
        raise ValueError('Requested parent SHA-256 mismatch')
    cfg = Config(**saved['config'])
    if args.context > cfg.context:
        raise ValueError('Requested window exceeds trained maximum context')
    meta, streams = checked_corpus(args.data_dir,saved)
    model = MirrorLM(cfg).eval()
    load_v25_compatible(model,saved['model'])
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.1)
    optimizer.load_state_dict(saved['optimizer'])
    if new_gate:
        model.enable_pair_successor(optimizer)
    if args.conditional_content:
        with torch.no_grad():
            parity_input = torch.tensor([[65,32,66,67,65,32,66,67]],dtype=torch.long)
            parity_before = model(parity_input)[0]
        model.enable_conditional_pair_successor(optimizer)
        with torch.no_grad():
            if not torch.equal(parity_before,model(parity_input)[0]):
                raise AssertionError('R58 zero-init migration changed inherited logits')
    if args.joint_content:
        with torch.no_grad():
            parity_input = torch.tensor([[65,32,66,67,65,32,66,67]],dtype=torch.long)
            parity_before = model(parity_input)[0]
        model.enable_joint_pair_successor(optimizer)
        with torch.no_grad():
            if not torch.equal(parity_before,model(parity_input)[0]):
                raise AssertionError('R59 zero-init migration changed inherited logits')
    if model.pair_successor is None:
        raise AssertionError('Successor gate not attached')
    for name, param in model.named_parameters():
        param.requires_grad_(name == ('pair_successor.joint_weight' if args.joint_content
                                      else 'pair_successor.content_weight'
                                      if args.conditional_content else
                                      'pair_successor.weight'))
    with memory_arm(model.pair_successor, 'neutral', None):
        train_cache = cache_books(model,streams['train'],meta['train_doc_lengths'],
                                  args.context,args.train_windows_per_book,args.batch,args.seed+1)
        dev_cache = cache_books(model,streams['valid'],meta['valid_doc_lengths'],
                                args.context,args.dev_windows_per_book,args.batch,args.seed)
    all_train = [item for book in train_cache for item in book]
    if not all_train:
        raise AssertionError('Empty TRAIN sample')
    # Optional unambiguous, query-only wrong-address permutation.
    wrong_queries = torch.arange(32640,dtype=torch.long).roll(137)
    begin = perf_counter()
    learning = train_only_gate(model.pair_successor,optimizer,train_cache,
                               args.updates,args.lr,args.seed,
                               args.nonspace_weight)
    training_seconds = perf_counter()-begin
    lineage = audit_inheritance(saved,model,optimizer,new_gate,args.conditional_content,
                                args.joint_content)
    if not lineage['pass']:
        raise AssertionError('Inherited lineage damaged: '+str(lineage['issues']))
    arm_rows, elapsed = {}, {}
    for arm in (COMPOUND_ARMS if args.joint_content else ARMS):
        tic = perf_counter()
        arm_rows[arm] = score_books(model.pair_successor,dev_cache,arm,wrong_queries)
        elapsed[arm] = perf_counter()-tic
    # Replay under neutral gate must be bit-identical after every intervention.
    replay = score_books(model.pair_successor,dev_cache,'neutral',wrong_queries)
    if replay != arm_rows['neutral']:
        raise AssertionError('Incorrect R55 intervention restoration')
    n = len(dev_cache)*args.dev_windows_per_book*args.context
    raw = sum(int((target[0] != 256).sum().item())
              for book in dev_cache for _,_,_,_,_,target in book)
    results, paired, passed = evaluate(arm_rows,args.bootstrap,args.seed,n,raw,elapsed)
    candidate = dict(saved)
    candidate['model'] = model.state_dict()
    candidate['optimizer'] = optimizer.state_dict()
    candidate['config'] = asdict(model.cfg)
    candidate['step'] = saved['step']+args.updates
    candidate['seen_tokens'] = saved['seen_tokens']+learning['tokens_processed']
    candidate['growth_history'] = list(saved.get('growth_history',[]))+[{
        'kind':('r59_compound_pair_byte_content_successor' if args.joint_content
                else 'r58_conditional_content_successor' if args.conditional_content
                else 'r55_train_only_causal_pair_successor'),
        'from_step':saved['step'],'to_step':candidate['step'],
        'parent_sha256':parent_sha,'seed':args.seed,
        'learning_rate':args.lr,'train_only_gate_updates':args.updates,
        'nonspace_weight':args.nonspace_weight,
        'train_windows_per_book':args.train_windows_per_book,
        'no_test_or_parent_weights_changed':True}]
    args.output_checkpoint.parent.mkdir(parents=True,exist_ok=True)
    temporary = args.output_checkpoint.with_name(args.output_checkpoint.name+'.tmp')
    try:
        torch.save(candidate,temporary)
        os.replace(temporary,args.output_checkpoint)
    finally:
        temporary.unlink(missing_ok=True)
    report = {'status':('R59_compound_successor_candidate_DEV_not_promoted'
                        if args.joint_content else
                        'R58_mass_preserving_content_candidate_DEV_not_promoted'
                        if args.conditional_content else
                        'R55_TRAIN_only_candidate_DEV_screen_not_promoted'),
        'parent_step':saved['step'],'candidate_step':candidate['step'],
        'parent_sha256':parent_sha,
        'candidate_sha256':file_hash(args.output_checkpoint),
        'train_sha256':meta['train_sha256'],'dev_sha256':meta['valid_sha256'],
        'test_opened':False,'books_train':len(train_cache),
        'books_dev':len(dev_cache),'window_context':args.context,
        'train_windows_per_book':args.train_windows_per_book,
        'dev_windows_per_book':args.dev_windows_per_book,
        'dev_targets_per_arm':n,'dev_literal_bytes_per_arm':raw,
        'bootstrap_unit':'whole DEV book','bootstrap_replicates':args.bootstrap,
        'familywise':('Bonferroni across exact vs parent/address/value/byte'
                      if args.joint_content else
                      'Bonferroni across exact vs parent/address/value'),
        'learning':learning,'train_seconds':training_seconds,
        'final_gate_weights':model.pair_successor.weight.detach().tolist(),
        'final_content_weights':(model.pair_successor.content_weight.detach().tolist()
                                 if model.pair_successor.content_weight is not None else None),
        'final_joint_weights':(model.pair_successor.joint_weight.detach().tolist()
                               if args.joint_content else None),
        'lineage':lineage,'results':results,'comparisons':paired,
        'dev_screen_pass':passed,'promoted':False,
        'next_gate':'Only after a positive, corrected book-disjoint DEV screen: '
                    'long-context regression and independent sealed TEST.'}
    args.output_report.parent.mkdir(parents=True,exist_ok=True)
    args.output_report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
