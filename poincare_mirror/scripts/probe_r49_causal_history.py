#!/usr/bin/env python3
"""FA2 R49: matched correct-vs-wrong history and reciprocal specificity probe.

DEV only. Both checkpoints are read-only. For each fixed target, replace the
older prefix with a prefix from a *different* DEV book, retain the final
`near` true bytes, and score exactly the same next literal byte. Subtract the
history advantage with reciprocal disabled from that with reciprocal enabled.
This difference-in-differences separates reciprocal-specific sensitivity
from the inherited transformer's ordinary history sensitivity; it is not a
claim of a causal effect of the geometry's internal address semantics.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poincare_frontier import file_hash, load_meta
from compare_geometry_books import book_starts
from probe_r47_selective_emission import load_model


MODES = ('full', 'coupling_only', 'reciprocal_disabled')


def paired_records(lengths, context, near, windows_per_book, seed):
    """(book, true_start, donor_book, donor_start), with exact book isolation."""
    if not (0 < near < context):
        raise ValueError('near must be strictly between zero and context')
    if len(lengths) < 2 or min(lengths) <= context + 2:
        raise ValueError('Need at least two full-window independent books')
    records, total = book_starts(lengths, context, windows_per_book, seed)
    offsets = np.cumsum(np.r_[0, np.asarray(lengths, dtype=np.int64)])
    rng = np.random.default_rng(seed + 4901)
    far = context - near
    result = []
    for i, starts in enumerate(records):
        group = []
        for start in starts:
            donor_book = (i + int(rng.integers(1, len(lengths)))) % len(lengths)
            donor_limit = int(lengths[donor_book]) - far - 1
            if donor_limit < 1:
                raise ValueError('Donor book has no interior prefix')
            donor_start = int(offsets[donor_book]) + int(rng.integers(0, donor_limit))
            group.append((i, int(start), donor_book, donor_start))
        result.append(group)
    return result, total


def paired_inputs(mem, position, donor_position, context, near):
    """Return visible causal inputs and one fixed, never-fed next-byte target."""
    far = context - near
    true = np.asarray(mem[position:position + context], dtype=np.int64)
    donor = np.asarray(mem[donor_position:donor_position + far], dtype=np.int64)
    if len(true) != context or len(donor) != far:
        raise ValueError('Truncated true or donor history')
    wrong = np.concatenate((donor, true[far:]))
    target = int(mem[position + context])
    if not np.array_equal(true[far:], wrong[far:]):
        raise AssertionError('Recent true suffix must stay identical')
    return true, wrong, target


@torch.no_grad()
def score_books(model, mem, groups, context, near, batch_size):
    """Pair genuine and donor input in the same forward batch, score last step."""
    rows = []
    for group in groups:
        count = 0
        totals = np.zeros(6, dtype=np.float64)
        for index in range(0, len(group), batch_size):
            chunk = group[index:index + batch_size]
            paired = [paired_inputs(mem, pos, donor, context, near)
                      for _, pos, _, donor in chunk]
            true = torch.from_numpy(np.stack([x[0] for x in paired]))
            wrong = torch.from_numpy(np.stack([x[1] for x in paired]))
            target = torch.tensor([x[2] for x in paired], dtype=torch.long)
            logits = model(torch.cat((true, wrong), dim=0))[0][:, -1, :]
            n = len(chunk)
            actual = torch.cat((target, target), dim=0)
            ce = F.cross_entropy(logits, actual, reduction='none')
            top1 = logits.argmax(-1) == actual
            top5 = (logits.topk(5, dim=-1).indices == actual[:, None]).any(-1)
            totals += np.array([ce[:n].sum().item(), ce[n:].sum().item(),
                                top1[:n].sum().item(), top1[n:].sum().item(),
                                top5[:n].sum().item(), top5[n:].sum().item()])
            count += n
        ce_true, ce_wrong, a1_true, a1_wrong, a5_true, a5_wrong = totals / count
        rows.append({'ce_true': float(ce_true), 'ce_wrong': float(ce_wrong),
                     'correct_history_advantage': float(ce_wrong - ce_true),
                     'top1_true': float(a1_true), 'top1_wrong': float(a1_wrong),
                     'top5_true': float(a5_true), 'top5_wrong': float(a5_wrong)})
    return rows


def book_bootstrap(values, replicates=2000, seed=2026, comparisons=1):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Need at least two finite per-book contrasts')
    if min(replicates, comparisons) < 1:
        raise ValueError('Invalid bootstrap settings')
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[sample].mean(axis=1)
    alpha_tail = .025 / comparisons
    return {'mean': float(values.mean()),
            'books_positive': int((values > 0).sum()),
            'ci95_bonferroni': [float(v) for v in np.quantile(
                means, (alpha_tail, 1 - alpha_tail))]}


def evaluate_model(checkpoint, meta, mem, groups, context, near, batch_size):
    model, saved = load_model(checkpoint, meta, context)
    step = int(saved['step'])
    del saved
    gc.collect()
    modes = {}
    try:
        for mode in MODES:
            model.reciprocal.enabled = mode != 'reciprocal_disabled'
            gains = (0.,) * model.cfg.layers if mode == 'coupling_only' else None
            model.reciprocal.set_emission_gains(gains, model.cfg.layers)
            modes[mode] = score_books(model, mem, groups, context, near, batch_size)
    finally:
        model.reciprocal.enabled = True
        model.reciprocal.set_emission_gains(None, model.cfg.layers)
        del model
        gc.collect()
    return step, modes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parent', type=Path, required=True)
    ap.add_argument('--candidate', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--context', type=int, default=128)
    ap.add_argument('--near', type=int, default=32)
    ap.add_argument('--windows-per-book', type=int, default=8)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--bootstrap', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if min(args.context, args.windows_per_book, args.batch,
           args.bootstrap, args.threads) < 1:
        ap.error('All workload quantities must be positive')
    if not 0 < args.near < args.context:
        ap.error('Need 0 < near < context')
    torch.set_num_threads(args.threads)
    meta = load_meta(args.data_dir)
    if meta.get('split_unit') != 'whole_book' or meta.get('unit') != 'UTF8_bytes_plus_eos':
        raise ValueError('Require independent whole-book UTF-8 byte/EOS split')
    if file_hash(args.data_dir / 'split_manifest.json') != meta.get('split_manifest_sha256'):
        raise ValueError('Split manifest has changed')
    valid = args.data_dir / 'valid.bin'
    digest = meta.get('valid_sha256') or meta.get('bin_valid_sha256')
    if not digest or file_hash(valid) != digest:
        raise ValueError('DEV binary has changed')
    mem = np.memmap(valid, dtype=meta['token_dtype'], mode='r')
    records, total = paired_records(meta['valid_doc_lengths'], args.context,
                                    args.near, args.windows_per_book, args.seed)
    if total != len(mem):
        raise ValueError('Independent-book offsets do not cover DEV binary')
    ph, ch = file_hash(args.parent), file_hash(args.candidate)
    if ph == ch:
        raise ValueError('Parent and candidate must be distinct checkpoints')
    steps = {}
    scored = {}
    for name, checkpoint in (('parent', args.parent), ('candidate', args.candidate)):
        steps[name], scored[name] = evaluate_model(checkpoint, meta, mem,
            records, args.context, args.near, args.batch)

    results = {}
    # Exactly four preregistered geometry-specific contrasts:
    # two non-disabled modes on each of the two fixed checkpoints.
    for name, modes in scored.items():
        summary = {}
        disabled = np.array([r['correct_history_advantage']
                             for r in modes['reciprocal_disabled']])
        for mode, rows in modes.items():
            adv = np.array([r['correct_history_advantage'] for r in rows])
            report = {'metrics': {
                key: float(np.mean([r[key] for r in rows])) for key in rows[0]},
                'correct_history_advantage_unadjusted': book_bootstrap(
                    adv, args.bootstrap, args.seed, 1)}
            if mode != 'reciprocal_disabled':
                report['reciprocal_specific_history_gain'] = book_bootstrap(
                    adv - disabled, args.bootstrap, args.seed, 4)
            summary[mode] = report
        results[name] = summary
    out = {
        'status': 'read_only_DEV_causal_history_probe_not_a_promotion',
        'parent_sha256': ph, 'candidate_sha256': ch,
        'parent_step': steps['parent'], 'candidate_step': steps['candidate'],
        'dev_sha256': digest, 'books': len(records),
        'contexts_per_book': args.windows_per_book, 'context': args.context,
        'near_true_bytes': args.near, 'wrong_book_prefix_bytes': args.context - args.near,
        'next_byte_targets_per_mode': len(records) * args.windows_per_book,
        'seed': args.seed, 'bootstrap': args.bootstrap,
        'bootstrap_unit': 'independent book (sampled target windows)',
        'multiplicity': 'four reciprocal-specific contrasts; 95% two-sided Bonferroni',
        'question': 'Does reciprocal geometry improve correct-history specificity '
                    'over the same transformer with reciprocal disabled?',
        'warning': 'This contrast isolates reciprocal processing, not unique '
                   'C(256,2)/C(256,3) address semantics; no TEST use or promotion.',
        'results': results,
    }
    encoded = json.dumps(out, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding='utf-8')
    print(encoded, end='')


if __name__ == '__main__':
    main()
