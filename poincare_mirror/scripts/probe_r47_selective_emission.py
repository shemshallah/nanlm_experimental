#!/usr/bin/env python3
"""R47: read-only DEV ablation of per-layer reciprocal emission in FA2.

Every configuration keeps internal pair/triangle/apex coupling enabled and
shares a *single loaded candidate model*. Only the amount emitted to the
outer transformer residual changes. The protected parent is separately
loaded, never trained. TEST is intentionally not accepted as a split here;
the DEV sweep must not be used to tune on sealed books.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poincare_frontier import Config, MirrorLM, file_hash, load_meta, load_v25_compatible
from compare_geometry_books import METRICS, book_starts, measure_books


def emission_variants(layers):
    """Fixed before looking at DEV metrics; never optimized against TEST."""
    if layers < 2:
        raise ValueError('Selective-emission probe needs at least two blocks')
    middle = layers // 2
    return {
        'full': None,  # use the actual, unmodified checkpoint forward path
        'coupling_only': (0.,) * layers,
        'quarter': (.25,) * layers,
        'half': (.5,) * layers,
        'early_only': tuple(1. if i < middle else 0. for i in range(layers)),
        'late_only': tuple(0. if i < middle else 1. for i in range(layers)),
    }


def book_comparison(baseline, candidate, replicates=2000, seed=2026,
                    comparisons=1):
    """Book-resampled paired CIs; Bonferroni across the fixed R47 variants.

    Positive gain means the candidate improves the metric. These are
    descriptive intervals, not independent evidence after DEV selection.
    """
    if not baseline or len(baseline) != len(candidate) or len(baseline) < 2:
        raise ValueError('Need matched metric rows from at least two books')
    if min(replicates, comparisons) < 1:
        raise ValueError('Resamples and comparison count must be positive')
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(baseline), (replicates, len(baseline)))
    report = {}
    for metric in METRICS:
        first = np.asarray([row[metric] for row in baseline], dtype=np.float64)
        second = np.asarray([row[metric] for row in candidate], dtype=np.float64)
        signed = second-first if metric.startswith(('acc_', 'top5_')) else first-second
        draws = signed[sample].mean(1)
        lower_tail = .025 / comparisons
        report[metric] = {
            'baseline': float(first.mean()),
            'candidate': float(second.mean()),
            'mean_gain': float(signed.mean()),
            'books_improved': int((signed > 0).sum()),
            'ci95_bonferroni': [float(x) for x in np.quantile(
                draws, (lower_tail, 1.-lower_tail))],
        }
    return report


def diagnostic_gate(metrics):
    """Conservative screening only: this never promotes a checkpoint."""
    def lower(metric):
        return metrics[metric]['ci95_bonferroni'][0]
    return (lower('ce_1') > 0 and lower('ce_nonspace_1') > 0 and
            metrics['acc_nonspace_1']['mean_gain'] >= 0 and
            metrics['top5_nonspace_1']['mean_gain'] >= 0 and
            metrics['top5_1']['mean_gain'] >= -.005 and
            all(metrics[name]['mean_gain'] >= 0 for name in ('ce_2','ce_3')))


def literal_byte_count(mem, records, context):
    """Exclude EOS=256 from literal byte denominator, not from CE scoring."""
    count = 0
    for starts in records:
        for start in starts:
            targets = mem[int(start)+1:int(start)+context+1]
            if len(targets) != context:
                raise ValueError('Truncated target window')
            count += int(np.count_nonzero(targets != 256))
    if count == 0:
        raise ValueError('Cannot compute bits per raw byte with no bytes')
    return count


def load_model(checkpoint, metadata, context):
    data = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if data.get('data') != metadata:
        raise ValueError('Checkpoint and corpus manifests are not identical')
    if context > data['config']['context']:
        raise ValueError('Requested context exceeds checkpoint maximum')
    model = MirrorLM(Config(**data['config'])).eval()
    load_v25_compatible(model, data['model'])
    if model.reciprocal is None:
        raise ValueError('Candidate needs the existing reciprocal FA2 architecture')
    return model, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='R44 candidate checkpoint (read-only)')
    parser.add_argument('--parent', type=Path,
                        help='protected pre-R44 checkpoint (read-only)')
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--context', type=int, default=128)
    parser.add_argument('--windows-per-book', type=int, default=8)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.context, args.windows_per_book, args.batch,
           args.bootstrap, args.threads) <= 0:
        parser.error('All workload counts must be positive')
    torch.set_num_threads(args.threads)
    meta = load_meta(args.data_dir)
    if meta.get('split_unit') != 'whole_book' or meta.get('unit') != 'UTF8_bytes_plus_eos':
        raise ValueError('FA2 R47 requires whole-book literal-byte provenance')
    manifest_sha = meta.get('split_manifest_sha256')
    if not manifest_sha or file_hash(args.data_dir / 'split_manifest.json') != manifest_sha:
        raise ValueError('Whole-book split manifest is missing or has changed')
    candidate_hash = file_hash(args.checkpoint)
    parent_hash = file_hash(args.parent) if args.parent else None
    if parent_hash is not None and parent_hash == candidate_hash:
        raise ValueError('Candidate and protected parent cannot be the same checkpoint')
    valid = args.data_dir / 'valid.bin'
    expected = meta.get('valid_sha256') or meta.get('bin_valid_sha256')
    if not expected or file_hash(valid) != expected:
        raise ValueError('DEV bytes do not match the checkpoint manifest')
    mem = np.memmap(valid, dtype=meta['token_dtype'], mode='r')
    starts, total = book_starts(meta['valid_doc_lengths'], args.context,
                                args.windows_per_book, args.seed)
    if total != len(mem):
        raise ValueError('Book offsets do not cover the exact DEV split')
    predictions = len(starts)*args.windows_per_book*args.context
    raw_bytes = literal_byte_count(mem, starts, args.context)
    model, saved = load_model(args.checkpoint, meta, args.context)
    variants = emission_variants(model.cfg.layers)
    measured = {}
    try:
        for label, gains in variants.items():
            model.reciprocal.set_emission_gains(gains, model.cfg.layers)
            measured[label] = measure_books(model, mem, starts, args.context, args.batch)
    finally:
        model.reciprocal.set_emission_gains(None, model.cfg.layers)
    parent_rows = None
    parent_step = None
    if args.parent:
        parent, parent_saved = load_model(args.parent, meta, args.context)
        parent_rows = measure_books(parent, mem, starts, args.context, args.batch)
        parent_step = parent_saved['step']

    # Five nonreference schedules compete with the full or coupling-only
    # reference. All SIX schedules compete with the protected parent, so
    # that comparison requires a six-arm family-wise correction.
    reference_family_size = len(variants)-1
    parent_family_size = len(variants)
    results = {}
    for label, rows in measured.items():
        means = {key: float(np.mean([row[key] for row in rows]))
                 for key in METRICS}
        results[label] = {
            'gains': variants[label],
            'metrics': {**means,
                        'bits_per_byte_or_eos_target': means['ce_1']/np.log(2.),
                        'bits_per_literal_byte': means['ce_1']*predictions/raw_bytes/np.log(2.),
                        'bytes_per_token': raw_bytes/predictions},
            'vs_full': book_comparison(measured['full'], rows, args.bootstrap,
                                       args.seed, reference_family_size),
            'vs_coupling_only': book_comparison(measured['coupling_only'], rows,
                                                args.bootstrap, args.seed,
                                                reference_family_size),
        }
        if parent_rows is not None:
            results[label]['vs_parent'] = book_comparison(
                parent_rows, rows, args.bootstrap, args.seed, parent_family_size)
            results[label]['diagnostic_gate_vs_parent'] = diagnostic_gate(
                results[label]['vs_parent'])
    report = {
        'status': 'read_only_DEV_diagnostic_not_a_promotion',
        'candidate_checkpoint_sha256': candidate_hash,
        'parent_checkpoint_sha256': parent_hash,
        'candidate_step': saved['step'], 'protected_parent_step': parent_step,
        'split': 'valid', 'split_sha256': expected,
        'books': len(starts), 'context': args.context,
        'windows_per_book': args.windows_per_book,
        'sampling_unit': 'causal windows drawn within each independent DEV book',
        'all_book_positions_scored': False,
        'prediction_targets_per_variant': predictions,
        'actual_raw_byte_targets_per_variant': raw_bytes,
        'seed': args.seed, 'bootstrap_replicates': args.bootstrap,
        'bootstrap_unit': 'whole-book clusters of sampled causal windows',
        'simultaneous_ci_adjustment': {
            'vs_full_and_coupling_only':
                f'Bonferroni across {reference_family_size} nonreference arms',
            'vs_protected_parent':
                f'Bonferroni across {parent_family_size} candidate arms',
        },
        'byte_denominator': 'CE includes EOS=256; raw-byte BPB excludes EOS from denominator',
        'results': results,
        'decision': 'No automatic checkpoint promotion; sealed TEST not opened; '
                    'must pass independent long-context/TEST evaluation after DEV locking.',
    }
    encoded = json.dumps(report, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding='utf-8')
    print(encoded, end='')


if __name__ == '__main__':
    main()
