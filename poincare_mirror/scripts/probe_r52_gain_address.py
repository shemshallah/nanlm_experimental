#!/usr/bin/env python3
"""FA2 R52: DEV-only reversible gain/address intervention on ONE checkpoint.

Tests whether the R50 near-null address result follows from a suppressed
reciprocal gate. No backpropagation, optimizer, corpus reassignment, TEST data,
or checkpoint write occurs. Every condition uses identical causal book windows.
"""
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poincare_frontier import file_hash, load_meta
from compare_geometry_books import METRICS, book_starts, measure_books
from probe_r47_selective_emission import book_comparison, literal_byte_count, load_model
from geometry_address_controls import address_ablation, fixed_permutations


# Fixed independently of DEV loss; both signs control for the learned negative gate.
GAIN_LEVELS = (-0.10, -0.03, 0.0, 0.03, 0.10)


@contextmanager
def reversible_cold_gain(model, gain):
    """Override only the inherited gate scalar, exactly restoring its bits.

    ``None`` uses the stored value. A requested gain is the *effective*
    tanh(8*g), not the raw checkpoint parameter g. No gradients are tracked.
    """
    if gain is not None and (not math.isfinite(gain) or abs(gain) >= 1):
        raise ValueError('Effective cold gain must be finite and in (-1,1)')
    if model.reciprocal is None:
        raise ValueError('Inherited reciprocal geometry is required')
    gate = model.reciprocal.gate
    original = gate.detach().clone()
    try:
        if gain is not None:
            with torch.no_grad():
                gate.fill_(math.atanh(gain) / 8.)
        yield float(torch.tanh(8. * gate).detach().item())
    finally:
        with torch.no_grad():
            gate.copy_(original)


def conditions():
    """Paired order and multiplicity are locked before seeing the DEV split."""
    result = [('inherited_exact', None, 'exact'),
              ('inherited_scrambled', None, 'all_scrambled'),
              ('reciprocal_disabled', None, 'reciprocal_disabled')]
    for i, value in enumerate(GAIN_LEVELS):
        for mode in ('exact', 'all_scrambled'):
            result.append((f'gain_{i}_{mode}', value, mode))
    return tuple(result)


def paired_rows(model, mem, records, context, batch, permutations):
    """Every temporary perturbation is removed, including after exceptions."""
    observed, elapsed, gain_used = {}, {}, {}
    for label, gain, mode in conditions():
        start = perf_counter()
        with reversible_cold_gain(model, gain) as effective:
            with address_ablation(model, mode, permutations):
                observed[label] = measure_books(model, mem, records, context, batch)
        elapsed[label] = perf_counter() - start
        gain_used[label] = effective
    replay = measure_books(model, mem, records, context, batch)
    if len(replay) != len(observed['inherited_exact']) or any(
            any(a[key] != b[key] for key in METRICS)
            for a, b in zip(replay, observed['inherited_exact'])):
        raise AssertionError('R52 failed byte-exact inherited-model replay')
    return observed, elapsed, gain_used


def build_report(observed, elapsed, gain_used, n, raw_bytes, bootstrap, seed):
    baseline = observed['inherited_exact']
    family = len(observed)-1
    out = {}
    for label, rows in observed.items():
        mean = {metric: float(np.mean([row[metric] for row in rows]))
                for metric in METRICS}
        out[label] = {
            'effective_cold_gain': gain_used[label],
            'metrics': {**mean,
                        'bits_per_byte_or_eos_target': mean['ce_1']/np.log(2),
                        'bits_per_literal_byte': mean['ce_1']*n/raw_bytes/np.log(2)},
            'scored_targets_per_second': n/max(elapsed[label], 1e-9),
            'vs_inherited_exact': book_comparison(baseline, rows, bootstrap,
                                                   seed, family),
        }
    # Positive advantage means the aligned addresses beat the misaligned
    # intervention. CI signs are reversed from R50's candidate-favoring form.
    specificity = {}
    for exact, scrambled in [('inherited_exact', 'inherited_scrambled')] + [
            (f'gain_{i}_exact', f'gain_{i}_all_scrambled')
            for i in range(len(GAIN_LEVELS))]:
        contrast = book_comparison(observed[exact], observed[scrambled],
                                   bootstrap, seed, len(GAIN_LEVELS)+1)
        def negate(metric):
            entry = contrast[metric]
            low, high = entry['ci95_bonferroni']
            return {'exact_advantage': -entry['mean_gain'],
                    'ci95_bonferroni': [-high, -low],
                    'books_exact_better': sum(
                        a[metric] < b[metric]
                        for a, b in zip(observed[exact], observed[scrambled]))}
        specificity[exact] = {
            'effective_cold_gain': gain_used[exact],
            'ce_1': negate('ce_1'),
            'ce_nonspace_1': negate('ce_nonspace_1'),
        }
    return out, specificity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--context', type=int, default=64)
    parser.add_argument('--windows-per-book', type=int, default=8)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.context, args.windows_per_book, args.batch, args.bootstrap,
           args.threads) < 1:
        parser.error('Workload quantities must be positive')
    torch.set_num_threads(args.threads)
    meta = load_meta(args.data_dir)
    if meta.get('split_unit') != 'whole_book' or meta.get('unit') != 'UTF8_bytes_plus_eos':
        raise ValueError('R52 requires whole-book byte/EOS DEV data')
    if file_hash(args.data_dir / 'split_manifest.json') != meta.get('split_manifest_sha256'):
        raise ValueError('Book split manifest checksum mismatch')
    path = args.data_dir / 'valid.bin'
    digest = meta.get('valid_sha256') or meta.get('bin_valid_sha256')
    if not digest or file_hash(path) != digest:
        raise ValueError('DEV checksum mismatch')
    mem = np.memmap(path, dtype=meta['token_dtype'], mode='r')
    records, covered = book_starts(meta['valid_doc_lengths'], args.context,
                                   args.windows_per_book, args.seed)
    if len(mem) != covered:
        raise ValueError('DEV book boundaries do not cover the corpus')
    model, saved = load_model(args.checkpoint, meta, args.context)
    original = {name: value.clone() for name, value in model.state_dict().items()}
    observed, elapsed, gain_used = paired_rows(model, mem, records,
        args.context, args.batch, fixed_permutations(args.seed))
    if any(not torch.equal(value, model.state_dict()[name])
           for name, value in original.items()):
        raise AssertionError('R52 changed inherited model tensors')
    n = len(records)*args.windows_per_book*args.context
    raw = literal_byte_count(mem, records, args.context)
    results, specificity = build_report(observed, elapsed, gain_used,
                                        n, raw, args.bootstrap, args.seed)
    report = {
        'status': 'read_only_FA2_R52_DEV_not_a_candidate_or_promotion',
        'checkpoint_sha256': file_hash(args.checkpoint), 'step': saved['step'],
        'dev_sha256': digest, 'books': len(records), 'context': args.context,
        'windows_per_book': args.windows_per_book,
        'prediction_targets_per_arm': n, 'literal_bytes_per_arm': raw,
        'bootstrap_replicates': args.bootstrap,
        'bootstrap_unit': 'whole independent DEV books', 'seed': args.seed,
        'multiplicity': 'Bonferroni across all arms vs inherited and all six address contrasts',
        'replay': 'bit-exact per-book inherited metrics and unchanged state_dict',
        'limitations': 'Forced gates are inference interventions, not trained successor models; '
                       'DEV selected windows only, not whole-book exhaustive scoring',
        'no_TEST_or_optimizer_updates': True,
        'conditions': results, 'address_specificity': specificity,
    }
    text = json.dumps(report, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    print(text, end='')


if __name__ == '__main__':
    main()
