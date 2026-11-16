#!/usr/bin/env python3
"""R50: DEV-only causal LM ablation of *address alignment* in FA2 geometry.

All runs use exactly one inherited checkpoint, identical book-safe contexts,
identical trainable tensors, and no optimizer. Only inference-time lookup IDs
or nonpersistent incidence buffers are changed and then restored. A scrambled
incidence is a consistently permuted chain complex, preserving d^2=0, but no
longer agrees with the model's fixed point/edge/triangle token ordering.
"""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poincare_frontier import file_hash, load_meta
from geometry_address_controls import (MODES, fixed_permutations,
                                       chain_relabelled, address_ablation)
from compare_geometry_books import METRICS, book_starts, measure_books
from probe_r47_selective_emission import book_comparison, literal_byte_count, load_model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--context', type=int, default=64)
    ap.add_argument('--windows-per-book', type=int, default=8)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--bootstrap', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if min(args.context, args.windows_per_book, args.batch, args.bootstrap,
           args.threads) < 1:
        ap.error('All workload quantities must be positive')
    torch.set_num_threads(args.threads)
    meta = load_meta(args.data_dir)
    if meta.get('split_unit') != 'whole_book' or meta.get('unit') != 'UTF8_bytes_plus_eos':
        raise ValueError('R50 requires whole-book UTF-8 byte/EOS development split')
    if file_hash(args.data_dir / 'split_manifest.json') != meta.get('split_manifest_sha256'):
        raise ValueError('DEV split manifest differs from checkpoint provenance')
    valid = args.data_dir / 'valid.bin'
    digest = meta.get('valid_sha256') or meta.get('bin_valid_sha256')
    if not digest or file_hash(valid) != digest:
        raise ValueError('DEV binary checksum mismatch')
    mem = np.memmap(valid, dtype=meta['token_dtype'], mode='r')
    records, covered = book_starts(meta['valid_doc_lengths'], args.context,
                                   args.windows_per_book, args.seed)
    if len(mem) != covered:
        raise ValueError('Book offsets do not cover DEV binary')
    model, saved = load_model(args.checkpoint, meta, args.context)
    permutations = fixed_permutations(args.seed)
    score_rows, elapsed = {}, {}
    for mode in MODES:
        begin = perf_counter()
        with address_ablation(model, mode, permutations):
            score_rows[mode] = measure_books(model, mem, records, args.context,
                                            args.batch)
        elapsed[mode] = perf_counter() - begin
    # Re-running the exact path after every ablation detects leaked hooks or
    # non-restored incidence even if all predictions move only very slightly.
    restored = measure_books(model, mem, records, args.context, args.batch)
    if any(any(a[k] != b[k] for k in METRICS)
           for a, b in zip(score_rows['exact'], restored)):
        raise AssertionError('R50 ablations failed to restore the exact model')
    n = len(records) * args.windows_per_book * args.context
    raw_bytes = literal_byte_count(mem, records, args.context)
    results = {}
    for mode in MODES:
        rows = score_rows[mode]
        mean = {k: float(np.mean([row[k] for row in rows])) for k in METRICS}
        results[mode] = {
            'metrics': {**mean, 'bits_per_byte_or_eos_target': mean['ce_1']/np.log(2),
                        'bits_per_literal_byte': mean['ce_1']*n/raw_bytes/np.log(2),
                        'bytes_per_token': raw_bytes/n},
            'scored_targets_per_second': n/max(elapsed[mode], 1e-9),
            'vs_exact': book_comparison(score_rows['exact'], rows, args.bootstrap,
                                        args.seed, len(MODES)-1),
        }
    out = {
        'status': 'read_only_DEV_R50_address_integrity_not_a_promotion',
        'checkpoint_sha256': file_hash(args.checkpoint), 'step': saved['step'],
        'dev_sha256': digest, 'books': len(records),
        'context': args.context, 'windows_per_book': args.windows_per_book,
        'prediction_targets_per_arm': n, 'literal_byte_targets_per_arm': raw_bytes,
        'seed': args.seed, 'bootstrap': args.bootstrap,
        'bootstrap_unit': 'independent whole DEV books (sampled windows)',
        'multiplicity': 'Bonferroni over five prespecified contrasts to exact',
        'interpretation': 'Significantly worse scrambled addressing indicates dependence on '
                          'learned geometric address alignment; it does not prove semantic understanding.',
        'restoration_check': 'exact prediction metrics identical after all ablations',
        'tests_not_run': 'independent TEST, training, checkpoint promotion',
        'results': results,
    }
    encoded = json.dumps(out, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding='utf-8')
    print(encoded, end='')


if __name__ == '__main__':
    main()
