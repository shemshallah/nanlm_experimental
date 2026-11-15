#!/usr/bin/env python3
"""R43 diagnostic: read-only reciprocal bridge-strength sweep on DEV books.

This does not update checkpoints, optimizer states, weights on disk, or TEST.
Multiplier 1 is the learned gate, 0 is its zero-coupling control; signed
multipliers probe whether the learned signal is underpowered or misoriented.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poincare_frontier import Config, MirrorLM, file_hash, load_meta, load_v25_compatible
from compare_geometry_books import book_starts, bootstrap_gain, measure_books


def replacement_gate(original_gate, multiplier):
    if not math.isfinite(multiplier):
        raise ValueError('Multiplier must be finite')
    strength = math.tanh(8. * float(original_gate))
    target = max(-1. + 1e-7, min(1. - 1e-7, multiplier * strength))
    return math.atanh(target) / 8.


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--context', type=int, default=128)
    ap.add_argument('--windows-per-book', type=int, default=4)
    ap.add_argument('--multipliers', default='0,1,4,16,64,-4,-16')
    ap.add_argument('--bootstrap', type=int, default=2000)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if min(args.context, args.windows_per_book, args.bootstrap, args.threads) <= 0:
        raise ValueError('Window, thread, and bootstrap counts must be positive')
    multipliers = [float(x) for x in args.multipliers.split(',')]
    if len(multipliers) != len(set(multipliers)) or 1. not in multipliers or 0. not in multipliers:
        raise ValueError('Provide distinct multipliers including controls 0 and 1')
    if any(not math.isfinite(x) for x in multipliers):
        raise ValueError('Multipliers must be finite')
    torch.set_num_threads(args.threads)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    meta = load_meta(args.data_dir)
    if ckpt['data'] != meta or meta.get('split_unit') != 'whole_book':
        raise ValueError('Checkpoint and whole-book data do not match')
    if args.context > ckpt['config']['context']:
        raise ValueError('Probed context exceeds checkpoint')
    path = args.data_dir / 'valid.bin'
    expected = meta.get('valid_sha256') or meta.get('bin_valid_sha256')
    if not expected or file_hash(path) != expected:
        raise ValueError('DEV bytes do not match their manifest')
    mem = np.memmap(path, dtype=meta['token_dtype'], mode='r')
    records, n = book_starts(meta['valid_doc_lengths'], args.context,
                             args.windows_per_book, 2026)
    if n != len(mem):
        raise ValueError('DEV book offsets do not cover the corpus')
    model = MirrorLM(Config(**ckpt['config'])).eval()
    load_v25_compatible(model, ckpt['model'])
    if model.reciprocal is None or model.reciprocal.context_trust is None:
        raise ValueError('An R41+ reciprocal trust checkpoint is required')
    original = float(model.reciprocal.gate.detach().item())
    result = {}
    for multiplier in multipliers:
        with torch.no_grad():
            model.reciprocal.gate.fill_(replacement_gate(original, multiplier))
        rows = measure_books(model, mem, records, args.context, batch_size=4)
        result[str(multiplier)] = {
            'effective_gate': math.tanh(8. * float(model.reciprocal.gate.item())),
            'metrics': {k: float(np.mean([r[k] for r in rows])) for k in rows[0]},
            'books': rows,
        }
    with torch.no_grad():
        model.reciprocal.gate.fill_(original)
    model.reciprocal.enabled = False
    try:
        disabled = measure_books(model, mem, records, args.context, batch_size=4)
    finally:
        model.reciprocal.enabled = True
    result['disabled'] = {
        'effective_gate': None,
        'metrics': {k: float(np.mean([r[k] for r in disabled])) for k in disabled[0]},
        'books': disabled,
    }
    baseline = result['1.0']['books']
    neutral = result['0.0']['books']
    for label, entry in result.items():
        rows = entry.pop('books')
        entry['gain_vs_learned'] = {key: bootstrap_gain(
            [r[key] for r in baseline], [r[key] for r in rows],
            resamples=args.bootstrap, larger_better=key.startswith(('acc_','top5_')))
            for key in ('ce_1','ce_2','ce_3','ce_nonspace_1','acc_1','top5_1')}
        entry['gain_vs_zero'] = {key: bootstrap_gain(
            [r[key] for r in neutral], [r[key] for r in rows],
            resamples=args.bootstrap, larger_better=key.startswith(('acc_','top5_')))
            for key in ('ce_1','ce_nonspace_1')}
    report = {'status':'read_only_DEV_strength_probe_not_a_training_result',
              'checkpoint_step':ckpt['step'],'original_gate':original,
              'original_effective_gate':math.tanh(8.*original),
              'books':len(records),'positions':len(records)*args.windows_per_book*args.context,
              'bootstrap_unit':'whole book','multipliers':result,
              'warning':'DEV-only sweep; no promotion, TEST remains sealed'}
    output = json.dumps(report, indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(output, encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
