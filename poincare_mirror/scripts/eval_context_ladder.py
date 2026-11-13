#!/usr/bin/env python3
"""R32 aligned raw-byte long-history and geometric-pathway evaluation.

Use a checksum-verified book-disjoint/external raw byte file. Every context,
ablation and unrelated-document control scores identical target positions.
This diagnostic is not, by itself, a checkpoint promotion gate.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from eval_external_bytes import benchmark, digest
from poincare_frontier import Config, MirrorLM, load_v25_compatible


def unrelated_history_transform(donor, positions, starts, maximum, context, retain):
    """Substitute ONLY distant history; leave the entire recent tail untouched.

    Donor windows are chosen once at the longest context, then right-aligned
    for every shorter context. Each original position has a stable donor.
    Callers must supply a genuinely independent donor book.
    """
    if (len(positions) != len(starts) or len(set(map(int, positions))) != len(positions)
            or context > maximum or retain < 1 or context <= retain
            or np.any(np.asarray(starts) < 0)
            or np.any(np.asarray(starts) + maximum > len(donor))):
        raise ValueError('Invalid aligned unrelated-history experiment')
    lookup = dict(zip(map(int, positions), map(int, starts)))

    def substitute(windows, batch_positions):
        output = windows.copy()
        for i, position in enumerate(batch_positions):
            start = lookup[int(position)] + maximum - context
            output[i, :-retain] = donor[start:start + context - retain]
        return output

    return substitute


def ce_improvement(ablated, original):
    """Positive values mean the original intact system predicts better."""
    return {h: round(ablated[h]['ce_nats_per_byte'] -
                     original[h]['ce_nats_per_byte'], 6)
            for h in ('+1', '+2', '+3')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--raw-test', type=Path, required=True)
    parser.add_argument('--expected-test-sha256', required=True)
    parser.add_argument('--expected-checkpoint-sha256')
    parser.add_argument('--contexts', default='256,4096,8192,32768')
    parser.add_argument('--positions', type=int, default=16)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--ablate-geometric-memory', action='store_true')
    parser.add_argument('--ablations', action='store_true',
                        help='Evaluate temporal geometric memory AND the full symbiotic path')
    parser.add_argument('--unrelated-history', type=Path,
                        help='Independent donor document for matched-history replacement')
    parser.add_argument('--expected-unrelated-sha256',
                        help='Mandatory SHA-256 whenever unrelated history is supplied')
    parser.add_argument('--preserve-recent-bytes', type=int, default=256)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    context_sizes = [int(v.strip()) for v in args.contexts.split(',')]
    if (not context_sizes or len(set(context_sizes)) != len(context_sizes) or
            min(context_sizes) < 4 or args.positions < 1 or args.batch < 1 or
            args.threads < 1 or args.preserve_recent_bytes < 1):
        raise ValueError('Require distinct context lengths >=4 and positive workload limits')
    if bool(args.unrelated_history) != bool(args.expected_unrelated_sha256):
        raise ValueError('Independent donor path and expected SHA-256 must be provided together')
    checksum = digest(args.raw_test)
    if checksum != args.expected_test_sha256.lower():
        raise ValueError('Raw held-out file checksum mismatch')
    checkpoint_sha = digest(args.checkpoint)
    if args.expected_checkpoint_sha256 and checkpoint_sha != args.expected_checkpoint_sha256.lower():
        raise ValueError('Checkpoint checksum mismatch')
    torch.set_num_threads(args.threads)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = Config(**saved['config'])
    if cfg.vocab_size != 257 or max(context_sizes) > cfg.context:
        raise ValueError('Requires fixed byte/EOS vocabulary and checkpoint max context')
    model = MirrorLM(cfg).eval()
    load_v25_compatible(model, saved['model'])
    data = np.fromfile(args.raw_test, dtype=np.uint8)
    largest = max(context_sizes)
    if len(data) < largest + 3 + args.positions:
        raise ValueError('Insufficient held-out bytes for aligned long-context evaluation')
    rng = np.random.default_rng(args.seed)
    targets = np.sort(rng.choice(np.arange(largest, len(data)-2),
                                 args.positions, replace=False))
    position_sha = hashlib.sha256(targets.astype('<i8').tobytes()).hexdigest()
    donor = None
    donor_starts = None
    donor_checksum = None
    if args.unrelated_history is not None:
        donor_checksum = digest(args.unrelated_history)
        if donor_checksum != args.expected_unrelated_sha256.lower() or donor_checksum == checksum:
            raise ValueError('Donor checksum mismatch or donor equals target document')
        donor = np.fromfile(args.unrelated_history, dtype=np.uint8)
        if len(donor) < largest:
            raise ValueError('Independent donor document is too short for maximum context')
        donor_starts = rng.integers(0, len(donor)-largest+1, size=len(targets))
    rows = {}
    memories = [block.ff.symbiotic.memory for block in model.blocks
                if block.use_moe and block.ff.symbiotic is not None
                and block.ff.symbiotic.memory is not None]
    geometry_moes = [block.ff for block in model.blocks
                     if block.use_moe and block.ff.symbiotic is not None]
    for context in sorted(context_sizes):
        metrics, speed, hash_check = benchmark(model, data, context=context,
                                               positions=args.positions,
                                               batch_size=args.batch, locations=targets)
        assert position_sha == hash_check
        row = {'results': metrics, 'target_positions_per_second': speed}
        if donor is not None and context > args.preserve_recent_bytes:
            transform = unrelated_history_transform(
                donor, targets, donor_starts, largest, context, args.preserve_recent_bytes)
            unrelated, _, donor_position_sha = benchmark(
                model, data, context=context, positions=args.positions,
                batch_size=args.batch, locations=targets, window_transform=transform)
            assert donor_position_sha == position_sha
            row['unrelated_history'] = unrelated
            row['correct_history_ce_advantage_nats'] = ce_improvement(unrelated, metrics)
        if (args.ablate_geometric_memory or args.ablations) and memories:
            with torch.no_grad():
                gates = [memory.gate.clone() for memory in memories]
                try:
                    for memory in memories:
                        memory.gate.zero_()
                    off, _, off_sha = benchmark(model, data, context=context,
                                                positions=args.positions,
                                                batch_size=args.batch, locations=targets)
                    assert off_sha == position_sha
                    row['no_geometric_memory'] = off
                    row['temporal_memory_ce_gain_nats'] = ce_improvement(off, metrics)
                finally:
                    for memory, gate in zip(memories, gates):
                        memory.gate.copy_(gate)
        if args.ablations and geometry_moes:
            original = [moe.symbiotic for moe in geometry_moes]
            try:
                for moe in geometry_moes:
                    moe.symbiotic = None
                off, _, off_sha = benchmark(
                    model, data, context=context, positions=args.positions,
                    batch_size=args.batch, locations=targets)
                assert off_sha == position_sha
                row['no_symbiotic_geometry'] = off
                row['symbiotic_geometry_ce_gain_nats'] = ce_improvement(off, metrics)
            finally:
                for moe, geometry in zip(geometry_moes, original):
                    moe.symbiotic = geometry
        rows[str(context)] = row
    report = {'checkpoint_sha256':checkpoint_sha,'checkpoint_step':saved['step'],
              'corpus_sha256':checksum,'aligned_target_positions_sha256':position_sha,
              'positions_per_context':args.positions,'attention_window':cfg.attention_window,
              'geometry_memory_chunk':cfg.geometry_memory_chunk,
              'seed':args.seed,'results_by_context':rows,
              'unrelated_history_sha256':donor_checksum,
              'preserved_recent_bytes':args.preserve_recent_bytes,
              'history_control':'Independent donor replaces older prefix only; '
                                'recent bytes, prediction positions and labels are fixed'
                                if donor is not None else 'Not run',
              'metric_direction':'Positive CE advantage/gain means the intact model is better',
              'note':'Require documented book-disjoint provenance and independent held-out '
                     'promotion gates; synthetic smoke tests do not establish accuracy.'}
    rendered = json.dumps(report, indent=2) + '\n'
    print(rendered, end='')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == '__main__':
    main()
