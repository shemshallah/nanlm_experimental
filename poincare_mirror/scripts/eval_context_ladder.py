#!/usr/bin/env python3
"""R32 aligned 256/4K/8K/32K raw-byte and geometric-memory ablation.

Use a verified book-disjoint/external raw byte file. The *same positions* are
scored at every context size; this is not a promotion gate by itself.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from eval_external_bytes import benchmark, digest
from poincare_frontier import Config, MirrorLM, load_v25_compatible


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
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    context_sizes = [int(v.strip()) for v in args.contexts.split(',')]
    if (not context_sizes or len(set(context_sizes)) != len(context_sizes) or
            min(context_sizes) < 4 or args.positions < 1 or args.batch < 1 or args.threads < 1):
        raise ValueError('Require distinct context lengths >=4 and positive workload limits')
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
    rows = {}
    for context in sorted(context_sizes):
        metrics, speed, hash_check = benchmark(model, data, context=context,
                                               positions=args.positions,
                                               batch_size=args.batch, locations=targets)
        assert position_sha == hash_check
        row = {'results': metrics, 'target_positions_per_second': speed}
        memories = [block.ff.symbiotic.memory for block in model.blocks
                    if block.use_moe and block.ff.symbiotic is not None
                    and block.ff.symbiotic.memory is not None]
        if args.ablate_geometric_memory and memories:
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
                finally:
                    for memory, gate in zip(memories, gates):
                        memory.gate.copy_(gate)
        rows[str(context)] = row
    report = {'checkpoint_sha256':checkpoint_sha,'checkpoint_step':saved['step'],
              'corpus_sha256':checksum,'aligned_target_positions_sha256':position_sha,
              'positions_per_context':args.positions,'attention_window':cfg.attention_window,
              'geometry_memory_chunk':cfg.geometry_memory_chunk,
              'seed':args.seed,'results_by_context':rows,
              'note':'Book-disjoint provenance, correct-history controls and held-out gates '
                     'are required before promoting candidate weights.'}
    rendered = json.dumps(report, indent=2) + '\n'
    print(rendered, end='')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == '__main__':
    main()
