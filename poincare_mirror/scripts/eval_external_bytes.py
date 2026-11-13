#!/usr/bin/env python3
"""Reproducible external UTF-8 byte +1/+2/+3 comparison for one checkpoint.

Unlike whole-book evaluation this script accepts a single raw held-out byte
file (for instance, a frozen WikiText-103 test rendering). Both model variants
must use the identical file hash, sampled positions, seed and 256-byte context.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poincare_frontier import Config, MirrorLM, load_v25_compatible


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


@torch.no_grad()
def benchmark(model, data, context=256, positions=2048, batch_size=8, seed=2026,
              locations=None, window_transform=None):
    """Evaluate fixed targets with no look-ahead in any model input."""
    if locations is None:
        if len(data) < context + 3 + positions:
            raise ValueError('Insufficient raw held-out bytes for the requested positions')
        candidates = np.arange(context, len(data) - 2)
        rng = np.random.default_rng(seed)
        locations = np.sort(rng.choice(candidates, size=positions, replace=False))
    else:
        locations = np.asarray(locations, dtype=np.int64)
        if (locations.ndim != 1 or len(locations) != positions or
                np.any(locations < context) or np.any(locations + 2 >= len(data))):
            raise ValueError('Aligned target positions outside available context/labels')
    positions_sha256 = hashlib.sha256(locations.astype('<i8').tobytes()).hexdigest()
    sums = np.zeros((3, 3), dtype=np.float64)  # nats CE, top1, top5
    elapsed = 0.0
    model.eval()
    for start in range(0, len(locations), batch_size):
        points = locations[start:start + batch_size]
        windows = np.stack([data[p-context:p] for p in points]).astype(np.int64)
        if window_transform is not None:
            replaced = np.asarray(window_transform(windows.copy(), points))
            if (replaced.shape != windows.shape or not np.issubdtype(replaced.dtype, np.integer)
                    or np.any(replaced < 0) or np.any(replaced >= model.cfg.vocab_size)):
                raise ValueError('History transformation must return valid same-shaped token IDs')
            windows = replaced
        x = torch.from_numpy(windows)
        y = torch.from_numpy(np.stack([data[p:p+3] for p in points]).astype(np.int64))
        begin = time.perf_counter()
        logits, hidden, _ = model(x)
        second, third = model.future_logits(hidden)
        elapsed += time.perf_counter() - begin
        for horizon, prediction in enumerate((logits, second, third)):
            scores = prediction[:, -1, :]
            targets = y[:, horizon]
            sums[horizon, 0] += F.cross_entropy(scores, targets, reduction='sum').item()
            sums[horizon, 1] += (scores.argmax(-1) == targets).sum().item()
            sums[horizon, 2] += (scores.topk(min(5, scores.shape[-1]), dim=-1).indices ==
                                 targets.unsqueeze(-1)).any(-1).sum().item()
    results = {}
    for horizon in range(3):
        ce, top1, top5 = sums[horizon] / positions
        results['+' + str(horizon + 1)] = {
            'ce_nats_per_byte': round(float(ce), 6),
            'bits_per_byte': round(float(ce / math.log(2)), 6),
            'perplexity_per_byte': round(float(math.exp(min(ce, 30))), 4),
            'top1': round(float(top1), 6),
            'top5': round(float(top5), 6),
        }
    return results, round(positions / elapsed, 2), positions_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--raw-test', type=Path, required=True)
    parser.add_argument('--expected-test-sha256', default=None)
    parser.add_argument('--expected-checkpoint-sha256', default=None)
    parser.add_argument('--expected-step', type=int, default=None)
    parser.add_argument('--context', type=int, default=256)
    parser.add_argument('--positions', type=int, default=2048)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    if args.context < 4 or args.positions < 1 or args.batch < 1 or args.threads < 1:
        raise ValueError('Context, positions, batch and threads must be positive')
    file_sha = digest(args.raw_test)
    checkpoint_sha = digest(args.checkpoint)
    if args.expected_test_sha256 and args.expected_test_sha256.lower() != file_sha:
        raise ValueError('Held-out corpus SHA-256 mismatch')
    if args.expected_checkpoint_sha256 and args.expected_checkpoint_sha256.lower() != checkpoint_sha:
        raise ValueError('Checkpoint SHA-256 mismatch (refusing to unpickle)')
    torch.set_num_threads(args.threads)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if args.expected_step is not None and saved['step'] != args.expected_step:
        raise ValueError('Wrong checkpoint step')
    cfg = Config(**saved['config'])
    if cfg.vocab_size != 257:
        raise ValueError('External raw-byte benchmark requires 257 byte/EOS vocabulary')
    model = MirrorLM(cfg)
    load_v25_compatible(model, saved['model'])
    raw = np.fromfile(args.raw_test, dtype=np.uint8)
    result, throughput, positions_hash = benchmark(model, raw, context=args.context,
                                                   positions=args.positions,
                                                   batch_size=args.batch, seed=args.seed)
    report = {'checkpoint_sha256':checkpoint_sha, 'checkpoint_step':saved['step'],
              'test_sha256':file_sha,
              'external_sha256_verified':bool(args.expected_test_sha256),
              'test_positions':args.positions, 'context_bytes':args.context,
              'target_positions_sha256':positions_hash,
              'seed':args.seed, 'threads':args.threads,
              'target_positions_per_second':throughput, 'results':result,
              'warning':'One external distribution, no book-disjoint or long-context promotion gate.'}
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
