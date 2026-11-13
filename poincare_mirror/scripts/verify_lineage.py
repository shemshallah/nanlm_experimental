#!/usr/bin/env python3
"""Fail-closed recovery of the single Poincare Mirror optimizer checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--expected-sha256')
    ap.add_argument('--expected-step', type=int, required=True)
    ap.add_argument('--minimum-adam-states', type=int, default=63)
    ap.add_argument('--report', type=Path)
    args = ap.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    digest = hashlib.sha256()
    with args.checkpoint.open('rb') as input_file:
        for block in iter(lambda: input_file.read(4 * 1024 * 1024), b''):
            digest.update(block)
    actual_sha = digest.hexdigest()
    if args.expected_sha256 and actual_sha != args.expected_sha256:
        raise ValueError(f'Checkpoint checksum mismatch: {actual_sha}')

    # PyTorch checkpoint pickle decoding happens only after the SHA-256 gate.
    # Checkpoint must come from our trusted canonical release; do not load
    # arbitrary unauthenticated .pt files.
    import torch
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = saved['config']
    for field, expected in [('vocab_size',257), ('dim',192), ('layers',6),
                            ('heads',16), ('pairs',32), ('expert_hidden',256),
                            ('context',4096)]:
        if config.get(field) != expected:
            raise ValueError(f'Incompatible architecture {field}: {config.get(field)}')
    if saved['step'] != args.expected_step:
        raise ValueError(f'Unexpected checkpoint step {saved["step"]}, expected {args.expected_step}')
    optimizer = saved['optimizer']
    adam_states = optimizer.get('state', {})
    if len(adam_states) < args.minimum_adam_states:
        raise ValueError(f'Missing Adam optimizer history: {len(adam_states)} states')
    for name in ('model', 'rng_numpy', 'rng_torch', 'rng_python', 'seen_tokens', 'data'):
        if name not in saved:
            raise ValueError(f'Missing lineage item {name}')
    if saved['data']['vocab_size'] != 257 or saved['data'].get('unit') != 'UTF8_bytes_plus_eos':
        raise ValueError('Fixed byte-tokenizer identity violated')
    report = dict(sha256=actual_sha, size_bytes=args.checkpoint.stat().st_size,
                  step=saved['step'], seen_tokens=saved['seen_tokens'],
                  adam_states=len(adam_states), parameter_tensors=len(saved['model']),
                  architecture=config, train_sha256=saved['data'].get('train_sha256'),
                  valid_sha256=saved['data'].get('valid_sha256'),
                  test_sha256=saved['data'].get('test_sha256'))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'architecture'}, sort_keys=True))


if __name__ == '__main__':
    main()
