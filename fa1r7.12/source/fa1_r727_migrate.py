#!/usr/bin/env python3
"""Convert R7.24 into one restorable R7.27 checkpoint, without training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fa1_r727_checkpoint import digest, save_unpromoted, load_unified
from fa1_r727_optimizer import load_r727_geometric, _deep_equal


@torch.no_grad()
def run(a):
    source_sha = digest(a.candidate)
    accepted = Path(a.root) / 'checkpoint/pal_active.pt'
    accepted_sha = digest(accepted)
    model, optimizer, _, _, metadata = load_r727_geometric(a.candidate, a.root, a.threads)
    output = save_unpromoted(a.output, model, optimizer, a.root, a.candidate, metadata)
    restored, recovered_optimizer, _, _, restored_meta = load_unified(a.output, a.root, a.threads)
    for key, tensor in model.state_dict().items():
        if not torch.equal(tensor, restored.state_dict()[key]):
            raise AssertionError('recovered R7 weight differs at ' + key)
    if not _deep_equal(optimizer.state_dict(), recovered_optimizer.state_dict()):
        raise AssertionError('R7.27 optimizer-state round trip changed inherited moments')
    if digest(a.candidate) != source_sha or digest(accepted) != accepted_sha:
        raise AssertionError('protected ancestor or experimental parent changed')
    result = {
        'format': 'FA1_R7_27_BITWISE_MIGRATION_REPORT_V1',
        'new_checkpoint': str(a.output),
        'new_checkpoint_sha256': output['sha256'],
        'source_checkpoint_sha256': source_sha,
        'accepted_checkpoint_sha256': accepted_sha,
        'bytes': output['bytes'],
        'model_tensor_count': len(model.state_dict()),
        'parameter_groups': len(optimizer.param_groups),
        'single_optimizer_state': True,
        'weights_round_trip_bitwise_equal': True,
        'optimizer_round_trip_bitwise_equal': True,
        'accepted_step': 14416, 'promoted': False, 'training_updates': 0,
        'corpus_metadata_locked': restored_meta['corpus_metadata_sha256'],
    }
    if a.report:
        report = Path(a.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', default='')
    parser.add_argument('--threads', type=int, default=2)
    run(parser.parse_args())
