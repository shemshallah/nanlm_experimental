"""The single canonical FA2 model entry point (R68 user-directed promotion).

The promoted checkpoint is an external recovery artifact, not a second model.
The inherited backbone, geometric memory, and all Adam moments stay together.
"""
import json
import os
from pathlib import Path
import sys

import torch

# The inherited FA2 modules use absolute sibling imports, even when users
# import this entry point as poincare_mirror.fa2_active from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'scripts'))
from poincare_frontier import Config, MirrorLM, file_hash, load_v25_compatible


ACTIVE_MANIFEST = Path(__file__).with_name('FA2_ACTIVE.json')


def resolve_active_checkpoint(path=None, manifest_path=ACTIVE_MANIFEST):
    """Resolve and hash-check the one active FA2 checkpoint, fail closed."""
    manifest_path = Path(manifest_path)
    record = json.loads(manifest_path.read_text())
    if record['architecture'] != 'FA2_UNIFIED' or record['active_step'] != 4209:
        raise ValueError('Unsupported active FA2 lineage')
    declared = record['checkpoint_sha256']
    filename = record['checkpoint_filename']
    if path is not None:
        candidates = [Path(path)]
    elif os.environ.get('FA2_ACTIVE_CHECKPOINT'):
        candidates = [Path(os.environ['FA2_ACTIVE_CHECKPOINT'])]
    else:
        candidates = [Path.cwd() / filename,
                      Path.cwd() / 'checkpoint' / filename,
                      manifest_path.parent / filename,
                      manifest_path.parent.parent / 'checkpoint' / filename]
    for candidate in candidates:
        if candidate.is_file():
            if file_hash(candidate) != declared:
                raise ValueError('Active FA2 checkpoint SHA-256 mismatch')
            return candidate, record
    raise FileNotFoundError('FA2 active checkpoint not installed. Unpack the R68 '
                            'recovery archive or set FA2_ACTIVE_CHECKPOINT')


def load_active_fa2(path=None, *, manifest_path=ACTIVE_MANIFEST,
                    include_optimizer=False):
    """Load the same unified model and optionally the complete optimizer."""
    checkpoint, record = resolve_active_checkpoint(path, manifest_path)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved.get('step') != record['active_step'] or not saved['config'].get(
            'pair_successor_space_rank_guard', False):
        raise ValueError('Active FA2 configuration or training step mismatched')
    model = MirrorLM(Config(**saved['config'])).eval()
    load_v25_compatible(model, saved['model'])
    if model.pair_successor is None or model.pair_successor.joint_weight is None:
        raise ValueError('Active FA2 missing joint geometric/lexical memory')
    optimizer = None
    if include_optimizer:
        from scripts.train_r68_multidepth import restore_joint_optimizer
        optimizer = restore_joint_optimizer(model, saved)
    return model, optimizer, saved
