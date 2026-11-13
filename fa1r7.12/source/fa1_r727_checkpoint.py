#!/usr/bin/env python3
"""Standalone R7.27 learned-state serialization: one model, one optimizer.

R7.27 keeps the original R7.13 accepted checkpoint immutable. The migrated
geometry candidate is *unpromoted*, irrespective of conversion success.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

import torch

from fa1_r721_unified import blank
from fa1_r722_unified import UnifiedR722
from fa1_r723_diamond import UnifiedR723Diamond
from fa1_r724_geometry import GeometricDiamond
from fa1_r727_optimizer import UnifiedR7Optimizer, FORMAT as OPTIMIZER_FORMAT


FORMAT = 'FA1_R7_27_SINGLE_MODEL_SINGLE_HYBRID_OPTIMIZER_V1'
CORPUS_FILES = ('vocab.words.txt', 'token_heads.u16', 'tree.i32')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def corpus_digests(root):
    return {key: digest(Path(root) / 'corpus' / key) for key in CORPUS_FILES}


def save_unpromoted(path, model, optimizer, root, source_path, source_metadata=None,
                    candidate_trained=False, training_round='', training_updates=0,
                    development_gate_passed=False, architecture_round=''):
    path = Path(path).resolve()
    if path == Path(source_path).resolve() or path == (Path(root) / 'checkpoint/pal_active.pt').resolve():
        raise ValueError('cannot overwrite protected accepted/candidate checkpoint')
    if path.exists():
        raise FileExistsError('R7.27 migration requires a new output path: ' + str(path))
    state = optimizer.state_dict()
    if state['format'] != OPTIMIZER_FORMAT:
        raise ValueError('expected one verified R7.27 unified optimizer')
    if candidate_trained and (not training_round or training_updates <= 0):
        raise ValueError('a trained successor must declare its round and positive update count')
    if not candidate_trained and (training_updates != 0 or development_gate_passed):
        raise ValueError('an untrained migration cannot claim training or a development pass')
    obj = {
        'format': FORMAT,
        'accepted_step': 14416,
        'promoted': False,
        'candidate_trained': bool(candidate_trained),
        'training_round':str(training_round),
        'candidate_training_updates':int(training_updates),
        'development_gate_passed':bool(development_gate_passed),
        'architecture_round':str(architecture_round),
        'source_checkpoint_sha256': digest(source_path),
        'corpus_metadata_sha256': corpus_digests(root),
        'parent_candidate_format': (source_metadata or {}).get('format'),
        'model': model.state_dict(),
        'optimizer': state,
        'topology': {'points': 256, 'mirror_pairs': 32640, 'triangles': 2763520,
                     'tetrahedra': 174792640, 'active_points': 8},
        'true_byte_autoregressive_bpb_available': False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix='.r727_', suffix='.pending')
    os.close(fd)
    try:
        torch.save(obj, temp)
        with open(temp, 'rb') as stream:
            os.fsync(stream.fileno())
        # Atomic create-if-absent on the same filesystem; even another writer
        # racing us cannot cause an existing checkpoint to be overwritten.
        os.link(temp, path)
        os.unlink(temp)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return {'sha256': digest(path), 'bytes': path.stat().st_size,
            'optimizer_count': 1, 'promoted': False}


def load_unified(path, root, threads=2):
    torch.set_num_threads(threads)
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if obj.get('format') != FORMAT or obj.get('promoted') is not False or obj.get('accepted_step') != 14416:
        raise ValueError('invalid or promoted R7.27 checkpoint')
    if obj.get('corpus_metadata_sha256') != corpus_digests(root):
        raise ValueError('R7.27 tokenizer/tree identity mismatch')
    old, old_optimizers, base, compositor = blank(root, threads, embedded_state=obj['model'])
    old = UnifiedR722(old.core, old.calibration, old.near, old.far)
    old_optimizers[4].add_param_group({
        'params': list(old.semantic.parameters()) + list(old.long_semantic.parameters()),
        'lr': 5e-4, 'weight_decay': 0.})
    model = UnifiedR723Diamond(old.core, old.calibration, old.near, old.far,
                               old.semantic, old.long_semantic, GeometricDiamond())
    old_optimizers[4].add_param_group({
        'params': list(model.diamond.parameters()), 'lr': 1e-3, 'weight_decay': 0.})
    if obj.get('architecture_round') in ('FA1_R7_29_CONTEXTUAL_GEOMETRY_V1',
                                        'FA1_R7_30_FACTORIZED_GEOMETRY_V1'):
        # The only new trainable state is an R7-attached scalar read gate.
        # Build the 13-group inherited optimizer first, then append one group
        # without recreating any learned historical Parameter or Adam moment.
        from fa1_r729_contextual_geometry import attach_contextual
        unified = UnifiedR7Optimizer(model, old_optimizers, allow_empty_inherited=True)
        attach_contextual(model, unified)
        if obj['architecture_round'] == 'FA1_R7_30_FACTORIZED_GEOMETRY_V1':
            from fa1_r730_factorized_geometry import attach_factorized
            attach_factorized(model, unified)
    else:
        unified = None
    model.load_state_dict(obj['model'], strict=True)
    if unified is None:
        unified = UnifiedR7Optimizer(model, old_optimizers, allow_empty_inherited=True)
    unified.load_state_dict(obj['optimizer'])
    return model, unified, base, compositor, obj
