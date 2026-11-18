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
        # Raw uninterrupted text BPB remains unavailable. R7.31 adds a
        # distinct conditional UTF-8 spelling likelihood (with EOW).
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
    serialized_architecture = obj.get('architecture_round')
    # R7.57 retains the complete R7.53 topology and appends one trained
    # transition-router group. Reconstruct the inherited base first.
    recorded_architecture = (
        'FA1_R7_53_CAUSAL_SUCCESSOR_MIXTURE_V1'
        if serialized_architecture in ('FA1_R7_57_TRAIN_ONLY_SIGNED_PAIR_ROUTER_V1',
                                        'FA1_R7_58_CAUSAL_COMPOSITIONAL_MIRROR_V1',
                                        'FA1_R7_59_WORD_CONDITIONED_GEOMETRIC_PRIOR_V1',
                                        'FA1_R7_60_SUPPORT_CALIBRATED_GEOMETRY_V1',
                                        'FA1_R7_61_TWO_RAIL_HIERARCHICAL_GEOMETRY_V1',
                                        'FA1_R7_62_SELECTIVE_GEOMETRIC_INTERSECTION_V1',
                                        'FA1_R7_63_TEMPORAL_CROSS_RAIL_GEOMETRY_V1',
                                        'FA1_R7_64_MULTISCALE_TEMPORAL_DIAMOND_V1')
        else serialized_architecture)
    # R7.44-50 are strictly additive extensions of R7.40. First rebuild the
    # original 24-group topology, then append the one backoff group. Do not
    # infer or reinitialize any of the inherited moments from the round tag.
    if recorded_architecture in ('FA1_R7_44_CAUSAL_WORD_BACKOFF_V1',
                                 'FA1_R7_45_GEOMETRIC_VALUE_READ_V1',
                                 'FA1_R7_47_EXACT_HEAD_ADDRESS_COPY_V1',
                                 'FA1_R7_48_TRAINABLE_MIRROR_PAIR_VALUES_V1',
                                 'FA1_R7_49_CAUSAL_PAIR_ADDRESS_RELIABILITY_V1',
                                 'FA1_R7_50_CALIBRATED_PAIR_INHIBITION_V1',
                                         'FA1_R7_51_BIDIRECTIONAL_PAIR_EVIDENCE_V1',
                                         'FA1_R7_53_CAUSAL_SUCCESSOR_MIXTURE_V1'):
        obj = dict(obj, architecture_round='FA1_R7_40_NORMALIZED_SIGNED_RAILS_V1')
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
                                        'FA1_R7_30_FACTORIZED_GEOMETRY_V1',
                                        'FA1_R7_31_LAYERED_DIAMOND_V1',
                                        'FA1_R7_32_CALIBRATED_DIAMOND_V1',
                                        'FA1_R7_UNIFIED_APEX_FEEDBACK_V1',
                                        'FA1_R7_35_CAUSAL_HISTORY_VALUES_V1',
                                        'FA1_R7_36_DEEP_WIRING_V1',
                                        'FA1_R7_37_CLOSED_FEEDBACK_V1',
                                        'FA1_R7_38_CAUSAL_RELIABILITY_V1',
                                        'FA1_R7_39_SIGNED_CROSS_RAIL_V1',
                                        'FA1_R7_40_NORMALIZED_SIGNED_RAILS_V1'):
        # The only new trainable state is an R7-attached scalar read gate.
        # Build the 13-group inherited optimizer first, then append one group
        # without recreating any learned historical Parameter or Adam moment.
        from fa1_r729_contextual_geometry import attach_contextual
        unified = UnifiedR7Optimizer(model, old_optimizers, allow_empty_inherited=True)
        attach_contextual(model, unified)
        if obj['architecture_round'] in ('FA1_R7_30_FACTORIZED_GEOMETRY_V1',
                                         'FA1_R7_31_LAYERED_DIAMOND_V1',
                                         'FA1_R7_32_CALIBRATED_DIAMOND_V1',
                                         'FA1_R7_UNIFIED_APEX_FEEDBACK_V1',
                                         'FA1_R7_35_CAUSAL_HISTORY_VALUES_V1',
                                         'FA1_R7_36_DEEP_WIRING_V1',
                                         'FA1_R7_37_CLOSED_FEEDBACK_V1',
                                         'FA1_R7_38_CAUSAL_RELIABILITY_V1',
                                         'FA1_R7_39_SIGNED_CROSS_RAIL_V1',
                                         'FA1_R7_40_NORMALIZED_SIGNED_RAILS_V1'):
            from fa1_r730_factorized_geometry import attach_factorized
            attach_factorized(model, unified)
        if obj['architecture_round'] in ('FA1_R7_31_LAYERED_DIAMOND_V1',
                                         'FA1_R7_32_CALIBRATED_DIAMOND_V1',
                                         'FA1_R7_UNIFIED_APEX_FEEDBACK_V1',
                                         'FA1_R7_35_CAUSAL_HISTORY_VALUES_V1',
                                         'FA1_R7_36_DEEP_WIRING_V1',
                                         'FA1_R7_37_CLOSED_FEEDBACK_V1',
                                         'FA1_R7_38_CAUSAL_RELIABILITY_V1',
                                         'FA1_R7_39_SIGNED_CROSS_RAIL_V1',
                                         'FA1_R7_40_NORMALIZED_SIGNED_RAILS_V1'):
            from fa1_r731_layered_decoder import attach_layered
            attach_layered(model, unified)
        if obj['architecture_round'] in ('FA1_R7_32_CALIBRATED_DIAMOND_V1',
                                         'FA1_R7_UNIFIED_APEX_FEEDBACK_V1',
                                         'FA1_R7_35_CAUSAL_HISTORY_VALUES_V1',
                                         'FA1_R7_36_DEEP_WIRING_V1',
                                         'FA1_R7_37_CLOSED_FEEDBACK_V1',
                                         'FA1_R7_38_CAUSAL_RELIABILITY_V1',
                                         'FA1_R7_39_SIGNED_CROSS_RAIL_V1',
                                         'FA1_R7_40_NORMALIZED_SIGNED_RAILS_V1'):
            from fa1_r732_geometry_calibration import (attach_calibrated,
                                                        attach_apex_feedback)
            attach_calibrated(model, unified)
            if obj['architecture_round'] in ('FA1_R7_UNIFIED_APEX_FEEDBACK_V1',
                                             'FA1_R7_35_CAUSAL_HISTORY_VALUES_V1',
                                             'FA1_R7_36_DEEP_WIRING_V1',
                                             'FA1_R7_37_CLOSED_FEEDBACK_V1',
                                             'FA1_R7_38_CAUSAL_RELIABILITY_V1',
                                             'FA1_R7_39_SIGNED_CROSS_RAIL_V1',
                                             'FA1_R7_40_NORMALIZED_SIGNED_RAILS_V1'):
                attach_apex_feedback(model, unified)
        if obj['architecture_round'] in ('FA1_R7_35_CAUSAL_HISTORY_VALUES_V1',
                                         'FA1_R7_36_DEEP_WIRING_V1',
                                         'FA1_R7_37_CLOSED_FEEDBACK_V1',
                                         'FA1_R7_38_CAUSAL_RELIABILITY_V1',
                                         'FA1_R7_39_SIGNED_CROSS_RAIL_V1',
                                         'FA1_R7_40_NORMALIZED_SIGNED_RAILS_V1'):
            from fa1_r735_history_values import attach_history_values
            attach_history_values(model, unified)
        if obj['architecture_round'] in ('FA1_R7_36_DEEP_WIRING_V1',
                                         'FA1_R7_37_CLOSED_FEEDBACK_V1',
                                         'FA1_R7_38_CAUSAL_RELIABILITY_V1',
                                         'FA1_R7_39_SIGNED_CROSS_RAIL_V1',
                                         'FA1_R7_40_NORMALIZED_SIGNED_RAILS_V1'):
            from fa1_r736_deep_wiring import attach_deep_wiring
            attach_deep_wiring(model, unified)
        if obj['architecture_round'] in ('FA1_R7_37_CLOSED_FEEDBACK_V1',
                                         'FA1_R7_38_CAUSAL_RELIABILITY_V1',
                                         'FA1_R7_39_SIGNED_CROSS_RAIL_V1',
                                         'FA1_R7_40_NORMALIZED_SIGNED_RAILS_V1'):
            from fa1_r737_feedback_closure import attach_feedback_closure
            attach_feedback_closure(model, unified)
        if obj['architecture_round'] in ('FA1_R7_38_CAUSAL_RELIABILITY_V1',
                                         'FA1_R7_39_SIGNED_CROSS_RAIL_V1',
                                         'FA1_R7_40_NORMALIZED_SIGNED_RAILS_V1'):
            from fa1_r738_reliability import attach_reliability
            attach_reliability(model, unified)
        if obj['architecture_round'] in ('FA1_R7_39_SIGNED_CROSS_RAIL_V1',
                                         'FA1_R7_40_NORMALIZED_SIGNED_RAILS_V1'):
            from fa1_r739_signed_rails import attach_signed_cross_rail
            attach_signed_cross_rail(model, unified)
        if obj['architecture_round'] == 'FA1_R7_40_NORMALIZED_SIGNED_RAILS_V1':
            from fa1_r740_rail_calibration import attach_rail_calibration
            attach_rail_calibration(model, unified)
            if recorded_architecture in ('FA1_R7_44_CAUSAL_WORD_BACKOFF_V1',
                                         'FA1_R7_45_GEOMETRIC_VALUE_READ_V1',
                                         'FA1_R7_47_EXACT_HEAD_ADDRESS_COPY_V1',
                                         'FA1_R7_48_TRAINABLE_MIRROR_PAIR_VALUES_V1',
                                         'FA1_R7_49_CAUSAL_PAIR_ADDRESS_RELIABILITY_V1',
                                         'FA1_R7_50_CALIBRATED_PAIR_INHIBITION_V1',
                                         'FA1_R7_51_BIDIRECTIONAL_PAIR_EVIDENCE_V1',
                                         'FA1_R7_53_CAUSAL_SUCCESSOR_MIXTURE_V1'):
                from fa1_r744_causal_word_backoff import attach_causal_word_backoff
                attach_causal_word_backoff(model, unified)
            if recorded_architecture in ('FA1_R7_45_GEOMETRIC_VALUE_READ_V1',
                                         'FA1_R7_47_EXACT_HEAD_ADDRESS_COPY_V1',
                                         'FA1_R7_48_TRAINABLE_MIRROR_PAIR_VALUES_V1',
                                         'FA1_R7_49_CAUSAL_PAIR_ADDRESS_RELIABILITY_V1',
                                         'FA1_R7_50_CALIBRATED_PAIR_INHIBITION_V1',
                                         'FA1_R7_51_BIDIRECTIONAL_PAIR_EVIDENCE_V1',
                                         'FA1_R7_53_CAUSAL_SUCCESSOR_MIXTURE_V1'):
                from fa1_r745_geometric_value_read import attach_geometric_value_read
                attach_geometric_value_read(model, unified)
            if recorded_architecture in ('FA1_R7_47_EXACT_HEAD_ADDRESS_COPY_V1',
                                         'FA1_R7_48_TRAINABLE_MIRROR_PAIR_VALUES_V1',
                                         'FA1_R7_49_CAUSAL_PAIR_ADDRESS_RELIABILITY_V1',
                                         'FA1_R7_50_CALIBRATED_PAIR_INHIBITION_V1',
                                         'FA1_R7_51_BIDIRECTIONAL_PAIR_EVIDENCE_V1',
                                         'FA1_R7_53_CAUSAL_SUCCESSOR_MIXTURE_V1'):
                from fa1_r747_head_address_copy import attach_exact_head_copy
                attach_exact_head_copy(model, unified, base)
            if recorded_architecture in ('FA1_R7_48_TRAINABLE_MIRROR_PAIR_VALUES_V1',
                                         'FA1_R7_49_CAUSAL_PAIR_ADDRESS_RELIABILITY_V1',
                                         'FA1_R7_50_CALIBRATED_PAIR_INHIBITION_V1',
                                         'FA1_R7_51_BIDIRECTIONAL_PAIR_EVIDENCE_V1',
                                         'FA1_R7_53_CAUSAL_SUCCESSOR_MIXTURE_V1'):
                from fa1_r748_mirror_pair_values import attach_mirror_pair_values
                attach_mirror_pair_values(model, unified)
            if recorded_architecture in ('FA1_R7_49_CAUSAL_PAIR_ADDRESS_RELIABILITY_V1',
                                         'FA1_R7_50_CALIBRATED_PAIR_INHIBITION_V1',
                                         'FA1_R7_51_BIDIRECTIONAL_PAIR_EVIDENCE_V1',
                                         'FA1_R7_53_CAUSAL_SUCCESSOR_MIXTURE_V1'):
                from fa1_r749_address_reliability import attach_address_reliability
                attach_address_reliability(model, unified)
            if recorded_architecture == 'FA1_R7_50_CALIBRATED_PAIR_INHIBITION_V1':
                from fa1_r750_calibrated_inhibition import attach_calibrated_inhibition
                attach_calibrated_inhibition(model, unified)
            if recorded_architecture == 'FA1_R7_51_BIDIRECTIONAL_PAIR_EVIDENCE_V1':
                from fa1_r751_bidirectional_evidence import attach_bidirectional_pair_evidence
                attach_bidirectional_pair_evidence(model, unified)
            if recorded_architecture == 'FA1_R7_53_CAUSAL_SUCCESSOR_MIXTURE_V1':
                from fa1_r753_causal_successor import attach_causal_successors
                attach_causal_successors(model, unified)
                if serialized_architecture in ('FA1_R7_57_TRAIN_ONLY_SIGNED_PAIR_ROUTER_V1',
                                               'FA1_R7_58_CAUSAL_COMPOSITIONAL_MIRROR_V1',
                                               'FA1_R7_59_WORD_CONDITIONED_GEOMETRIC_PRIOR_V1',
                                               'FA1_R7_60_SUPPORT_CALIBRATED_GEOMETRY_V1',
                                               'FA1_R7_61_TWO_RAIL_HIERARCHICAL_GEOMETRY_V1',
                                               'FA1_R7_62_SELECTIVE_GEOMETRIC_INTERSECTION_V1',
                                               'FA1_R7_63_TEMPORAL_CROSS_RAIL_GEOMETRY_V1',
                                               'FA1_R7_64_MULTISCALE_TEMPORAL_DIAMOND_V1'):
                    from fa1_r757_geometric_transition_router import (
                        BANK_FIELDS, attach_transition_router)
                    banks = {field: obj['model'][
                        'word_backoff.r757_router.' + field]
                        for field in BANK_FIELDS}
                    attach_transition_router(model, unified, banks)
                    if serialized_architecture in (
                            'FA1_R7_58_CAUSAL_COMPOSITIONAL_MIRROR_V1',
                            'FA1_R7_59_WORD_CONDITIONED_GEOMETRIC_PRIOR_V1',
                            'FA1_R7_60_SUPPORT_CALIBRATED_GEOMETRY_V1',
                            'FA1_R7_61_TWO_RAIL_HIERARCHICAL_GEOMETRY_V1',
                            'FA1_R7_62_SELECTIVE_GEOMETRIC_INTERSECTION_V1',
                            'FA1_R7_63_TEMPORAL_CROSS_RAIL_GEOMETRY_V1',
                            'FA1_R7_64_MULTISCALE_TEMPORAL_DIAMOND_V1'):
                        from fa1_r758_contextual_geometry import (
                            BANK_FIELDS as CONTEXT_FIELDS, attach_contextual_geometry)
                        context_banks = {field: obj['model'][
                            'word_backoff.r758_context.' + field]
                            for field in CONTEXT_FIELDS}
                        attach_contextual_geometry(model, unified, context_banks)
                        if serialized_architecture in (
                                'FA1_R7_59_WORD_CONDITIONED_GEOMETRIC_PRIOR_V1',
                                'FA1_R7_60_SUPPORT_CALIBRATED_GEOMETRY_V1',
                                'FA1_R7_61_TWO_RAIL_HIERARCHICAL_GEOMETRY_V1',
                                'FA1_R7_62_SELECTIVE_GEOMETRIC_INTERSECTION_V1',
                                'FA1_R7_63_TEMPORAL_CROSS_RAIL_GEOMETRY_V1',
                                'FA1_R7_64_MULTISCALE_TEMPORAL_DIAMOND_V1'):
                            from fa1_r759_geometric_prior import (
                                BANK_FIELDS as PRIOR_FIELDS, attach_geometric_prior)
                            prior_banks = {field: obj['model'][
                                'word_backoff.r759_prior.' + field]
                                for field in PRIOR_FIELDS}
                            attach_geometric_prior(model, unified, prior_banks)
                            if serialized_architecture in (
                                    'FA1_R7_60_SUPPORT_CALIBRATED_GEOMETRY_V1',
                                    'FA1_R7_61_TWO_RAIL_HIERARCHICAL_GEOMETRY_V1',
                                    'FA1_R7_62_SELECTIVE_GEOMETRIC_INTERSECTION_V1',
                                    'FA1_R7_63_TEMPORAL_CROSS_RAIL_GEOMETRY_V1',
                                    'FA1_R7_64_MULTISCALE_TEMPORAL_DIAMOND_V1'):
                                from fa1_r760_geometric_calibration import (
                                    attach_geometric_calibration)
                                attach_geometric_calibration(model, unified)
                                if serialized_architecture == 'FA1_R7_61_TWO_RAIL_HIERARCHICAL_GEOMETRY_V1':
                                    from fa1_r761_two_rail_geometry import (
                                        BANK_FIELDS as RAIL_FIELDS, attach_two_rail_geometry)
                                    rail_banks = {field: obj['model'][
                                        'word_backoff.r759_prior.' + field]
                                        for field in RAIL_FIELDS}
                                    attach_two_rail_geometry(model, unified, rail_banks)
                                if serialized_architecture in (
                                        'FA1_R7_62_SELECTIVE_GEOMETRIC_INTERSECTION_V1',
                                        'FA1_R7_63_TEMPORAL_CROSS_RAIL_GEOMETRY_V1',
                                        'FA1_R7_64_MULTISCALE_TEMPORAL_DIAMOND_V1'):
                                    from fa1_r762_selective_intersection import (
                                        BANK_FIELDS as RAIL_FIELDS,attach_selective_intersection)
                                    rail_banks = {field: obj['model'][
                                        'word_backoff.r759_prior.' + field]
                                        for field in RAIL_FIELDS}
                                    attach_selective_intersection(model,unified,rail_banks)
                                    if serialized_architecture in (
                                            'FA1_R7_63_TEMPORAL_CROSS_RAIL_GEOMETRY_V1',
                                            'FA1_R7_64_MULTISCALE_TEMPORAL_DIAMOND_V1'):
                                        from fa1_r763_temporal_cross import (
                                            BANK_FIELDS as TEMPORAL_FIELDS,
                                            attach_temporal_cross)
                                        temporal_banks={field:obj['model'][
                                            'word_backoff.r759_prior.'+field]
                                            for field in TEMPORAL_FIELDS}
                                        attach_temporal_cross(model,unified,temporal_banks)
                                        if serialized_architecture == 'FA1_R7_64_MULTISCALE_TEMPORAL_DIAMOND_V1':
                                            from fa1_r764_multiresolution import (
                                                BANK_FIELDS as MULTI_FIELDS,
                                                attach_multiresolution)
                                            multi_banks={field:obj['model'][
                                                'word_backoff.r759_prior.'+field]
                                                for field in MULTI_FIELDS}
                                            attach_multiresolution(model,unified,multi_banks)
    else:
        unified = None
    model.load_state_dict(obj['model'], strict=True)
    if unified is None:
        unified = UnifiedR7Optimizer(model, old_optimizers, allow_empty_inherited=True)
    unified.load_state_dict(obj['optimizer'])
    if serialized_architecture != obj.get('architecture_round'):
        obj['architecture_round'] = serialized_architecture
    return model, unified, base, compositor, obj
