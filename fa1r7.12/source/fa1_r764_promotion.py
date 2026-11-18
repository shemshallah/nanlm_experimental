#!/usr/bin/env python3
"""Promote one fully recoverable R7.64 DEV-validated checkpoint in-place lineage.

Never mutates the candidate. Requires R7.63-inherited 37-group optimizer,
exact candidate SHA, the full four-part book-bootstrap DEV pass and a
separately sampled DEV replication. Independent TEST remains explicitly
unopened. The promoted source is the same trained model, not a fresh fork.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import torch

from fa1_r727_checkpoint import FORMAT, digest
from fa1_r727_optimizer import FORMAT as OPTIMIZER_FORMAT
from fa1_r764_multiresolution import ROUND

GATES = (
    ('hierarchy_gain', 'hierarchy_bootstrap'),
    ('geometry_over_lexical_matched', 'book_bootstrap'),
    ('correct_vs_wrong_older', 'book_bootstrap'),
    ('correct_vs_wrong_geometry_values', 'book_bootstrap'))


def check_dev_gates(metrics):
    if not metrics.get('strict_geometry_gate_passed'):
        raise ValueError('R7.64 strict DEV gate was not passed')
    for field, group in GATES:
        value = float(metrics[field])
        ci = (metrics[group]['book_bootstrap_ci95'] if
              group == 'hierarchy_bootstrap' else
              metrics[group][field]['book_bootstrap_ci95'])
        if not (value > 0 and ci[0] > 0 and ci[1] >= ci[0]):
            raise ValueError('non-positive or malformed DEV gate: ' + field)


def promote(candidate, dev_path, replication_path, output):
    candidate, output = Path(candidate).resolve(), Path(output).resolve()
    if output.exists() or output == candidate:
        raise FileExistsError('promotion requires a new immutable path')
    dev = json.loads(Path(dev_path).read_text())
    replication = json.loads(Path(replication_path).read_text())
    expected = digest(candidate)
    if (dev.get('round') != ROUND or dev.get('candidate_sha256') != expected
            or dev.get('one_optimizer_groups', dev.get('optimizer_groups')) != 37
            or dev.get('inherited_groups_preserved') != 36
            or dev.get('full_checkpoint_reload_exact') is not True):
        raise ValueError('R7.64 checkpoint provenance mismatch')
    is_continuation = dev.get('prior_updates', 0) > 0
    if is_continuation:
        if (dev.get('prior_updates') < 1000
                or dev.get('selected_step', 0) <= dev['prior_updates']
                or dev['selected_step'] > dev['prior_updates'] + dev.get('steps_executed', 0)):
            raise ValueError('R7.64 continuation steps are inconsistent')
    elif dev.get('steps_executed') != 1000 or dev.get('selected_step') != 1000:
        raise ValueError('R7.64 first-1k checkpoint provenance mismatch')
    check_dev_gates(dev['matched_dev'])
    if (replication.get('model_round') != ROUND
            or replication.get('checkpoint_sha256') != expected
            or replication.get('optimizer_groups') != 37
            or replication.get('independent_test_accessed') is not False
            or replication.get('dev_positions', 0) < 4096
            or replication.get('strictly_passed') is not True):
        raise ValueError('fresh-position DEV replication did not pass')
    check_dev_gates(replication['metrics'])
    checkpoint = torch.load(candidate, map_location='cpu', weights_only=False)
    state = checkpoint.get('optimizer')
    if (checkpoint.get('format') != FORMAT or
            checkpoint.get('architecture_round') != ROUND or
            checkpoint.get('promoted') is not False or
            checkpoint.get('candidate_trained') is not True or
            checkpoint.get('candidate_training_updates') != dev['selected_step'] or
            state.get('format') != OPTIMIZER_FORMAT or
            len(state['groups']) != 37):
        raise ValueError('candidate architecture or optimizer history mismatch')
    checkpoint.update(
        promoted=True,
        development_gate_passed=True,
        promotion_source_checkpoint_sha256=expected,
        promotion_dev_report_sha256=digest(dev_path),
        promotion_replication_report_sha256=digest(replication_path),
        independent_test_opened=False,
        promotion_scope='DEV_verified_default_TEST_unopened',
        promotion_authorization='explicit_user_instruction')
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=output.parent, prefix='.r764_promote_', suffix='.pending')
    os.close(fd)
    try:
        torch.save(checkpoint, tmp)
        with open(tmp, 'rb') as stream:
            os.fsync(stream.fileno())
        os.link(tmp, output)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return {'round': ROUND, 'checkpoint': str(output),
            'sha256': digest(output), 'groups': len(state['groups']),
            'promoted': True, 'independent_test_opened': False,
            'original_sha256': expected}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', required=True)
    p.add_argument('--dev-report', required=True)
    p.add_argument('--replication-report', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    print(json.dumps(promote(args.candidate, args.dev_report,
                             args.replication_report, args.output),
                     indent=2, sort_keys=True))
