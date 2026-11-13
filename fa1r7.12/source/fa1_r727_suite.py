#!/usr/bin/env python3
"""FA1 R7.27 canonical unified test suite (source + optional real weights).

The source suite runs in GitHub Actions without releasing protected weights.
With --root, --candidate and --corpus-dir it additionally verifies learned
moment continuity and book-disjoint word/first-byte prediction on CPU.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


SOURCE_CASES = (
    'test_fa1_r722.py',
    'test_fa1_r723_topology.py',
    'test_fa1_r724_geometry.py',
    'test_fa1_r725_controls.py',
    'test_fa1_r726_factor_audit.py',
    'test_fa1_r727_optimizer.py',
    'test_fa1_r727_metrics.py',
)


def run(a):
    start = time.monotonic()
    source = Path(__file__).resolve().parent
    env = dict(os.environ)
    env['PYTHONPATH'] = str(source)
    env.setdefault('OMP_NUM_THREADS', '2')
    env.setdefault('OPENBLAS_NUM_THREADS', '1')
    results = []
    for name in SOURCE_CASES:
        process = subprocess.run([sys.executable, str(source / name)], cwd=source,
                                 env=env, capture_output=True, text=True, timeout=240)
        results.append({'test': name, 'passed': process.returncode == 0,
                        'exit_code': process.returncode,
                        'detail': (process.stdout + process.stderr)[-800:]})
        print(('PASS ' if process.returncode == 0 else 'FAIL ') + name, flush=True)
        if process.returncode:
            raise RuntimeError('unified R7 structural suite failed: ' + name + '\n' + results[-1]['detail'])
    report = {'format': 'FA1_R7_27_UNIFIED_TEST_SUITE_V1',
              'source_checks': results, 'source_passed': len(results),
              'one_R7_architecture': True, 'accepted_step': 14416,
              'promoted': False, 'weights_evaluated': False}
    if any((a.root, a.candidate, a.corpus_dir)):
        if not all((a.root, a.candidate, a.corpus_dir)):
            raise ValueError('real-weight evaluation requires root, candidate, and corpus-dir')
        from fa1_r727_real_parity import run as parity
        from fa1_r727_eval import run as evaluate
        from fa1_r727_checkpoint import digest
        accepted_path = Path(a.root) / 'checkpoint/pal_active.pt'
        accepted_before = digest(accepted_path)
        candidate_before = digest(a.candidate)
        probe = argparse.Namespace(root=a.root, candidate=a.candidate,
                                   threads=a.threads, output='')
        report['real_weight_parity'] = parity(probe)
        audit = argparse.Namespace(root=a.root, checkpoint=a.candidate,
                                   corpus_dir=a.corpus_dir, split='dev',
                                   targets=a.targets, batch=a.batch,
                                   threads=a.threads, seed=a.seed,
                                   output='', unified_checkpoint=False)
        report['book_disjoint_dev'] = evaluate(audit)
        if digest(accepted_path) != accepted_before or digest(a.candidate) != candidate_before:
            raise AssertionError('protected checkpoint bytes changed during R7.27 suite')
        report['accepted_checkpoint_sha256_unchanged'] = accepted_before
        report['candidate_checkpoint_sha256_unchanged'] = candidate_before
        report['weights_evaluated'] = True
    report['elapsed_seconds'] = round(time.monotonic() - start, 3)
    if a.output:
        path = Path(a.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print('R7.27 unified suite complete:', json.dumps({
        'source_passed': report['source_passed'],
        'weights_evaluated': report['weights_evaluated'],
        'elapsed_seconds': report['elapsed_seconds']}), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='')
    p.add_argument('--candidate', default='')
    p.add_argument('--corpus-dir', default='')
    p.add_argument('--targets', type=int, default=96)
    p.add_argument('--seed', type=int, default=727)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--output', default='')
    run(p.parse_args())
