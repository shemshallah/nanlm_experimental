#!/usr/bin/env python3
"""One cumulative R7.48 recovery ZIP: current checkpoint, sources, audit.

No nested ZIPs or obsolete intermediate trained checkpoints are included.
Each file has its own SHA-256 entry. Does not open sealed TEST content.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from fa1_r727_checkpoint import digest


def build(a):
    repo = Path(a.repo).resolve()
    archive = Path(a.output).resolve()
    current = Path(a.checkpoint).resolve()
    recovery = Path(a.r747_recovery).resolve()
    reports = Path(a.reports).resolve()
    if archive.exists():
        raise FileExistsError('never overwrite an existing cumulative archive')
    if not current.is_file() or not recovery.is_dir() or not reports.is_dir():
        raise FileNotFoundError('current recovery input missing')
    benchmark = json.loads((reports / 'FA1_R748_DEV512_MATCHED.json').read_text())
    integrity = json.loads((reports / 'FA1_R748_CPU96_INTEGRITY.json').read_text())
    status = json.loads((repo / 'fa1r7.12/results/FA1_R748_CPU_CONTINUATION_STATUS.json').read_text())
    ckpt_sha = digest(current)
    if (ckpt_sha != status['continued_r748_checkpoint_sha256'] or
            benchmark['candidate_sha256'] != ckpt_sha or
            integrity['candidate_sha256'] != ckpt_sha or
            integrity['inherited_optimizer_groups_bit_identical'] != 27 or
            integrity['inherited_adamw_states_bit_identical'] != 432 or
            status['independent_test_opened'] is not False or
            benchmark['independent_test_opened'] is not False):
        raise AssertionError('candidate recovery or provenance integrity mismatch')
    files = {'checkpoint/FA1_R748_CURRENT_UNPROMOTED.pt': current}
    for p in sorted((repo / 'fa1r7.12/source').glob('*.py')):
        files['source/' + p.name] = p
    for p in sorted((repo / 'fa1r7.12/results').glob('FA1_R748_*.json')):
        files['results/' + p.name] = p
    files['research/FA1_R7p48_MIRROR_PAIR_VALUE_MEMORY.md'] = (
        repo / 'fa1r7.12/FA1_R7p48_MIRROR_PAIR_VALUE_MEMORY.md')
    files['research/R7_CURRENT.md'] = repo / 'fa1r7.12/R7_CURRENT.md'
    files['workflows/fa1-r748-mirror-pair-values.yml'] = (
        repo / '.github/workflows/fa1-r748-mirror-pair-values.yml')
    for p in sorted(reports.glob('FA1_R748_*.json')):
        files['reports/' + p.name] = p
    for p in sorted((recovery / 'accepted_tokenizer').rglob('*')):
        if p.is_file():
            files['accepted_tokenizer/' + str(p.relative_to(recovery / 'accepted_tokenizer'))] = p
    for p in sorted((recovery / 'corpus_split').glob('*.json')):
        files['corpus_split/' + p.name] = p
    hashes = {name: digest(path) for name, path in sorted(files.items())}
    commit = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                            check=True, capture_output=True, text=True).stdout.strip()
    metrics = benchmark['matched_counterfactual_metrics']
    manifest = {
        'format': 'FA1_R748_CUMULATIVE_RECOVERY_V1', 'branch': 'master',
        'git_commit': commit, 'architecture_round': status['round'],
        'accepted_step': 14416, 'checkpoint_promoted': False,
        'independent_test_opened': False, 'optimizer_objects': 1,
        'optimizer_groups': 28, 'training_updates_since_r747': 120,
        'checkpoint_sha256': ckpt_sha, 'checkpoint_bytes': current.stat().st_size,
        'r747_ancestor_sha256': status['source_r747_checkpoint_sha256'],
        'immediate_parent_sha256': integrity['immediate_source_sha256'],
        'train_books': 270, 'dev_books': 15, 'test_books_sealed': 15,
        'na_nrouter_source_sha256':
            'd6ce3a23757e85fe1b9487ef984a5ba81db9c2dc6e962ac7c5df2960480e0054',
        'files': {name: {'sha256': hashes[name], 'bytes': path.stat().st_size}
                  for name, path in sorted(files.items())},
    }
    claims = {
        'positive_observations': {
            'pair_bank_necessity_dev_nats_word': metrics['pair_value_bank'],
            'pair_bank_necessity_book_ci95':
                metrics['pair_value_bank_book_bootstrap']['book_bootstrap_ci95'],
            'correct_history_advantage_nats_word': metrics['correct_history'],
        },
        'unpromoted_due_to': ['full word NLL gain CI includes zero',
                              'address-only specificity CI includes zero',
                              'signed orientation is negatively attributable',
                              'no independent TEST or aligned long-context gate'],
        'metric_scope': {'raw_text_bpb': None, 'full_word_top5': None,
                         'full_word_top1': None, 'first_byte_auxiliary_only': True},
        'disclosure': ['repeated DEV exploration',
                       'source model earlier-corpus overlap remains unknown',
                       '96-update pilot failed during post-save reload; saved bytes verified'],
    }
    log = ('FA1 R7.48 recovered directly from full R7.47 27-group trained checkpoint.\n'
           'The single model now has 32,640 trainable C(256,2) values and one 28-group optimizer.\n'
           '24 updates + 96 same-optimizer resumed updates completed on SHA-verified NaNROUTER TRAIN.\n'
           'One current 96-continuation checkpoint is included; no earlier checkpoint duplicates.\n'
           'All 454 inherited R7.47 tensors, 27 groups, 432 AdamW states exact.\n'
           '15-book DEV 512: word NLL 5.433400->5.431809, gain inconclusive.\n'
           'Pair-bank removal degrades NLL +0.002976; signed orientation is detrimental.\n'
           'Full BYTE BPB/mixture top-k unavailable; TEST sealed and promotion rejected.\n')
    derived = {
        'MANIFEST.json': json.dumps(manifest, sort_keys=True, indent=2) + '\n',
        'CLAIM_LEDGER.json': json.dumps(claims, sort_keys=True, indent=2) + '\n',
        'RESEARCH_LOG.md': log,
    }
    checksums = {**hashes, **{name: hashlib.sha256(data.encode()).hexdigest()
                            for name, data in derived.items()}}
    derived['SHA256SUMS.txt'] = ''.join(f'{sha}  {name}\n'
                                          for name, sha in sorted(checksums.items()))
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=3, allowZip64=True) as z:
        for name, path in sorted(files.items()):
            z.write(path, name)
        for name, text in sorted(derived.items()):
            z.writestr(name, text)
    with zipfile.ZipFile(archive, 'r') as z:
        bad = z.testzip()
        if bad:
            raise AssertionError('archive CRC failure: ' + bad)
        if z.namelist().count('checkpoint/FA1_R748_CURRENT_UNPROMOTED.pt') != 1:
            raise AssertionError('archive must contain exactly one current checkpoint')
        if any(name.lower().endswith('.zip') for name in z.namelist()):
            raise AssertionError('nested ZIP forbidden')
    result = {'path': str(archive), 'sha256': digest(archive),
              'bytes': archive.stat().st_size, 'files': len(files),
              'checkpoint_sha256': ckpt_sha, 'test_opened': False}
    print(json.dumps(result, sort_keys=True, indent=2), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True)
    p.add_argument('--r747-recovery', required=True)
    p.add_argument('--reports', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    build(p.parse_args())
