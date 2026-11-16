#!/usr/bin/env python3
"""Single cumulative FA1 R7.49 recovery ZIP with checksum provenance."""
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
    checkpoint = Path(a.checkpoint).resolve()
    recovery = Path(a.r747_recovery).resolve()
    reports = Path(a.reports).resolve()
    if archive.exists():
        raise FileExistsError('cumulative archives are immutable')
    bench = json.loads((reports/'FA1_R749_DEV512_MATCHED_CONTINUED.json').read_text())
    integrity = json.loads((reports/'FA1_R749_OPTIMIZER_INTEGRITY.json').read_text())
    status = json.loads((repo/'fa1r7.12/results/FA1_R749_CPU_CONTINUATION_STATUS.json').read_text())
    sha = digest(checkpoint)
    if (sha != bench['candidate_sha256'] or sha != integrity['candidate_sha256'] or
            sha != status['candidate_r749_checkpoint_sha256'] or
            status['independent_test_opened'] is not False or
            integrity['bitwise_original_adamw_states'] != 432 or
            status['strict_geometric_attribution_gate_passed'] is not False):
        raise AssertionError('unpromoted R7.49 evidence or checksum mismatch')
    files = {'checkpoint/FA1_R749_CURRENT_UNPROMOTED.pt': checkpoint}
    for p in sorted((repo/'fa1r7.12/source').glob('*.py')):
        files['source/'+p.name] = p
    for p in sorted((repo/'fa1r7.12/results').glob('FA1_R749_*.json')):
        files['results/'+p.name] = p
    files['research/FA1_R7p49_CAUSAL_ADDRESS_RELIABILITY.md'] = (
        repo/'fa1r7.12/FA1_R7p49_CAUSAL_ADDRESS_RELIABILITY.md')
    files['research/R7_CURRENT.md'] = repo/'fa1r7.12/R7_CURRENT.md'
    files['workflows/fa1-r749-address-reliability.yml'] = (
        repo/'.github/workflows/fa1-r749-address-reliability.yml')
    for p in sorted(reports.glob('FA1_R749_*.json')):
        files['reports/'+p.name] = p
    for p in sorted((recovery/'accepted_tokenizer').rglob('*')):
        if p.is_file():
            files['accepted_tokenizer/'+str(p.relative_to(recovery/'accepted_tokenizer'))] = p
    for p in sorted((recovery/'corpus_split').glob('*.json')):
        files['corpus_split/'+p.name] = p
    hashes = {name: digest(p) for name,p in sorted(files.items())}
    git_head = subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],
                              capture_output=True,text=True,check=True).stdout.strip()
    manifest = {'format':'FA1_R749_CUMULATIVE_RECOVERY_V1',
                'architecture_round':status['round'],'branch':'master',
                'git_commit':git_head,'author':'Justin Howard-Stanley',
                'accepted_step':14416,'candidate_promoted':False,
                'independent_test_opened':False,'optimizer_objects':1,
                'optimizer_groups':29,
                'selected_updates_since_r748':160,
                'checkpoint_sha256':sha,'checkpoint_bytes':checkpoint.stat().st_size,
                'parent_r748_sha256':status['source_r748_checkpoint_sha256'],
                'corpus_sha256':status['source_corpus_sha256'],
                'train_books':270,'dev_books':15,'test_books_sealed':15,
                'files':{name:{'sha256':hashes[name],'bytes':p.stat().st_size}
                         for name,p in sorted(files.items())}}
    claims = {'confirmed': {
        'inherited_backbone_tensors_unchanged':454,
        'inherited_optimizer_groups_frozen':27,
        'inherited_adamw_states_unchanged':432,
        'r748_to_r749_dev_gain_nats_per_word':bench['real_r748_to_r749_gain'],
        'dev_gain_whole_book_ci95':bench['real_gain_book_bootstrap']['book_bootstrap_ci95']},
        'nonclaims':['Exact-pair-address causality not demonstrated',
                     'Remaining signed rail still slightly harmful',
                     'No independent TEST or long-context confirmation',
                     'No full raw-byte BPB or full-word top-k available'],
        'dev_hypothesis':'Negative exact-match address coefficients may signal inhibition',
        'disclosures':['DEV already used in prior FA1 iterations',
                       'Two exploratory coefficient sweeps on DEV',
                       'No model promotion or change to accepted step-14416 checkpoint']}
    log = ('FA1 R7.49: same full R7.48 model and one optimizer, 29 groups.\n'
           '64 + 128 CPU steps executed; selected 64 + 96 = 160 updates.\n'
           'Actual checkpoint checksum: '+sha+'\n'
           'Inherited: 454 backbone tensors, 27 frozen groups, 432 AdamW states bit-identical.\n'
           'Independent reload on 512 DEV positions: 5.431809 -> 5.431276 nats/word.\n'
           'Word gain +0.000533; 95% whole-book DEV interval [+0.000185,+0.000904].\n'
           'Most gain comes from suppressing the harmful signed contribution.\n'
           'True exact-address attribution remains statistically unresolved.\n'
           'Inhibitory exact-match is a research hypothesis only. No TEST or promotion.\n')
    derived = {'MANIFEST.json':json.dumps(manifest,indent=2,sort_keys=True)+'\n',
               'CLAIM_LEDGER.json':json.dumps(claims,indent=2,sort_keys=True)+'\n',
               'RESEARCH_LOG.md':log}
    sums = {**hashes, **{n:hashlib.sha256(s.encode()).hexdigest() for n,s in derived.items()}}
    derived['SHA256SUMS.txt'] = ''.join(f'{h}  {n}\n' for n,h in sorted(sums.items()))
    archive.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,
                         compresslevel=3,allowZip64=True) as z:
        for name,p in sorted(files.items()):
            z.write(p,name)
        for name,s in sorted(derived.items()):
            z.writestr(name,s)
    with zipfile.ZipFile(archive,'r') as z:
        if z.testzip():
            raise AssertionError('ZIP CRC failed')
        if sum(name.endswith('.pt') for name in z.namelist()) != 1:
            raise AssertionError('exactly one current trained checkpoint required')
        if any(name.endswith('.zip') for name in z.namelist()):
            raise AssertionError('nested backup ZIP forbidden')
    result = {'path':str(archive),'sha256':digest(archive),
              'bytes':archive.stat().st_size,'entry_count':len(files)+len(derived),
              'checkpoint_sha256':sha,'sealed_test':True}
    print(json.dumps(result,indent=2,sort_keys=True),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--repo',required=True)
    p.add_argument('--r747-recovery',required=True)
    p.add_argument('--reports',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    build(p.parse_args())
