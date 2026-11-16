#!/usr/bin/env python3
"""Cumulative rejected R7.50 candidate recovery, source and SHA manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from fa1_r727_checkpoint import digest


def build(a):
    repo=Path(a.repo).resolve()
    output=Path(a.output).resolve()
    checkpoint=Path(a.checkpoint).resolve()
    reports=Path(a.reports).resolve()
    recovery=Path(a.r747_recovery).resolve()
    if output.exists():
        raise FileExistsError('cannot overwrite cumulative archive')
    meta=json.loads((repo/'fa1r7.12/results/FA1_R750_CPU_CONTINUATION_STATUS.json').read_text())
    audit=json.loads((reports/'FA1_R750_DEV512_MATCHED.json').read_text())
    integrity=json.loads((reports/'FA1_R750_OPTIMIZER_INTEGRITY.json').read_text())
    sha=digest(checkpoint)
    if (sha!=meta['candidate_r750_sha256'] or
            sha!=audit['candidate_sha256'] or sha!=integrity['candidate_sha256'] or
            meta['candidate_promoted'] is not False or
            meta['independent_test_opened'] is not False or
            meta['strict_address_attribution_passed'] is not False or
            integrity['inherited_adamw_states_bit_identical']!=444):
        raise AssertionError('R7.50 rejected-candidate provenance inconsistent')
    files={'checkpoint/FA1_R750_CURRENT_UNPROMOTED.pt':checkpoint}
    for p in sorted((repo/'fa1r7.12/source').glob('*.py')):
        files['source/'+p.name]=p
    for p in sorted((repo/'fa1r7.12/results').glob('FA1_R750_*.json')):
        files['results/'+p.name]=p
    files['research/R7_CURRENT.md']=repo/'fa1r7.12/R7_CURRENT.md'
    files['research/FA1_R7p50_CALIBRATED_INHIBITION.md']=(
        repo/'fa1r7.12/FA1_R7p50_CALIBRATED_INHIBITION.md')
    files['workflows/fa1-r750-calibrated-inhibition.yml']=(
        repo/'.github/workflows/fa1-r750-calibrated-inhibition.yml')
    for p in sorted(reports.glob('FA1_R750_*.json')):
        files['reports/'+p.name]=p
    for p in sorted((recovery/'accepted_tokenizer').rglob('*')):
        if p.is_file():
            files['accepted_tokenizer/'+str(p.relative_to(recovery/'accepted_tokenizer'))]=p
    for p in sorted((recovery/'corpus_split').glob('*.json')):
        files['corpus_split/'+p.name]=p
    hashes={name:digest(path) for name,path in sorted(files.items())}
    commit=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],
                          check=True,capture_output=True,text=True).stdout.strip()
    manifest={'format':'FA1_R750_CUMULATIVE_RECOVERY_V1',
              'git_commit_local':commit,'branch':'master',
              'round':meta['round'],'promoted':False,
              'independent_test_opened':False,
              'accepted_step':14416,'selected_r750_update':1,
              'executed_r750_updates':96,'optimizer_objects':1,
              'optimizer_groups':30,'checkpoint_sha256':sha,
              'checkpoint_bytes':checkpoint.stat().st_size,
              'source_r749_sha256':meta['source_r749_sha256'],
              'train_books':270,'dev_books':15,'test_books_sealed':True,
              'files':{name:{'sha256':hashes[name],'bytes':p.stat().st_size}
                       for name,p in sorted(files.items())}}
    ledger={'verified_facts':{
        'inherited_model_tensors_unchanged':466,
        'inherited_optimizer_groups_unchanged':29,
        'inherited_adamw_states_unchanged':444,
        'next_word_nll_gain':meta['word_gain_nats_per_word'],
        'gain_ci95':meta['word_gain_book_bootstrap_ci95']},
        'failed_requirements':['positive address-specificity CI',
                               'positive learned row-permutation CI',
                               'long-context safety','independent TEST'],
        'promotion':'rejected',
        'disclosure':['DEV repeatedly explored','R7.49 preserved separately',
                      'TRAIN diagnostic is hypothesis-generating']}
    log=('FA1 R7.50: append bounded inhibitory exact C(256,2) value bank to R7.49.\n'
         'All 466 old tensors, 29 optimizer groups, 444 Adam states unchanged.\n'
         '96 TRAIN-only updates; selected step 1, strict DEV gate failed.\n'
         'Independent DEV512 R7.49 5.431276 -> R7.50 5.431263 nats/word.\n'
         'Gain +0.000013; CI includes zero, exact address and row ablations fail.\n'
         'This is a rejected, recoverable research candidate; TEST sealed.\n')
    derived={'MANIFEST.json':json.dumps(manifest,sort_keys=True,indent=2)+'\n',
             'CLAIM_LEDGER.json':json.dumps(ledger,sort_keys=True,indent=2)+'\n',
             'RESEARCH_LOG.md':log}
    sums={**hashes,**{name:hashlib.sha256(body.encode()).hexdigest()
                       for name,body in derived.items()}}
    derived['SHA256SUMS.txt']=''.join(f'{sha256}  {name}\n'
                                        for name,sha256 in sorted(sums.items()))
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,
                         compresslevel=3,allowZip64=True) as z:
        for name,path in sorted(files.items()):
            z.write(path,name)
        for name,body in sorted(derived.items()):
            z.writestr(name,body)
    with zipfile.ZipFile(output,'r') as z:
        if z.testzip() is not None or sum(x.endswith('.pt') for x in z.namelist())!=1:
            raise AssertionError('archive CRC or one-current-checkpoint integrity failed')
        if any(x.endswith('.zip') for x in z.namelist()):
            raise AssertionError('nested backup ZIP forbidden')
    result={'path':str(output),'sha256':digest(output),
            'bytes':output.stat().st_size,'entries':len(files)+len(derived),
            'checkpoint_sha256':sha,'promoted':False}
    print(json.dumps(result,sort_keys=True,indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--repo',required=True)
    p.add_argument('--r747-recovery',required=True)
    p.add_argument('--reports',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    build(p.parse_args())
