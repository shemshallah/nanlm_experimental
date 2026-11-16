#!/usr/bin/env python3
"""One current FA1 R7.45 checkpoint/research/source archive, SHA audited."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile


def digest(path):
    result=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def build(repo,recovery,output):
    repo=Path(repo).resolve()
    recovery=Path(recovery).resolve()
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError('archive already exists')
    ckpt=recovery/'FA1_R745_CPU512_CONTINUED_UNPROMOTED.pt'
    reports={name:recovery/name for name in (
        'FA1_R744_CPU48_DEV.json','FA1_R744_DEV2048_INDEPENDENT_BOOKS.json',
        'FA1_R745_CPU128_DEV.json','FA1_R745_CPU512_CONTINUED_DEV.json',
        'FA1_R745_DEV2048_MATCHED_COMPARISON.json')}
    for path in (ckpt,*reports.values()):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    final=json.loads(reports['FA1_R745_CPU512_CONTINUED_DEV.json'].read_text())
    if (final['candidate_checkpoint']['sha256']!=digest(ckpt) or
            final['one_optimizer_groups']!=26 or
            final['candidate_checkpoint_promoted'] is not False or
            final['independent_test_opened'] is not False):
        raise AssertionError('not a verified unpromoted 26-group candidate')
    entries={}
    for src in sorted((repo/'fa1r7.12/source').glob('*.py')):
        entries['source/'+src.name]=src
    for name in ('FA1_R7p45_CAUSAL_GEOMETRIC_WORD_MEMORY.md',
                 'FA1_R7p40_NORMALIZED_RAIL_CPU.md','R7_CURRENT.md'):
        entries['research/'+name]=repo/'fa1r7.12'/name
    for name,path in reports.items():
        entries['reports/'+name]=path
    entries['checkpoint/'+ckpt.name]=ckpt
    ancestor=recovery/'r740/accepted_tokenizer'
    for name in ('corpus/vocab.json','corpus/vocab.words.txt','corpus/tree.i32',
                 'corpus/token_heads.u16','corpus/token_counts.i64',
                 'tokenizer/gutenberg_r7.model','tokenizer/gutenberg_r7.vocab'):
        entries['accepted_tokenizer/'+name]=ancestor/name
    entries['corpus_split/split_manifest.json']=(
        recovery/'r740/corpus_split/split_manifest.json')
    for split in ('train','dev','test'):
        entries['corpus_split/'+split+'_documents.json']=(
            recovery/'r740/corpus_split'/(split+'_documents.json'))
    entries['workflows/fa1-r733-matched-far-history.yml']=(
        repo/'.github/workflows/fa1-r733-matched-far-history.yml')
    if any(not path.is_file() for path in entries.values()):
        raise FileNotFoundError('missing current cumulative source input')
    hashes={name:digest(path) for name,path in sorted(entries.items())}
    manifest={'format':'FA1_R745_CUMULATIVE_RECOVERY_V1',
              'branch':'master',
              'architecture_round':'FA1_R7_45_GEOMETRIC_VALUE_READ_V1',
              'accepted_step':14416,'checkpoint_promoted':False,
              'independent_test_opened':False,'optimizer_objects':1,
              'optimizer_groups':26,'candidate_sha256':digest(ckpt),
              'candidate_bytes':ckpt.stat().st_size,
              'source_parent_sha256':final['source_checkpoint_sha256'],
              'dataset_release_url':'https://github.com/shemshallah/NaNROUTER/releases/download/corpora-short-stories-0004/short_stories-0004.tar.zst',
              'dataset_release_sha256':'d6ce3a23757e85fe1b9487ef984a5ba81db9c2dc6e962ac7c5df2960480e0054',
              'train_books':270,'dev_books':15,'sealed_test_books':15,
              'files':{name:{'bytes':path.stat().st_size,'sha256':hashes[name]}
                       for name,path in sorted(entries.items())}}
    large=json.loads(reports['FA1_R744_DEV2048_INDEPENDENT_BOOKS.json'].read_text())
    matched=json.loads(reports['FA1_R745_DEV2048_MATCHED_COMPARISON.json'].read_text())
    accepted_large=matched['runs']['continued_512_selected_1']['metrics']
    last=final['development_history'][-1]
    ledger={'round':'R7.44 causal copy + R7.45 geometric word value attention',
            'r744_large_dev_word_gain':large['word_gain_nats_per_word'],
            'r744_large_dev_word_gain_ci95':large['word_gain_book_bootstrap']['book_bootstrap_ci95'],
            'r744_large_dev_correct_history_advantage':large['correct_vs_foreign_word_history'],
            'r745_last_update_incremental_word_gain':last['r745_incremental_word_gain'],
            'r745_last_update_geometric_address_specificity':last['r745_geometric_address_specificity'],
            'r745_large_dev_incremental_word_gain':accepted_large['r745_incremental_word_gain'],
            'r745_large_dev_incremental_word_ci95':accepted_large['r745_incremental_word_bootstrap']['book_bootstrap_ci95'],
            'r745_large_dev_apex_ci95':accepted_large['r745_apex_bootstrap']['book_bootstrap_ci95'],
            'r745_large_dev_address_ci95':accepted_large['r745_geometric_address_bootstrap']['book_bootstrap_ci95'],
            'r745_selected_step':final['selected_step'],
            'r745_selected_dev_gate':final['selected_strict_dev_pass'],
            'claim':'R7.44 has positive book-disjoint DEV word memory contribution; geometric causality remains experimental',
            'provenance_limits':['unknown overlap with earlier R7 ancestor training books',
                                'full mixture word top5 unavailable',
                                'conditional spelling BPB is not raw text BPB',
                                'sealed TEST not consulted',
                                'no frontier/generalization claim']}
    log=('One recovered FA1 R7.40 model and one optimizer were continued.\n'
         'R7.44: appended group 25; 48 CPU updates; independent 2048 DEV positions.\n'
         'R7.45: appended group 26; 128 CPU updates, then resumed same 26-group optimizer '
         'for 512 further updates.\n'
         f"R7.45 final selected local update {final['selected_step']}; strict DEV gate "
         f"{final['selected_strict_dev_pass']}. No promotion.\n"
         'The accepted step-14416 checkpoint and sealed TEST selection remain protected.\n')
    derived={'MANIFEST.json':json.dumps(manifest,indent=2,sort_keys=True)+'\n',
             'CLAIM_LEDGER.json':json.dumps(ledger,indent=2,sort_keys=True)+'\n',
             'RESEARCH_LOG.md':log}
    audit={**hashes,**{name:hashlib.sha256(content.encode()).hexdigest()
                       for name,content in derived.items()}}
    derived['SHA256SUMS.txt']=''.join(f'{h}  {name}\n' for name,h in sorted(audit.items()))
    output.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.r745_',suffix='.pending',dir=output.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,
                             compresslevel=1,allowZip64=True) as dest:
            for name,path in sorted(entries.items()):
                dest.write(path,name)
            for name,contents in sorted(derived.items()):
                dest.writestr(name,contents)
        with zipfile.ZipFile(temp) as check:
            if check.testzip() is not None:
                raise AssertionError('ZIP CRC validation failed')
        os.link(temp,output)
    finally:
        if os.path.exists(temp):os.unlink(temp)
    return {'path':str(output),'bytes':output.stat().st_size,
            'sha256':digest(output),'members':len(entries)+len(derived)}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--repo',required=True)
    p.add_argument('--recovery',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    print(json.dumps(build(a.repo,a.recovery,a.output),indent=2))
