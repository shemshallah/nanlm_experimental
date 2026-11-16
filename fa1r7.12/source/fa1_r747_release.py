#!/usr/bin/env python3
"""Single cumulative FA1 R7.47 archive, one current checkpoint, SHA audited."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

from fa1_r727_checkpoint import digest


ROUND='FA1_R7_47_EXACT_HEAD_ADDRESS_COPY_V1'


def build(repo,recovery,output):
    repo=Path(repo).resolve()
    recovery=Path(recovery).resolve()
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError('never overwrite a cumulative recovery archive')
    ckpt=recovery/'FA1_R747_CPU192_UNPROMOTED.pt'
    reports={name:recovery/name for name in (
        'FA1_R744_CPU48_DEV.json',
        'FA1_R744_DEV2048_INDEPENDENT_BOOKS.json',
        'FA1_R745_CPU128_DEV.json',
        'FA1_R745_CPU512_CONTINUED_DEV.json',
        'FA1_R745_DEV2048_MATCHED_COMPARISON.json',
        'FA1_R746_FACTORIAL_DEV2048.json',
        'FA1_R747_CPU192_DEV512.json',
        'FA1_R747_DEV2048_MATCHED.json',
        'FA1_R747_INHERITED_STATE_INTEGRITY.json')}
    for path in (ckpt,*reports.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    train=json.loads(reports['FA1_R747_CPU192_DEV512.json'].read_text())
    large=json.loads(reports['FA1_R747_DEV2048_MATCHED.json'].read_text())
    integrity=json.loads(reports['FA1_R747_INHERITED_STATE_INTEGRITY.json'].read_text())
    if (train['round']!=ROUND or train['candidate_checkpoint']['sha256']!=digest(ckpt)
            or train['one_optimizer_groups']!=27 or train['selected_step']!=192
            or train['candidate_checkpoint_promoted'] is not False
            or train['independent_test_opened'] is not False
            or integrity['candidate_sha256']!=digest(ckpt)
            or integrity['inherited_optimizer_groups_bit_identical']!=26
            or integrity['inherited_adamw_states_bit_identical']!=427
            or large['candidate_sha256']!=digest(ckpt)
            or large['independent_test_opened'] is not False):
        raise AssertionError('unverified or promoted R7.47 candidate')
    entries={}
    for src in sorted((repo/'fa1r7.12/source').glob('*.py')):
        entries['source/'+src.name]=src
    for name in ('FA1_R7p47_EXACT_HEAD_ADDRESS_WORD_COPY.md',
                 'FA1_R7p45_CAUSAL_GEOMETRIC_WORD_MEMORY.md',
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
    entries['corpus_split/split_manifest.json']=recovery/'r740/corpus_split/split_manifest.json'
    for split in ('train','dev','test'):
        entries['corpus_split/'+split+'_documents.json']=(
            recovery/'r740/corpus_split'/(split+'_documents.json'))
    entries['workflows/fa1-r733-matched-far-history.yml']=(
        repo/'.github/workflows/fa1-r733-matched-far-history.yml')
    if any(not path.is_file() for path in entries.values()):
        raise FileNotFoundError('current cumulative source input missing')
    hashes={name:digest(path) for name,path in sorted(entries.items())}
    metric=large['metrics']
    ledger={
        'round':'R7.46 exact-factor control + R7.47 head-address word copy',
        'r746_exact_address_effect':json.loads(
            reports['FA1_R746_FACTORIAL_DEV2048.json'].read_text())['controls']['addresses_only'],
        'r747_dev_word_gain':metric['r747_incremental_word_gain'],
        'r747_word_gain_ci95':metric['r747_word_book_bootstrap']['book_bootstrap_ci95'],
        'r747_history_advantage':metric['r747_correct_vs_foreign_history'],
        'r747_history_advantage_ci95':metric['r747_history_book_bootstrap']['book_bootstrap_ci95'],
        'r747_address_delta':metric['r747_address_specificity'],
        'r747_address_ci95':metric['r747_address_book_bootstrap']['book_bootstrap_ci95'],
        'r747_pair_ci95':metric['r747_pair_book_bootstrap']['book_bootstrap_ci95'],
        'r747_apex_ci95':metric['r747_apex_book_bootstrap']['book_bootstrap_ci95'],
        'selected_update':train['selected_step'],
        'strict_dev_gate_passed':train['selected_strict_dev_gate_passed'],
        'claims':['positive full next-word NLL DEV effect',
                  'positive wrong-book causal history advantage on DEV',
                  '26 inherited optimizer groups and 427 AdamW states exact'],
        'not_established':['exact address specificity or pair path causality',
                           'frontier quality, independent TEST generalization',
                           'full normalized top5 for the entire mixture',
                           'uninterrupted byte entropy/bits-per-byte'],
        'limits':['15 DEV books repeatedly consulted; exploratory CIs',
                  'earlier ancestor corpus overlap unknown',
                  'TEST sealed; candidate unpromoted'],
    }
    manifest={
        'format':'FA1_R747_CUMULATIVE_RECOVERY_V1','branch':'master',
        'architecture_round':ROUND,'accepted_step':14416,
        'checkpoint_promoted':False,'independent_test_opened':False,
        'optimizer_objects':1,'optimizer_groups':27,
        'candidate_sha256':digest(ckpt),'candidate_bytes':ckpt.stat().st_size,
        'source_parent_sha256':train['source_checkpoint_sha256'],
        'dataset_release_url':'https://github.com/shemshallah/NaNROUTER/releases/download/corpora-short-stories-0004/short_stories-0004.tar.zst',
        'dataset_release_sha256':'d6ce3a23757e85fe1b9487ef984a5ba81db9c2dc6e962ac7c5df2960480e0054',
        'train_books':270,'dev_books':15,'sealed_test_books':15,
        'files':{name:{'bytes':path.stat().st_size,'sha256':hashes[name]}
                 for name,path in sorted(entries.items())},
    }
    research_log=(
        'One FA1 R7 model was continued in-place from the R7.45 26-group checkpoint.\n'
        'R7.46 performed exact-address/lexical-tower factorial DEV attribution, no weights.\n'
        'R7.47 added the inherited word-to-two-head mapping and signed pair feature read.\n'
        'A 192-update CPU pilot appended optimizer group 27 and selected update 192.\n'
        'Inherited 448 tensors, 26 groups and 427 AdamW states are bit-identical.\n'
        'Matched 2,048-position DEV: R7.45 NLL 5.404824 -> R7.47 NLL 5.399044.\n'
        'Exact address CI crosses zero; no geometric causality or promotion claim.\n'
        'Accepted step 14416 and sealed TEST remain protected.\n'
        'Remote GitHub publication requires explicit user authorization.\n')
    derived={'MANIFEST.json':json.dumps(manifest,sort_keys=True,indent=2)+'\n',
             'CLAIM_LEDGER.json':json.dumps(ledger,sort_keys=True,indent=2)+'\n',
             'RESEARCH_LOG.md':research_log}
    audit={**hashes,**{name:hashlib.sha256(content.encode()).hexdigest()
                       for name,content in derived.items()}}
    derived['SHA256SUMS.txt']=''.join(f'{h}  {name}\n' for name,h in sorted(audit.items()))
    output.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.r747_',suffix='.pending',dir=output.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,
                             compresslevel=1,allowZip64=True) as dest:
            for name,path in sorted(entries.items()):
                dest.write(path,name)
            for name,content in sorted(derived.items()):
                dest.writestr(name,content)
        with zipfile.ZipFile(temp) as check:
            if check.testzip() is not None:
                raise AssertionError('ZIP CRC test failed')
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
    print(json.dumps(build(a.repo,a.recovery,a.output),sort_keys=True,indent=2))
