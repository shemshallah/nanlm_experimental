#!/usr/bin/env python3
"""Build one complete, checked, non-nested FA1 R7.40 recovery archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def build(repo,inputs,output):
    repo=Path(repo).resolve()
    inputs=Path(inputs).resolve()
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError('release already exists: '+str(output))
    ckpt=inputs/'FA1_R740_CPU24_UNPROMOTED.pt'
    parent=inputs/'FA1_R739_CPU24_UNPROMOTED.pt'
    reports=[inputs/'FA1_R739_CPU24_DEV.json',inputs/'FA1_R740_CPU24_DEV.json']
    report=json.loads(reports[-1].read_text())
    if (report['chosen_dev_strict_gate_passed'] or
            report['source_checkpoint_never_overwritten'] is not True or
            report['unpromoted_candidate_checkpoint']['promoted'] is not False or
            report['one_optimizer_groups']!=24):
        raise AssertionError('R7.40 release must remain an unpromoted 24-group candidate')
    if digest(ckpt)!=report['unpromoted_candidate_checkpoint']['sha256']:
        raise AssertionError('candidate checkpoint digest mismatch')
    if digest(parent)!=report['source_checkpoint_sha256']:
        raise AssertionError('R7.39 parent digest mismatch')

    entries={}
    for path in sorted((repo/'fa1r7.12/source').glob('*.py')):
        entries['source/'+path.name]=path
    for name in ('FA1_R7p39_SIGNED_RAIL_CPU.md',
                 'FA1_R7p40_NORMALIZED_RAIL_CPU.md','R7_CURRENT.md'):
        entries['research/'+name]=repo/'fa1r7.12'/name
    for path in reports:
        entries['reports/'+path.name]=path
    entries['checkpoint/'+ckpt.name]=ckpt
    root=inputs/'r713_root'
    for name in ('corpus/vocab.json','corpus/vocab.words.txt','corpus/tree.i32',
                 'corpus/token_heads.u16','corpus/token_counts.i64',
                 'tokenizer/gutenberg_r7.model','tokenizer/gutenberg_r7.vocab'):
        entries['accepted_tokenizer/'+name]=root/name
    entries['corpus_split/split_manifest.json']=inputs/'nanrouter_split/split_manifest.json'
    for name in ('train','dev','test'):
        entries['corpus_split/'+name+'_documents.json']=inputs/'nanrouter_tokenized'/(name+'_documents.json')
    entries['workflows/fa1-r733-matched-far-history.yml']=(
        repo/'.github/workflows/fa1-r733-matched-far-history.yml')
    for path in entries.values():
        if not path.is_file():
            raise FileNotFoundError(str(path))
    hashes={name:digest(path) for name,path in sorted(entries.items())}
    ledger={
        'round':'FA1_R7.40 normalized signed rail correction',
        'claim':'source and real checkpoint continuity; local regression passed',
        'scope':'93 local FA1 tests; selected development experiment only',
        'result':'strict DEV gate failed; no independent TEST opened; not promoted',
        'limitation':'conditional spelling BPB is not raw-contiguous-text BPB',
        'source_parent_sha256':digest(parent),
        'source_parent_available_separately':True,
        'candidate_sha256':digest(ckpt),
    }
    log=('FA1 R7.39 -> R7.40: actual 24 CPU updates per round, each selected DEV update 1.\n'
         'R7.39 rail modulation RMS 7.55e-7 on 192 DEV positions.\n'
         'R7.40 calibration passed identity/continuity tests but failed independent-book efficacy gate.\n'
         'Accepted step-14,416 lineage unchanged. Full optimizer histories retained.\n'
         'Next: learn content-specific causal history values before increasing rail amplitude.\n')
    manifest={
        'format':'FA1_R740_CUMULATIVE_REPRODUCIBILITY_V1',
        'branch':'master',
        'accepted_step':14416,
        'checkpoint_architecture':'FA1_R7_40_NORMALIZED_SIGNED_RAILS_V1',
        'checkpoint_promoted':False,
        'optimizer_groups':24,
        'candidate_sha256':digest(ckpt),
        'candidate_bytes':ckpt.stat().st_size,
        'direct_parent_sha256':digest(parent),
        'nanrouter_release_sha256':'d6ce3a23757e85fe1b9487ef984a5ba81db9c2dc6e962ac7c5df2960480e0054',
        'dataset_release_url':'https://github.com/shemshallah/NaNROUTER/releases/download/corpora-short-stories-0004/short_stories-0004.tar.zst',
        'train_books':270,'dev_books':15,'sealed_test_books':15,
        'test_used_for_selection':False,
        'selected_update':report['chosen_dev_step'],
        'strict_gate_passed':False,
        'files':{name:{'bytes':path.stat().st_size,'sha256':hashes[name]}
                 for name,path in sorted(entries.items())},
    }
    additional={
        'CLAIM_LEDGER.json':json.dumps(ledger,indent=2,sort_keys=True)+'\n',
        'RESEARCH_LOG.md':log,
        'MANIFEST.json':json.dumps(manifest,indent=2,sort_keys=True)+'\n',
    }
    summed={**hashes,**{name:hashlib.sha256(value.encode()).hexdigest()
                        for name,value in additional.items()}}
    additional['SHA256SUMS.txt']=''.join(f'{value}  {name}\n'
        for name,value in sorted(summed.items()))
    output.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(dir=output.parent,prefix='.r740_',suffix='.pending')
    os.close(fd)
    try:
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,
                             allowZip64=True) as dest:
            for name,path in sorted(entries.items()):
                dest.write(path,name)
            for name,value in sorted(additional.items()):
                dest.writestr(name,value)
        with zipfile.ZipFile(temp) as test:
            if test.testzip() is not None:
                raise AssertionError('ZIP CRC verification failed')
        os.link(temp,output)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return {'path':str(output),'sha256':digest(output),'bytes':output.stat().st_size,
            'members':len(entries)+len(additional),'parent':digest(parent)}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--repo',required=True)
    p.add_argument('--inputs',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    print(json.dumps(build(a.repo,a.inputs,a.output),indent=2))
