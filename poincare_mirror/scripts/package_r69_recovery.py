#!/usr/bin/env python3
"""Create one cumulative R69 candidate ZIP with lineage and checked hashes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for key in ('checkpoint','repo','data-dir','results-dir','previous','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Refuse to overwrite archive')
    current_checkpoint=a.checkpoint.read_bytes()
    r69=json.loads((a.results_dir/'FA2_R69_TRAIN.json').read_text())
    if r69['candidate_sha256']!=sha(current_checkpoint):
        raise ValueError('R69 checkpoint does not match training provenance')
    manifest={
      'archive_round':'FA2_R69_CUMULATIVE_20261117',
      'author':'Justin Howard-Stanley',
      'affiliation':'Independent Researcher; Santa Cruz, California, USA',
      'created_utc':datetime.now(timezone.utc).isoformat(),
      'repository':'shemshallah/nanlm_experimental','branch':'master',
      'source_commit':a.source_commit,
      'checkpoint':'checkpoint/FA2_R69_STEP4465_CANDIDATE.pt',
      'checkpoint_sha256':r69['candidate_sha256'],
      'checkpoint_step':4465,'model_checkpoint_count':1,
      'optimizer_state_count':198,'inherited_optimizer_state_count':197,
      'parent_R68_sha256':r69['parent_sha256'],
      'promotion_status':'unpromoted_candidate',
      'sealed_TEST':True,'nested_archives':False}
    ledger={
      'R68_active_default':{'status':'user_override_protected','source':'source/poincare_mirror/FA2_ACTIVE.json'},
      'R69_6144_distinct_TRAIN_positions':{'status':'verified','source':'reports/FA2_R69_TRAIN.json'},
      'R69_197_inherited_Adam_states':{'status':'verified','source':'reports/FA2_R69_TRAIN.json'},
      'R69_+2_+3_future_memory':{'status':'causal_implemented_and_trained'},
      'R69_DEV':{'status':'unpromoted_until_all_book_disjoint_gates_pass'},
      'independent_TEST':{'status':'sealed'}}
    research=('FA2 R69 research log\n\n'
       'R69 continues R68 at step4209 in the SAME 192D model. The +2 and +3 '
       'causal geometric book histories mature only once historical labels are '
       'observed. Eight zero-init coefficients append to the existing AdamW '
       'group. 256 CPU updates trained them over 6,144 distinct TRAIN positions. '
       '201 inherited model tensors and 197 inherited Adam states remained '
       'bit-for-bit equal. R68 active source and checkpoint stay protected.\n')
    chat=('FA2 exchange 2026-11-17\n\n'
          'User: Continue, train and test.\n'
          'Assistant: Recovered active R68, added +2/+3 causal address memory, '
          'zero-init future rails, full historical Adam restoration, real CPU '
          'TRAIN round and strict book-disjoint DEV evaluator.\n')
    entries={
        'checkpoint/FA2_R69_STEP4465_CANDIDATE.pt':current_checkpoint,
        'MANIFEST.json':json.dumps(manifest,indent=2).encode(),
        'CLAIM_LEDGER.json':json.dumps(ledger,indent=2).encode(),
        'PROVENANCE.json':json.dumps({'parent_archive':'FA2_R68_CUMULATIVE_20261117.zip',
          'corpus':'NaNROUTER Gutenberg38 book-disjoint TRAIN/DEV',
          'independent_TEST':'never opened','github_commit':a.source_commit},indent=2).encode(),
        'RESEARCH_LOG.md':research.encode(),'CHAT_LOG.md':chat.encode()}
    for name in ('meta.json','split_manifest.json','train.bin','valid.bin'):
        entries['data/gutenberg38/'+name]=(a.data_dir/name).read_bytes()
    for file in sorted((a.repo/'poincare_mirror').rglob('*')):
        if file.is_file() and file.suffix in {'.py','.md','.json','.yml','.yaml','.txt','.sh'} and '__pycache__' not in file.parts:
            entries['source/'+str(file.relative_to(a.repo))]=file.read_bytes()
    for file in sorted((a.repo/'.github/workflows').glob('fa2-*.yml')):
        entries['source/'+str(file.relative_to(a.repo))]=file.read_bytes()
    for file in sorted(a.results_dir.glob('FA2_R69_*.json')):
        entries['reports/'+file.name]=file.read_bytes()
    with zipfile.ZipFile(a.previous) as old:
        for name in old.namelist():
            if name.startswith('reports/FA2_R6') and name not in entries:
                entries[name]=old.read(name)
    hashes=[]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.output,'w',compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6,allowZip64=True) as out:
        for name,data in entries.items():
            out.writestr(name,data)
            hashes.append(sha(data)+'  '+name)
        out.writestr('SHA256SUMS','\n'.join(hashes)+'\n')
    print(json.dumps({'archive':str(a.output),'bytes':a.output.stat().st_size,
                      'entries':len(entries)+1,'checkpoint_sha256':r69['candidate_sha256']},indent=2))


if __name__=='__main__':
    main()
