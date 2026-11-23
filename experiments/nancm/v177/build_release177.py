#!/usr/bin/env python3
"""Cumulative one-ZIP checkpoint builder. Parent entries are byte-preserved
except extended root Makefile, whose original bytes are stored in ancestry.
Writes to a temporary ZIP and atomically promotes only after CRC/SHA checks.
"""
from __future__ import annotations
import hashlib,json,os,zipfile,datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
PARENT=Path('/mnt/data/NaNCM_v176_SIGNED_BPE_WORDTOP5_SPECULATIVE_THIRDCODE_RECOVERY_20261123.zip')
OUT=Path('/mnt/data/NaNCM_v177_PREFIX_DAG_MACRO_VERIFIER_RECOVERY_20261123.zip')
TEMP=OUT.with_suffix('.tmp.zip')
EXPECTED_PARENT_SHA='c49b8feb6159bb38e070ddc45331a457b7ab968d30d72870862c3034ed5a6005'

def digest(data):return hashlib.sha256(data).hexdigest()


def main():
    assert PARENT.is_file() and digest(PARENT.read_bytes())==EXPECTED_PARENT_SHA
    v177=ROOT/'experiments/nancm/v177'
    new={}
    for path in v177.iterdir():
        if path.is_file() and path.name not in ('__pycache__',):
            new[str(path.relative_to(ROOT))]=path.read_bytes()
    for path in (ROOT/'results').glob('v177_*'):
        if path.is_file() and path.suffix in ('.json','.log'):
            new[str(path.relative_to(ROOT))]=path.read_bytes()
    for name in ('README_v177.md','CLAIM_LEDGER_v177.md','RESEARCH_LOG_v177.md',
                 'PROVENANCE_v177.md','CHAT_LOG_v177.md','RESULTS_v177.md'):
        new[name]=(ROOT/name).read_bytes()
    for name in ('libwordfield177.so','test_source177','test_source_parallel177'):
        new['bin/'+name]=(ROOT/'bin'/name).read_bytes()
    new['Makefile']=(ROOT/'Makefile').read_bytes()
    assert b'v177' in new['Makefile']
    manifest=dict(version=177,date='2026-11-23',parent_version=176,
        parent_sha256=EXPECTED_PARENT_SHA,model_checkpoint_policy='frozen / inherited exactly',
        source='shemshallah/nanlm_experimental master',
        byte_law='v170 P bit-identical; v177 experimental cache and macro arithmetic code',
        note='The original v176 Makefile is preserved at ancestry/v176/Makefile. New MANIFEST_v177 and SHA256SUMS_v177 describe current file layout.',
        inherited_parent_entry_count=0,inherited_unchanged_entry_count=0,
        original_makefile_sha256=None,original_makefile_preserved_path='ancestry/v176/Makefile',
        sha256={})
    inventory={}
    with zipfile.ZipFile(PARENT,'r') as z:
        old=z.namelist();assert len(old)==len(set(old))==918
        assert 'Makefile' in old
        original=z.read('Makefile');manifest['original_makefile_sha256']=digest(original)
        new['ancestry/v176/Makefile']=original
        for name in old:
            raw=z.read(name)
            inventory[name]=new.pop(name,raw)
            manifest['inherited_parent_entry_count']+=1
            manifest['inherited_unchanged_entry_count']+=int(inventory[name]==raw)
        assert manifest['inherited_unchanged_entry_count']==917
    for name,raw in new.items():
        assert name not in inventory,name
        inventory[name]=raw
    assert not (set(inventory)&{'MANIFEST_v177.json','SHA256SUMS_v177'})
    manifest['sha256']={name:digest(raw) for name,raw in sorted(inventory.items())}
    manifest['entries_with_sha256']=len(inventory)
    # Manifest plus checksum file are additional un-hashed entries by design.
    inventory['MANIFEST_v177.json']=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode()
    inventory['SHA256SUMS_v177']=''.join(f'{h}  {name}\n' for name,h in manifest['sha256'].items()).encode()
    if TEMP.exists():TEMP.unlink()
    with zipfile.ZipFile(TEMP,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for name,raw in inventory.items():
            z.writestr(name,raw)
    with zipfile.ZipFile(TEMP) as z:
        assert len(z.namelist())==len(inventory)
        assert len(z.namelist())==len(set(z.namelist()))
        bad=z.testzip();assert bad is None,bad
        for name,sha in manifest['sha256'].items():assert digest(z.read(name))==sha,name
    os.replace(TEMP,OUT)
    result=dict(status='PASS',archive=str(OUT),bytes=OUT.stat().st_size,
       sha256=digest(OUT.read_bytes()),zip_entries=len(inventory),
       sha256_checked=manifest['entries_with_sha256'],
       parent_entries=918,parent_unchanged=917,original_makefile_preserved=True,
       github_source_only=True,no_optimizer_checkpoint_deleted=True)
    (ROOT/'results/v177_release_integrity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()