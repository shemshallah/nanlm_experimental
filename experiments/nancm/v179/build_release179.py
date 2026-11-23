#!/usr/bin/env python3
"""Deterministic, fully recoverable, cumulative v179 ZIP release builder.

Parent v178 = 989 verified entries; only root Makefile may change, and the
original goes into ancestry/v178/Makefile. Every trained checkpoint and Adam
state is inherited unchanged. ZipInfo metadata is fixed for SHA stability.
"""
from __future__ import annotations
import hashlib,json,os,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
PARENT=Path('/mnt/data/NaNCM_v178_SHARED_HUB_EXACT_VERIFIER_RECOVERY_20261123.zip')
PARENT_SHA='d7f67c4a17a96fc0f7d0a196971f80c3fa9da6713a60fe2f888665ea4022deeb'
OUT=Path('/mnt/data/NaNCM_v179_COW_PARALLEL_FULLCONTEXT_RECOVERY_20261123.zip')
TEMP=OUT.with_suffix('.tmp.zip')

def sha(b):return hashlib.sha256(b).hexdigest()
def sha_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def main():
    assert PARENT.is_file() and sha_file(PARENT)==PARENT_SHA,'missing or mismatched v178 parent'
    new={}
    for p in sorted((ROOT/'experiments/nancm/v179').iterdir()):
        if p.is_file() and p.name!='RELEASE_179.json':
            new[str(p.relative_to(ROOT))]=p.read_bytes()
    for name in ('libwordfield179.so','test_source179','native_trained179'):
        path=ROOT/'bin'/name
        if not path.is_file():raise RuntimeError('compiled binary absent: '+name)
        new[str(path.relative_to(ROOT))]=path.read_bytes()
    for p in sorted((ROOT/'results').glob('v179_*')):
        if p.name=='v179_release_integrity.json':continue
        if p.is_file() and p.suffix in ('.json','.log','.bin'):
            new[str(p.relative_to(ROOT))]=p.read_bytes()
    for name in ('README_v179.md','CLAIM_LEDGER_v179.md','RESEARCH_LOG_v179.md',
                 'PROVENANCE_v179.md','CHAT_LOG_v179.md','RESULTS_v179.md'):
        new[name]=(ROOT/name).read_bytes()
    new['.github/workflows/nancm_v179_source.yml']=(ROOT/'.github/workflows/nancm_v179_source.yml').read_bytes()
    new['Makefile']=(ROOT/'Makefile').read_bytes()
    if b'include experiments/nancm/v179/Makefile.fragment' not in new['Makefile']:
        raise RuntimeError('v179 is absent from root Makefile')
    inherited={}
    with zipfile.ZipFile(PARENT,'r') as z:
        names=z.namelist()
        if len(names)!=989 or len(names)!=len(set(names)) or z.testzip() is not None:
            raise RuntimeError('v178 inherited archive integrity failure')
        previous=z.read('Makefile')
        new['ancestry/v178/Makefile']=previous
        for name in names:inherited[name]=z.read(name)
    modifications={name for name in inherited if name in new and inherited[name]!=new[name]}
    if modifications!={'Makefile'}:
        raise RuntimeError(f'Unexpected inherited data mutation: {modifications}')
    dup=set(new)&(set(inherited)-{'Makefile'})
    if dup:raise RuntimeError(f'duplicate inherited paths: {sorted(dup)}')
    inventory=dict(inherited);inventory.update(new)
    protect=('MANIFEST_v179.json','SHA256SUMS_v179')
    if any(x in inventory for x in protect):raise RuntimeError('v179 manifest collision')
    digests={name:sha(value) for name,value in sorted(inventory.items())}
    summary=json.loads((ROOT/'experiments/nancm/v179/RESULTS.json').read_text())
    manifest=dict(version=179,date='2026-11-23',parent_version=178,parent_sha256=PARENT_SHA,
        parent_entries=len(inherited),inherited_unchanged=len(inherited)-len(modifications),
        original_makefile_sha256=sha(previous),original_makefile_location='ancestry/v178/Makefile',
        trained_checkpoints_and_optimizer_moments='exactly inherited; unchanged',
        github='shemshallah/nanlm_experimental master, source-only',
        tested_48_book_four_worker_speedup_mean=summary['full_context_4worker_mean_speedup'],
        deployment='opt-in COW Linux native multi-context, single-stream default protected',
        per_entry_sha256=digests)
    inventory['MANIFEST_v179.json']=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
    inventory['SHA256SUMS_v179']=''.join(f'{x}  {name}\n' for name,x in digests.items()).encode()
    if TEMP.exists():TEMP.unlink()
    with zipfile.ZipFile(TEMP,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for name,data in inventory.items():
            zi=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            zi.create_system=3;zi.external_attr=0o644<<16
            zi.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(zi,data,compress_type=zipfile.ZIP_DEFLATED,compresslevel=1)
    with zipfile.ZipFile(TEMP) as z:
        if z.testzip() is not None:raise RuntimeError('new release failed CRC')
        names=z.namelist()
        if len(names)!=len(inventory) or len(set(names))!=len(names):raise RuntimeError('new release duplicate/missing entries')
        for name,d in digests.items():
            if sha(z.read(name))!=d:raise RuntimeError('SHA mismatch: '+name)
        for name,data in inherited.items():
            if name!='Makefile' and z.read(name)!=data:raise RuntimeError('Inherited file mutation: '+name)
    os.replace(TEMP,OUT)
    result=dict(status='PASS',path=str(OUT),bytes=OUT.stat().st_size,
        sha256=sha_file(OUT),entries=len(inventory),sha256_payload_entries=len(digests),
        inherited_entries=len(inherited),inherited_unchanged=len(inherited)-1,
        crc='PASS',reproducible_zip_metadata=True,old_makefile_preserved=True,
        trained_checkpoints_unchanged=True,no_optimizer_moments_modified=True)
    (ROOT/'results/v179_release_integrity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
