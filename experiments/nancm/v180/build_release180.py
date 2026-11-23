#!/usr/bin/env python3
"""One cumulative, deterministic NaNCM v180 recovery ZIP; inherits v179 byte-exact.
No nested ZIP, no checkpoint/optimizer deletion or modification; SHA256 checked.
"""
from __future__ import annotations
import hashlib,json,os,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
PARENT=Path('/mnt/data/NaNCM_v179_COW_PARALLEL_FULLCONTEXT_RECOVERY_20261123.zip')
PARENT_SHA='bac062c4ece8cdd7099e08d1ca5c37e7f186b41615ea35337678c38127d3af4c'
OUT=Path('/mnt/data/NaNCM_v180_SIGNED_WORD_MEMORY_RECOVERY_20261123.zip')
TEMP=OUT.with_suffix('.tmp.zip')

def sha(b):return hashlib.sha256(b).hexdigest()
def sha_file(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()

def main():
 assert PARENT.is_file() and sha_file(PARENT)==PARENT_SHA,'parent v179 absent or checksum mismatch'
 new={}
 for p in sorted((ROOT/'experiments/nancm/v180').iterdir()):
  if p.is_file() and p.name!='RELEASE_180.json':new[str(p.relative_to(ROOT))]=p.read_bytes()
 for name in ('libwordfield180.so','test_source180','nancm_v180_memory'):
  path=ROOT/'bin'/name
  if not path.is_file():raise RuntimeError('missing native binary: '+name)
  new[str(path.relative_to(ROOT))]=path.read_bytes()
 for p in sorted((ROOT/'results/v180').iterdir()):
  if p.is_file() and p.name!='v180_release_integrity.json':
   new[str(p.relative_to(ROOT))]=p.read_bytes()
 for name in ('README_v180.md','CLAIM_LEDGER_v180.md','RESEARCH_LOG_v180.md',
              'PROVENANCE_v180.md','CHAT_LOG_v180.md','RESULTS_v180.md'):
  new[name]=(ROOT/name).read_bytes()
 new['.github/workflows/nancm_v180_source.yml']=(ROOT/'.github/workflows/nancm_v180_source.yml').read_bytes()
 new['Makefile']=(ROOT/'Makefile').read_bytes()
 assert b'include experiments/nancm/v180/Makefile.fragment' in new['Makefile']
 inherited={}
 with zipfile.ZipFile(PARENT) as z:
  names=z.namelist()
  if len(names)!=1028 or len(names)!=len(set(names)) or z.testzip() is not None:
   raise RuntimeError('inherited v179 invalid')
  for name in names:inherited[name]=z.read(name)
 original_makefile=inherited['Makefile']
 new['ancestry/v179/Makefile']=original_makefile
 modifications={p for p in inherited if p in new and inherited[p]!=new[p]}
 if modifications!={'Makefile'}:raise RuntimeError(f'unexpected inherited modifications: {modifications}')
 duplicates=(set(new)&set(inherited))-{'Makefile'}
 if duplicates:raise RuntimeError(f'duplicate inherited name: {sorted(duplicates)}')
 inventory=dict(inherited);inventory.update(new)
 special=('MANIFEST_v180.json','SHA256SUMS_v180')
 if any(x in inventory for x in special):raise RuntimeError('manifest collision')
 digests={p:sha(b) for p,b in sorted(inventory.items())}
 m=json.loads((ROOT/'results/v180/v180_frozen_word_gate.json').read_text())
 manifest=dict(version=180,date='2026-11-23',parent_version=179,parent_sha256=PARENT_SHA,
  inherited_entries=len(inherited),inherited_unchanged=len(inherited)-1,
  original_makefile_sha256=sha(original_makefile),original_makefile_location='ancestry/v179/Makefile',
  trained_checkpoint_optimizer_moments='unchanged inherited all versions',
  train_reconstruction='396 TRAIN books, 2,650,489 valid transitions',
  gamma=[.1,.1,.1,.1],
  native_full_validation_top1=m['v180_top1'],parent_top1=m['parent_top1'],
  native_full_validation_top5=m['v180_top5'],parent_top5=m['parent_top5'],
  val_word_nll=m['v180_word_nll'],val_parent_word_nll=m['parent_word_nll'],
  val_status='reused 55-book VAL, not pristine TEST',user_target_top1=0.5,target_met=False,
  github='shemshallah/nanlm_experimental master source, no trained weights uploaded',
  per_entry_sha256=digests)
 inventory['MANIFEST_v180.json']=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
 inventory['SHA256SUMS_v180']=''.join(f'{checksum}  {p}\n' for p,checksum in digests.items()).encode()
 if TEMP.exists():TEMP.unlink()
 with zipfile.ZipFile(TEMP,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
  for name,datum in inventory.items():
   zi=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));zi.create_system=3;zi.external_attr=0o644<<16
   zi.compress_type=zipfile.ZIP_DEFLATED
   z.writestr(zi,datum,compress_type=zipfile.ZIP_DEFLATED,compresslevel=1)
 with zipfile.ZipFile(TEMP) as z:
  if z.testzip() is not None:raise RuntimeError('release CRC invalid')
  names=z.namelist()
  if len(names)!=len(inventory) or len(set(names))!=len(inventory):raise RuntimeError('release name count')
  for name,checksum in digests.items():
   if sha(z.read(name))!=checksum:raise RuntimeError('digest mismatch '+name)
  for name,datum in inherited.items():
   if name!='Makefile' and z.read(name)!=datum:raise RuntimeError('inherited mutation '+name)
 os.replace(TEMP,OUT)
 report=dict(status='PASS',archive=str(OUT),bytes=OUT.stat().st_size,
  sha256=sha_file(OUT),entries=len(inventory),sha256_payload_entries=len(digests),
  parent_entries=len(inherited),unchanged_parent_entries=len(inherited)-1,
  original_makefile_preserved=True,crc='PASS',native_all_context_word_parity=True,
  optimizer_checkpoint_preserved=True,all_inherited_checkpoints=True,
  no_nested_zip=True,reproducible_timestamp_metadata=True)
 (ROOT/'results/v180/v180_release_integrity.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
