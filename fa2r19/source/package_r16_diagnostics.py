#!/usr/bin/env python3
"""Build a single current complete R16 diagnostics archive (same R15 weights)."""
import hashlib,json,pathlib,zipfile,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'PAL_R16_CONTENT_ATTENTION_GEOMETRY_DIAGNOSTICS_SINGLE_CHECKPOINT_full.zip'
PARENT='dc61500ec3abd424409f136ffff517242134af77435e66f135f4363cb7ec450f'

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def include(p):
 rel=p.relative_to(ROOT).as_posix()
 if '.pal_preprocessed/' in rel:return False
 if rel in ('manifest.json','SHA256SUMS'):return False
 if p.suffix in ('.tmp','.pyc'):return False
 if '__pycache__' in p.parts:return False
 if rel=='reports/r16_content_attention_ablation.json':return False # duplicate of 400chunks
 return True

def main():
 assert sha(ROOT/'checkpoint/r12_active.pal12')==PARENT
 files=sorted([p for p in ROOT.rglob('*') if p.is_file() and include(p)],key=lambda p:p.relative_to(ROOT).as_posix())
 rows=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
 manifest={'release':'PAL_R16_DIAGNOSTICS_R15_UNIFIED_CHECKPOINT','checkpoint':'checkpoint/r12_active.pal12','checkpoint_sha256':PARENT,
  'r12_updates':10756,'r16_promoted_updates':0,'file_count_excluding_manifest_and_checksums':len(rows),'files':rows,
  'nested_archives':False,'new_training_data_added':False}
 (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 files.append(ROOT/'manifest.json')
 with (ROOT/'SHA256SUMS').open('w') as fh:
  for p in sorted(files,key=lambda p:p.relative_to(ROOT).as_posix()):
   fh.write(sha(p)+'  '+p.relative_to(ROOT).as_posix()+'\n')
 files.append(ROOT/'SHA256SUMS')
 with zipfile.ZipFile(OUT,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
  for p in files:z.write(p,p.relative_to(ROOT).as_posix())
 with zipfile.ZipFile(OUT,'r') as z:
  bad=z.testzip();assert bad is None,bad
  assert len(z.namelist())==len(files)
  for row in rows:assert hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256'],row['path']
  assert hashlib.sha256(z.read('checkpoint/r12_active.pal12')).hexdigest()==PARENT
 print(json.dumps({'zip':str(OUT),'bytes':OUT.stat().st_size,'files':len(files),'sha256':sha(OUT),'checkpoint_unchanged':True},indent=2))
if __name__=='__main__':main()