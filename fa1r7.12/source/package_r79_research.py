#!/usr/bin/env python3
"""PAL R7.9 cumulative research release, exactly one accepted model checkpoint."""
from pathlib import Path
import hashlib,json,zipfile,time
ROOT=Path(__file__).resolve().parent.parent
OUT=Path('/mnt/data/PAL_R7p9_MEMORY_DIAGNOSTICS_ACCEPTED_R7p7_14416_FULL.zip')
EXPECTED='54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d'
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
assert sha(ROOT/'checkpoint/pal_active.pt')==EXPECTED, 'ACCEPTED CHECKPOINT MODIFIED'
files=[]
for p in ROOT.rglob('*'):
 if not p.is_file():continue
 rel=p.relative_to(ROOT).as_posix()
 if rel.startswith('checkpoint/') and rel!='checkpoint/pal_active.pt':continue
 if '__pycache__' in p.parts or rel.endswith(('.pyc','.tmp','.zip')):continue
 if rel in ('MANIFEST.json','SHA256SUMS.txt'):continue
 files.append(p)
files.sort()
assert len([p for p in files if p.relative_to(ROOT).as_posix().startswith('checkpoint/')])==1
entries=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
manifest={'release':'PAL R7.9 research continuation','candidate_validation':'failed, NO PROMOTION',
          'accepted_model_step':14416,'accepted_model_sha256':EXPECTED,
          'rank':64,'physical_memory_matrices':1,'optimizer_histories':5,
          'ordinary_continuation_trial_step':14928,'normalized_read_trial_step':14544,
          'independent_book_testing':False,'rejected_candidate_checkpoints_packaged':False,
          'total_structural_regressions_passed':30,'files':entries}
(ROOT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
(ROOT/'SHA256SUMS.txt').write_text(''.join(f"{r['sha256']}  {r['path']}\n" for r in entries)
                                  +f"{sha(ROOT/'MANIFEST.json')}  MANIFEST.json\n")
files.extend((ROOT/'MANIFEST.json',ROOT/'SHA256SUMS.txt'))
t=time.monotonic()
with zipfile.ZipFile(OUT,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
 for p in files:z.write(p,p.relative_to(ROOT).as_posix())
with zipfile.ZipFile(OUT) as z:
 assert z.testzip() is None
 contents=z.namelist()
 assert sum(n.startswith('checkpoint/') for n in contents)==1
 assert z.read('MANIFEST.json')
 print(json.dumps({'path':str(OUT),'sha256':sha(OUT),'bytes':OUT.stat().st_size,
                   'file_count':len(contents),'only_accepted_checkpoint':True,
                   'accepted_checkpoint_sha256':EXPECTED,'seconds':round(time.monotonic()-t,2)},indent=2))
