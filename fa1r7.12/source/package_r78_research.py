#!/usr/bin/env python3
"""Complete research release, exactly one last-accepted checkpoint; no rejected candidates."""
from pathlib import Path
import zipfile,json,hashlib,datetime,os
ROOT=Path(__file__).resolve().parent.parent
OUT=Path('/mnt/data/PAL_R7p8_ORDER_SENSITIVE_RESEARCH_ACCEPTED_R7p7_14416_FULL.zip')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
files=sorted(p for p in ROOT.rglob('*') if p.is_file() and 'checkpoint/pal_r78_' not in p.relative_to(ROOT).as_posix()
             and p.relative_to(ROOT).as_posix() not in ('MANIFEST.json','SHA256SUMS.txt'))
assert len([p for p in files if p.parent.name=='checkpoint'])==1
assert sha(ROOT/'checkpoint/pal_active.pt')=='54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d'
entries=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
manifest={'round':'PAL R7.8 experimental research','accepted_checkpoint_step':14416,
          'candidate_step_not_promoted':14912,'accepted_checkpoint':'checkpoint/pal_active.pt',
          'single_current_checkpoint':True,'files':entries}
(ROOT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
(ROOT/'SHA256SUMS.txt').write_text(''.join(f'{x["sha256"]}  {x["path"]}\n' for x in entries)
                   +f'{sha(ROOT/"MANIFEST.json")}  MANIFEST.json\n')
files.extend([ROOT/'MANIFEST.json',ROOT/'SHA256SUMS.txt'])
with zipfile.ZipFile(OUT,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
 for p in files:z.write(p,p.relative_to(ROOT).as_posix())
with zipfile.ZipFile(OUT) as z:
 assert z.testzip() is None
 names=z.namelist();assert sum(x.startswith('checkpoint/') for x in names)==1
 assert z.read('checkpoint/pal_active.pt')[:6] != b''
print(json.dumps({'path':str(OUT),'bytes':OUT.stat().st_size,'sha256':sha(OUT),
                  'files':len(files),'single_checkpoint':True},indent=2))
