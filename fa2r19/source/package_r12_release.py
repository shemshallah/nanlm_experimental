#!/usr/bin/env python3
"""Build one integrity-checked self-contained PAL R12 archive with one checkpoint."""
import hashlib,json,pathlib,zipfile,sys
root=pathlib.Path(sys.argv[1]).resolve();out=pathlib.Path(sys.argv[2]).resolve()
all_files=sorted(p for p in root.rglob('*') if p.is_file() and not any(x in str(p) for x in ('/__pycache__/','/.git/'))
           and p.relative_to(root).as_posix() not in ('SHA256SUMS','manifest.json','reports/package_output.json')
           and not p.name.endswith(('.tmp','.pyc')))
ckpts=[p for p in all_files if 'checkpoint/' in p.relative_to(root).as_posix()]
assert len(ckpts)==1 and ckpts[0].name=='r12_active.pal12',ckpts
inv=[]
for p in all_files:
    s=hashlib.sha256();size=0
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            s.update(block);size+=len(block)
    inv.append({'path':p.relative_to(root).as_posix(),'bytes':size,'sha256':s.hexdigest()})
manifest={'release':'PAL_R12_412_SINGLE_CHECKPOINT','file_count':len(inv)+2,
          'only_active_checkpoint':ckpts[0].relative_to(root).as_posix(),
          'files':inv}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
sha_lines=[f"{row['sha256']}  {row['path']}" for row in inv]
sha_lines.append(f"{hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()}  manifest.json")
(root/'SHA256SUMS').write_text('\n'.join(sha_lines)+'\n')
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
  for p in all_files+[root/'manifest.json',root/'SHA256SUMS']:
    z.write(p,p.relative_to(root).as_posix())
with zipfile.ZipFile(out) as z:
  bad=z.testzip();assert bad is None,bad
  assert sorted([x for x in z.namelist() if x.startswith('checkpoint/')])==['checkpoint/r12_active.pal12']
result={'archive':str(out),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
        'file_count':len(all_files)+2,'only_checkpoint':str(ckpts[0])}
print(json.dumps(result,indent=2))