#!/usr/bin/env python3
"""Package a single R15 recoverable checkpoint and reproducibility materials."""
import pathlib,hashlib,json,zipfile,sys
root=pathlib.Path(sys.argv[1]).resolve();out=pathlib.Path(sys.argv[2]).resolve()
exclude={'SHA256SUMS','manifest.json','reports/r15_train.pid','reports/r14_package.json'}
files=sorted(p for p in root.rglob('*') if p.is_file() and
   not ({'__pycache__','.git','.pal_preprocessed'} & set(p.relative_to(root).parts)) and
   p.relative_to(root).as_posix() not in exclude and not p.name.endswith(('.tmp','.pyc')))
ckpts=[p.relative_to(root).as_posix() for p in files if p.relative_to(root).parts[0]=='checkpoint']
assert ckpts==['checkpoint/r12_active.pal12'],ckpts
rep=json.loads((root/'reports/r15_1000_then_9000.json').read_text())
assert rep['completed_all_10000'] and rep['promoted_r12_updates']==10756
inventory=[]
for p in files:
 with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 inventory.append({'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':digest})
assert next(x['sha256'] for x in inventory if x['path']==ckpts[0])==rep['active_checkpoint_sha256']
m={'release':'PAL_R15_CPU_10000_ADDITIONAL_R12_10756_SINGLE_CHECKPOINT',
   'current_checkpoint':ckpts[0],'file_count':len(inventory)+2,'files':inventory}
(root/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
(root/'SHA256SUMS').write_text(''.join(f"{i['sha256']}  {i['path']}\n" for i in inventory)+
 f"{hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()}  manifest.json\n")
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
 for p in files+[root/'manifest.json',root/'SHA256SUMS']:
  z.write(p,p.relative_to(root).as_posix())
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None
 assert [n for n in z.namelist() if n.startswith('checkpoint/')]==ckpts
 assert not any('/.pal_preprocessed/' in n or n.startswith('nanrouter/.pal_preprocessed/') for n in z.namelist())
with out.open('rb') as f:zip_sha=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps({'file_count':len(inventory)+2,'archive_bytes':out.stat().st_size,
 'archive_sha256':zip_sha,'checkpoint_sha256':rep['active_checkpoint_sha256']},indent=2))