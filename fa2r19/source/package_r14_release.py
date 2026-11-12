#!/usr/bin/env python3
"""Reproducible single checkpoint archive, excluding redundant derived corpora cache."""
import pathlib,hashlib,json,zipfile,sys
root=pathlib.Path(sys.argv[1]).resolve();out=pathlib.Path(sys.argv[2]).resolve()
exclude={'SHA256SUMS','manifest.json','reports/package_output.json'}
files=sorted(p for p in root.rglob('*') if p.is_file() and
    not ({'__pycache__','.git','.pal_preprocessed'} & set(p.relative_to(root).parts)) and
    p.relative_to(root).as_posix() not in exclude and
    not p.name.endswith(('.tmp','.pyc')))
checkpoints=[p for p in files if p.relative_to(root).parts[0]=='checkpoint']
assert len(checkpoints)==1 and checkpoints[0].name=='r12_active.pal12',checkpoints
inventory=[]
for p in files:
    digest=hashlib.sha256();size=0
    with p.open('rb') as f:
        for blk in iter(lambda:f.read(2**20),b''):
            digest.update(blk);size+=len(blk)
    inventory.append({'path':p.relative_to(root).as_posix(),'bytes':size,'sha256':digest.hexdigest()})
m={'release':'PAL_R14_CPU_BROADTRAIN_R12_756_SINGLE_CHECKPOINT',
    'only_active_checkpoint':'checkpoint/r12_active.pal12','file_count':len(files)+2,'files':inventory}
(root/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
(root/'SHA256SUMS').write_text(''.join(f"{row['sha256']}  {row['path']}\n" for row in inventory)+
   f"{hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()}  manifest.json\n")
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
    for p in files+[root/'manifest.json',root/'SHA256SUMS']:
        z.write(p,p.relative_to(root).as_posix())
with zipfile.ZipFile(out) as z:
    assert z.testzip() is None
    assert [n for n in z.namelist() if n.startswith('checkpoint/')]==['checkpoint/r12_active.pal12']
    assert not [n for n in z.namelist() if '.pal_preprocessed/' in n]
print(json.dumps({'file_count':len(files)+2,'archive_size':out.stat().st_size,
  'archive_sha256':hashlib.file_digest(open(out,'rb'),'sha256').hexdigest(),
  'checkpoint_sha256':next(x['sha256'] for x in inventory if x['path']=='checkpoint/r12_active.pal12')},indent=2))