#!/usr/bin/env python3
"""Package exactly one latest R27 recovery ZIP with per-file checksums."""
import argparse,hashlib,json,pathlib,zipfile,os

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for part in iter(lambda:f.read(4<<20),b''):h.update(part)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',default='.')
 ap.add_argument('--output',default='/mnt/data/FA2_R27_UNIFIED_LATEST_RECOVERY.zip')
 args=ap.parse_args();root=pathlib.Path(args.root).resolve();out=pathlib.Path(args.output)
 audit=json.loads((root/'reports/r27_inheritance_audit.json').read_text())
 if audit['optimizer_groups']!=12 or audit['optimizer_states']!=144 or audit['inherited_adam_states_exact']!=137:
  raise ValueError('not an audited R27 checkpoint')
 required=[root/'reports/fa2_r27_unified_candidate.pth',root/'reports/fa2_2_r21_gate_candidate.pth',
  root/'reports/r22_multiscale_candidate.pth',root/'reports/r27_inheritance_audit.json',
  root/'reports/r27_realbook_local_pilot.json',root/'reports/r27_reserve_long_132k.json',
  root/'reports/r27_extracted_cpu_cache.pth',root/'reports/r27_local_overlay.pt',
  root/'logs/r27_pilot.log',root/'logs/r27_pilot_v2.log',root/'logs/r27_reserved.log',
  root/'R27_RECOVERY_README.md',root/'requirements.txt']
 files=sorted((root/'source').glob('*.py'))+required
 if not all(x.is_file() for x in files):raise FileNotFoundError('missing '+str([str(x) for x in files if not x.is_file()]))
 if audit['r27_checkpoint_sha256']!=sha(required[0]):raise ValueError('checkpoint SHA mismatch')
 record={f.relative_to(root).as_posix():{'sha256':sha(f),'bytes':f.stat().st_size} for f in files}
 sums=''.join(f'{d["sha256"]}  {name}\n' for name,d in sorted(record.items()))
 manifest={'format':'FA2_R27_SINGLE_UNIFIED_RECOVERY_V1','selected_r27_step':audit['r27_step'],
           'legacy_states_exact':audit['inherited_adam_states_exact'],
           'optimizer_groups':audit['optimizer_groups'],'optimizer_states':audit['optimizer_states'],
           'r18_ancestor_sha256':audit['r18_parent_sha256'],
           'r26_parent_sha256':audit['r26_parent_sha256'],
           'r27_checkpoint_sha256':audit['r27_checkpoint_sha256'],
           'separate_original_R19_release_required':True,
           'separate_R26_binary_required':False,'accepted_model_promoted':False,
           'files':record}
 out.parent.mkdir(parents=True,exist_ok=True);temp=out.with_suffix('.tmp')
 with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
  for f in files:z.write(f,f.relative_to(root).as_posix())
  z.writestr('reports/r27_latest_manifest.json',json.dumps(manifest,sort_keys=True,indent=2)+'\n')
  z.writestr('SHA256SUMS.txt',sums)
 with zipfile.ZipFile(temp) as z:
  assert z.testzip() is None
  for name,attrs in record.items():
   if hashlib.sha256(z.read(name)).hexdigest()!=attrs['sha256']:raise ValueError('corrupted '+name)
 temp.replace(out)
 print('R27_ARCHIVE_PASS',json.dumps({'path':str(out),'bytes':out.stat().st_size,
  'sha256':sha(out),'members':len(record)+2,'checkpoint_sha256':audit['r27_checkpoint_sha256']}))

if __name__=='__main__':main()