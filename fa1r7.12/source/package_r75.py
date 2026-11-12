#!/usr/bin/env python3
"""Package a single reproducible PAL R7.5 release, no nested old ZIPs."""
import argparse,hashlib,json,pathlib,zipfile,time
import torch

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(4<<20),b''):h.update(c)
 return h.hexdigest()

def package(root,out):
 root=pathlib.Path(root);out=pathlib.Path(out)
 ck=torch.load(root/'checkpoint/pal_active.pt',map_location='cpu',weights_only=False)
 assert ck['format']=='PAL_R7_5_ASSOCIATIVE_V1' and ck['joint_steps']>=3500
 assert list(ck['optimizers'])==['legacy_accum','adapter_adamw','mirror_adamw','outer_adamw','retrieval_adamw']
 chosen=[]
 for dirname in ('source','corpus','tokenizer','reports'):
  chosen.extend(p for p in (root/dirname).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
 chosen.extend([root/'checkpoint/pal_active.pt',root/'README.md',root/'RELEASE_NOTES.md',root/'CLAIM_LEDGER.md'])
 sums={str(p.relative_to(root)):sha(p) for p in sorted(chosen)}
 manifest={'format':'PAL_R7_5_ASSOCIATIVE_RELEASE_V1','joint_step':int(ck['joint_steps']),
    'single_active_checkpoint':'checkpoint/pal_active.pt',
    'optimizer_state_names':list(ck['optimizers']),'file_sha256':sums}
 (root/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
 (root/'SHA256SUMS.txt').write_text(''.join(f'{v}  {k}\n' for k,v in sums.items()))
 chosen.extend([root/'MANIFEST.json',root/'SHA256SUMS.txt'])
 out.parent.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as archive:
  for p in chosen:archive.write(p,p.relative_to(root).as_posix())
 with zipfile.ZipFile(out) as z:
  bad=z.testzip();assert bad is None,bad
  assert len(z.namelist())==len(set(z.namelist()))
 print(json.dumps({'file':str(out),'size_bytes':out.stat().st_size,
   'entries':len(chosen),'joint_step':ck['joint_steps'],'sha256':sha(out)}))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--root',default='.');a.add_argument('--output',required=True)
 q=a.parse_args();package(q.root,q.output)
