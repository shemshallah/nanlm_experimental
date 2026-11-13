#!/usr/bin/env python3
"""Deterministic cumulative FA1 R7 source/reports release, latest round only."""
import argparse,hashlib,json,pathlib,zipfile

def sha(b):return hashlib.sha256(b).hexdigest()
def eligible(root):
 for p in root.rglob('*'):
  if not p.is_file():continue
  rel=p.relative_to(root)
  if any(x in ('checkpoint','__pycache__','.git') for x in rel.parts):continue
  if p.suffix.lower() in ('.pt','.pth','.bin','.db','.sqlite','.gguf','.tmp','.pyc','.zip','.zst'):continue
  if rel.name in ('MANIFEST.json','SHA256SUMS'):continue
  yield p

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1])
 ap.add_argument('--output',type=pathlib.Path,default=pathlib.Path('/mnt/data/FA1_R7p22_UNIFIED_CPU_SOURCE.zip'))
 a=ap.parse_args();root=a.root;files=sorted(eligible(root));assets={p.relative_to(root).as_posix():{'sha256':sha(p.read_bytes()),'size':p.stat().st_size} for p in files}
 manifest={'release':'FA1 R7.22 single unified R7 on master',
  'accepted_parent_step':14416,'accepted_checkpoint_sha256':'54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d',
  'r721_ancestor_sha256':'7d2ff0c17430075e3fc6ddeb02f61bf447c07eed027ae77afac0506737734b1f',
  'r722_unified_candidate_sha256':'eb0b7f0505f027095af23bc85ef00f2c036f3d2d5fb160ee0c45e60fb02e3455',
  'model_parameter_tensors':105,'optimizer_histories':5,
  'semantic_updates_executed':48,'selected_semantic_updates':0,
  'newbook_validation_passed':False,'promoted':False,
  'one_model':True,'no_other_r7_branches':True,'binary_checkpoint_included':False,'assets':assets}
 (root/'MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 files.append(root/'MANIFEST.json')
 (root/'SHA256SUMS').write_text(''.join(f'{sha(p.read_bytes())}  {p.relative_to(root).as_posix()}\n' for p in sorted(files)))
 files.append(root/'SHA256SUMS')
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(a.output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for p in sorted(files):
   info=zipfile.ZipInfo(p.relative_to(root).as_posix(),date_time=(2026,11,12,16,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
   z.writestr(info,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
 with zipfile.ZipFile(a.output) as z:
  assert z.testzip() is None
  assert len(z.namelist())==len(set(z.namelist()))==len(files)
  for n,s in assets.items():assert sha(z.read(n))==s['sha256']
  assert not any(n.endswith(('.pt','.zip','.db','.bin')) or n.startswith('checkpoint/') for n in z.namelist())
 print(json.dumps({'archive':str(a.output),'size_bytes':a.output.stat().st_size,'sha256':sha(a.output.read_bytes()),'files':len(files),'verified':True,'binary_weights_included':False},indent=2))
if __name__=='__main__':main()