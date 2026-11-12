#!/usr/bin/env python3
"""One cumulative R7.12 recoverable archive: one accepted parent + two overlays."""
import datetime,hashlib,json,pathlib,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=pathlib.Path('/mnt/data/PAL_R7p12_STREAMING_FRONTIER_BRIDGE_SINGLE_CHECKPOINT_FULL.zip')
ACCEPTED='54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d'
R711='a7ff6f371f7a08474d80f74195c0a081e96ea71c183aa3d6f3b16c8a8898402c'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for buf in iter(lambda:f.read(8<<20),b''):h.update(buf)
 return h.hexdigest()

def main():
 parent=ROOT/'checkpoint/pal_active.pt'
 assert sha(parent)==ACCEPTED
 assert sha(ROOT/'adaptation/r711_experimental_delta.pt')==R711
 gate=ROOT/'adaptation/r712_gate_delta.pt'
 assert sha(gate)==gate.with_suffix(gate.suffix+'.sha256').read_text().split()[0]
 files=sorted(p for p in ROOT.rglob('*') if p.is_file()
   and p.name not in ('MANIFEST.json','SHA256SUMS.txt','R7p12_packaging_summary.json')
   and not p.name.endswith(('.pyc','.pending','.tmp','.bak','.zip'))
   and '__pycache__' not in p.parts)
 full=[p for p in files if 'checkpoint' in p.relative_to(ROOT).parts and p.suffix=='.pt']
 assert full==[parent],full
 overlays=sorted(p for p in files if p.parent.name=='adaptation' and p.suffix=='.pt')
 assert [p.name for p in overlays]==['r711_experimental_delta.pt','r712_gate_delta.pt'],overlays
 manifest={'release':'PAL R7.12 experimental causal streaming and LM bridge; unpromoted',
  'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'accepted_joint_steps':14416,'accepted_full_checkpoint':str(parent.relative_to(ROOT)),
  'accepted_checkpoint_sha256':ACCEPTED,'five_accepted_optimizers_unchanged':True,
  'r711_directed_selected_aux_updates':80,
  'r712_gate_updates_attempted':128,'r712_gate_promoted':False,
  'reliable_retrial_256_logged_updates_selected':False,
  'historical_splits_not_independent_sealed_books':True,'unseen_nanrouter_books_evaluated':False,
  'all_new_overlays_experimental':True,
  'files':{str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}}
 (ROOT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
 files.append(ROOT/'MANIFEST.json')
 with (ROOT/'SHA256SUMS.txt').open('w') as f:
  for p in sorted(files):f.write(f'{sha(p)}  {p.relative_to(ROOT)}\n')
 files.append(ROOT/'SHA256SUMS.txt')
 with zipfile.ZipFile(OUT,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
  for p in sorted(files):z.write(p,p.relative_to(ROOT).as_posix())
 with zipfile.ZipFile(OUT) as z:
  assert z.testzip() is None
  names=z.namelist()
  assert names.count('checkpoint/pal_active.pt')==1
  assert len([x for x in names if x.endswith('.pt')])==3
  assert len([x for x in names if x.endswith('.zip')])==0
 report={'path':str(OUT),'sha256':sha(OUT),'bytes':OUT.stat().st_size,'entries':len(files),
  'archive_crc':'PASS','one_full_checkpoint':True,'accepted_sha256':ACCEPTED,
  'overlays':{p.name:sha(p) for p in overlays}}
 (ROOT/'reports/R7p12_packaging_summary.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
