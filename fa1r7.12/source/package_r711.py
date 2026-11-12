#!/usr/bin/env python3
"""PAL R7.11 single-parent archive with fully resumable 173KB experimental delta."""
import datetime,hashlib,json,pathlib,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=pathlib.Path('/mnt/data/PAL_R7p11_DIRECTED_HARD_NEGATIVE_SINGLE_CHECKPOINT_FULL.zip')
ACCEPTED='54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d'
DELTA='a7ff6f371f7a08474d80f74195c0a081e96ea71c183aa3d6f3b16c8a8898402c'

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()

def main():
 assert sha(ROOT/'checkpoint/pal_active.pt')==ACCEPTED
 assert sha(ROOT/'adaptation/r711_experimental_delta.pt')==DELTA
 forbidden={'.zip','.pending','.tmp','.bak'}
 files=sorted(p for p in ROOT.rglob('*') if p.is_file()
   and p.name not in ('MANIFEST.json','SHA256SUMS.txt','R7p11_packaging_summary.json')
   and not p.name.endswith('.pyc') and '__pycache__' not in p.parts
   and not any(p.name.endswith(ext) for ext in forbidden))
 checkfiles=[p for p in files if 'checkpoint' in p.relative_to(ROOT).parts and p.suffix=='.pt']
 assert len(checkfiles)==1 and checkfiles[0].name=='pal_active.pt',checkfiles
 assert not any(p.suffix.lower()=='.zip' for p in files)
 assert len([p for p in files if p.suffix=='.pt'])==2, 'one full checkpoint and one overlay only'
 report={'release':'PAL R7.11 ordered directed historical recall research, unpromoted',
  'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'lineage':'PAL R7.7 accepted 14416 -> R7.8/R7.9/R7.10 research -> R7.11 directed 80-step selected auxiliary overlay',
  'accepted_lm_joint_steps':14416,'auxiliary_updates_attempted':320,'auxiliary_updates_selected':80,
  'promoted':False,'full_checkpoints':1,'inherited_optimizer_histories':5,
  'accepted_checkpoint':'checkpoint/pal_active.pt','accepted_checkpoint_sha256':ACCEPTED,
  'experimental_overlay':'adaptation/r711_experimental_delta.pt','experimental_overlay_sha256':DELTA,
  'whole_book_sealed_test':False,'original_historical_corpus_has_doc_start_metadata':False,
  'files':{str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}}
 (ROOT/'MANIFEST.json').write_text(json.dumps(report,indent=2)+'\n')
 files.append(ROOT/'MANIFEST.json')
 with (ROOT/'SHA256SUMS.txt').open('w') as f:
  for p in sorted(files):f.write(f'{sha(p)}  {p.relative_to(ROOT)}\n')
 files.append(ROOT/'SHA256SUMS.txt')
 with zipfile.ZipFile(OUT,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5,allowZip64=True) as z:
  for p in sorted(files):z.write(p,p.relative_to(ROOT).as_posix())
 with zipfile.ZipFile(OUT) as z:
  assert z.testzip() is None
  assert z.namelist().count('checkpoint/pal_active.pt')==1
  assert z.namelist().count('adaptation/r711_experimental_delta.pt')==1
  assert z.namelist().count('MANIFEST.json')==1
  assert z.namelist().count('SHA256SUMS.txt')==1
 result={'path':str(OUT),'sha256':sha(OUT),'size_bytes':OUT.stat().st_size,
    'entries':len(files),'accepted_checkpoint_sha256':ACCEPTED,'experimental_overlay_sha256':DELTA,
    'archive_crc':'PASS','one_full_checkpoint':True}
 (ROOT/'reports/R7p11_packaging_summary.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))

if __name__=='__main__':main()
