#!/usr/bin/env python3
"""Build one current R7.10 reproducibility archive, no nested checkpoints/zips."""
import datetime,hashlib,json,pathlib,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=pathlib.Path('/mnt/data/PAL_R7p10_CHECKPOINT_HARD_NEGATIVES_ACCEPTED_14416_FULL.zip')

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def build():
    accepted='54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d'
    assert sha(ROOT/'checkpoint/pal_active.pt')==accepted
    forbidden=['.zip','.pending','.tmp','.bak']
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and
                 p.name not in ('MANIFEST.json','SHA256SUMS.txt') and
                 not p.name.endswith('.pyc') and '__pycache__' not in p.parts and
                 not any(p.name.endswith(ext) for ext in forbidden))
    checkpoints=[p for p in files if 'checkpoint' in p.relative_to(ROOT).parts and p.suffix=='.pt']
    assert len(checkpoints)==1 and checkpoints[0].name=='pal_active.pt'
    assert not any(p.suffix.lower()=='.zip' for p in files)
    met={'release':'PAL R7.10 checkpoint hard-negative memory diagnostics',
         'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'lineage':'PAL R7.7 -> R7.8/R7.9 research -> R7.10 read-only diagnostics',
         'accepted_step':14416,'new_training_updates':0,'promoted':False,
         'accepted_checkpoint_sha256':accepted,
         'active_checkpoint':'checkpoint/pal_active.pt',
         'optimizer_histories':5,'historical_evaluation_only':True,
         'files':{str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}}
    (ROOT/'MANIFEST.json').write_text(json.dumps(met,indent=2)+'\n')
    files.append(ROOT/'MANIFEST.json')
    with (ROOT/'SHA256SUMS.txt').open('w') as f:
        for p in sorted(files):f.write(f'{sha(p)}  {p.relative_to(ROOT)}\n')
    files.append(ROOT/'SHA256SUMS.txt')
    with zipfile.ZipFile(OUT,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5,allowZip64=True) as z:
        for p in sorted(files):z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(OUT) as z:
        corrupt=z.testzip()
        assert corrupt is None,corrupt
        members=z.namelist()
        assert members.count('checkpoint/pal_active.pt')==1
        assert z.read('checkpoint/pal_active.pt')[:4]==(ROOT/'checkpoint/pal_active.pt').read_bytes()[:4]
    result={'path':str(OUT),'sha256':sha(OUT),'size_bytes':OUT.stat().st_size,
            'entry_count':len(files),'checkpoint_sha256':accepted}
    (ROOT/'reports/R7p10_packaging_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':build()
