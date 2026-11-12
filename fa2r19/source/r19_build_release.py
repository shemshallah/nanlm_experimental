#!/usr/bin/env python3
"""Build one cumulative PAL release ZIP with one accepted checkpoint.

No nested zips, generated preprocessed caches or stale alternative weights.
All included source/corpora/reports/checkpoint bytes are SHA-256 manifested.
"""
import argparse,hashlib,json,pathlib,zipfile

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()

def main():
    a=argparse.ArgumentParser();a.add_argument('--root',default='.');a.add_argument('--out',default=None);arg=a.parse_args()
    root=pathlib.Path(arg.root).resolve()
    out=pathlib.Path(arg.out) if arg.out else root.parent/'PAL_R19_SEMANTIC_FRONTIER_AUDIT_ONE_CHECKPOINT_FULL.zip'
    checkpoint=root/'checkpoint/r12_active.pal12'
    expected='dd040106568786e7fbad2a880f4010635b626b51c3697e2ab08712905a2de91f'
    assert sha(checkpoint)==expected, 'accepted checkpoint drift'
    excluded={'MANIFEST_R19.json','SHA256SUMS'}
    files=sorted(p for p in root.rglob('*') if p.is_file() and
          p.name not in excluded and not p.name.endswith(('.pyc','.tmp','.zip')) and
          '.pal_preprocessed' not in p.parts and '__pycache__' not in p.parts)
    ck=[p for p in files if p.parent.name=='checkpoint']
    assert ck==[checkpoint],('multiple checkpoints',ck)
    meta={p.relative_to(root).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}
    report={'release':'PAL R19 semantic audit','status':'R18 accepted checkpoint unchanged; R19 semantic candidates rejected',
            'checkpoint':'checkpoint/r12_active.pal12','checkpoint_sha256':expected,
            'inherited_r12_updates':10756,'one_active_checkpoint':True,
            'new_model_weights':False,'new_optimizer_histories_promoted':False,
            'all_sources_and_reports_included':True,
            'cache_note':'Transient .pal_preprocessed NPZ cache excluded; raw split books and tokenizer included',
            'files':meta}
    manifest=root/'MANIFEST_R19.json';manifest.write_text(json.dumps(report,indent=2)+'\n')
    sums=root/'SHA256SUMS';sums.write_text(''.join(f"{info['sha256']}  {path}\n" for path,info in meta.items())+
                        f'{sha(manifest)}  MANIFEST_R19.json\n')
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for p in files+[manifest,sums]:z.write(p,p.relative_to(root).as_posix())
    with zipfile.ZipFile(out) as z:
        bad=z.testzip()
        assert bad is None,bad
        paths=set(z.namelist())
        assert paths==set(meta)|{'MANIFEST_R19.json','SHA256SUMS'}
        assert len([p for p in paths if p.startswith('checkpoint/')])==1
        assert 'source/test_r19.py' in paths and 'reports/r19_semantic_frontier_late.json' in paths
        # Stream check entire one single learned checkpoint exactly.
        h=hashlib.sha256()
        with z.open('checkpoint/r12_active.pal12') as f:
            for part in iter(lambda:f.read(1<<20),b''):h.update(part)
        assert h.hexdigest()==expected
    print(json.dumps({'archive':str(out),'bytes':out.stat().st_size,'sha256':sha(out),
                      'files':len(paths),'checkpoint_sha256':expected,'verified':True},indent=2))
if __name__=='__main__':main()
