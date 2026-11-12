#!/usr/bin/env python3
"""Reproducibly package latest FA1 R7.19 source/research, NO full checkpoint."""
import argparse, hashlib, json, pathlib, zipfile

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1]);p.add_argument('--out',type=pathlib.Path,default=pathlib.Path('FA1_R7p19_CPU_REPRODUCIBILITY.zip'));a=p.parse_args()
    root=a.root
    all_files=sorted(f for f in root.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix not in ('.pyc','.tmp') and f.name not in ('MANIFEST.json','SHA256SUMS'))
    assets={f.relative_to(root).as_posix():{'sha256':sha(f),'size':f.stat().st_size} for f in all_files}
    manifest={'release':'FA1 R7.19 incremental 128K weighted suffix index','status':'unpromoted experimental CPU acceleration','accepted_step':14416,'inherited_optimizer_histories':5,
       'checkpoint_sha256':'54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d',
       'parent_full_archive_sha256':'99f25e5ff604492c6c043335eddbf638415a423f43d94f5c9a426f5a283a2728',
       'r716_adapter_sha256':'436ff87fae69ef1f44b53e2ffde92313da408342f65a1897c6eaff1876f9c09c',
       'r718_adapter_sha256':'f4d9175ff202d9ec1ff3a06dcddfd3d262cea25f8441fa5dab972a313309aa9f',
       'parity_tests':{'r719':11,'r718':12,'real_historical_positions':4},'independent_new_books_test':False,'promoted':False,'assets':assets}
    (root/'MANIFEST.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    all_files=sorted(f for f in root.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix not in ('.pyc','.tmp') and f.name!='SHA256SUMS')
    (root/'SHA256SUMS').write_text(''.join(f'{sha(f)}  {f.relative_to(root).as_posix()}\n' for f in all_files))
    all_files.append(root/'SHA256SUMS')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for f in sorted(all_files):
            info=zipfile.ZipInfo(f.relative_to(root).as_posix(),date_time=(2026,11,12,12,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
            z.writestr(info,f.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    with zipfile.ZipFile(a.out) as z:
        assert z.testzip() is None
        assert sorted(z.namelist())==sorted(f.relative_to(root).as_posix() for f in all_files)
        assert all(hashlib.sha256(z.read(path)).hexdigest()==spec['sha256'] for path,spec in assets.items())
        assert len(z.read('MANIFEST.json'))>0
    print(json.dumps({'archive':str(a.out),'files':len(all_files),'bytes':a.out.stat().st_size,'sha256':sha(a.out),'no_full_checkpoint':True,'verified':True},indent=2))
if __name__=='__main__':main()