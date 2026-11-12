#!/usr/bin/env python3
"""Deterministic, minimal cumulative FA1 R7 source/reports ZIP (never embed weights)."""
from __future__ import annotations
import argparse,hashlib,json,pathlib,zipfile

def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha_file(p):return sha_bytes(p.read_bytes())

def eligible(root):
    for p in root.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(root)
        if any(part in ('checkpoint','__pycache__','.git') for part in rel.parts):continue
        if p.suffix in ('.pt','.pth','.bin','.db','.sqlite','.gguf','.tmp','.pyc','.zip','.zst'):
            # Some historical files are corpus binary, not model source; exclude.
            continue
        if rel.name in ('MANIFEST.json','SHA256SUMS'):continue
        yield p

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1])
    ap.add_argument('--output',type=pathlib.Path,default=pathlib.Path('/mnt/data/FA1_R7p21_UNIFIED_CPU_SOURCE.zip'))
    a=ap.parse_args()
    root=a.root;files=sorted(eligible(root))
    assets={p.relative_to(root).as_posix():{'size':p.stat().st_size,'sha256':sha_file(p)} for p in files}
    manifest={
        'architecture':'FA1 R7.21 ONE unified R7 model (master)',
        'accepted_step':14416,'promoted':False,'independent_books':False,
        'accepted_checkpoint_sha256':'54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d',
        'unified_candidate_sha256':'7d2ff0c17430075e3fc6ddeb02f61bf447c07eed027ae77afac0506737734b1f',
        'one_unified_model':True,'learned_parameter_tensors':95,'optimizer_histories':5,
        'legacy_test_positions_parity':1024,'max_parity_difference':0.0,
        'joint_adapter_updates_executed':96,'joint_adapter_updates_selected':0,
        'checkpoint_included':False,'assets':assets}
    (root/'MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    files.append(root/'MANIFEST.json')
    checksum=''.join(f'{sha_file(p)}  {p.relative_to(root).as_posix()}\n' for p in sorted(files))
    (root/'SHA256SUMS').write_text(checksum)
    files.append(root/'SHA256SUMS')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(files):
            rel=p.relative_to(root).as_posix()
            info=zipfile.ZipInfo(rel,date_time=(2026,11,12,12,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o644<<16
            z.writestr(info,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    with zipfile.ZipFile(a.output) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(files)
        assert len(set(z.namelist()))==len(files)
        for path,spec in assets.items():
            assert sha_bytes(z.read(path))==spec['sha256'],path
        assert not any(x.startswith('checkpoint/') or x.endswith('.pt') for x in z.namelist())
    print(json.dumps({'archive':str(a.output),'size_bytes':a.output.stat().st_size,
          'sha256':sha_file(a.output),'files':len(files),'verified':True,
          'contains_model_or_database':False,'one_unified_R7_checkpoint_separate':True},indent=2))
if __name__=='__main__':main()