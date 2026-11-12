#!/usr/bin/env python3
"""Release one current complete PAL R7.6 artifact, no superseded checkpoint copies."""
import hashlib,json,pathlib,zipfile,datetime,sys,os
import torch

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(4*1024*1024),b''):h.update(part)
    return h.hexdigest()

def main(root,out):
    root=pathlib.Path(root);out=pathlib.Path(out)
    ck=root/'checkpoint/pal_active.pt'
    data=torch.load(ck,map_location='cpu',weights_only=False)
    assert data['format']=='PAL_R7_6_PAIR_ASSOCIATIVE_V1',data['format']
    assert data['joint_steps']==14000,data['joint_steps']
    assert len(data['optimizers'])==5
    assert 'retrieval.pair.strength' in data['model']
    results=json.loads((root/'reports/R7p6_10k_results.json').read_text())
    assert results['completed_updates']==10000
    assert results['end_joint_step']==14000
    assert results['checkpoint_sha256']==sha(ck)
    fnames=[]
    for d in ('source','corpus','tokenizer','reports'):
        for f in sorted((root/d).rglob('*')):
            if not f.is_file() or '__pycache__' in f.parts or f.name.endswith(('.tmp','.pyc')):continue
            if f.name.endswith('.exit'):continue
            fnames.append(f)
    fnames.extend([ck,root/'README.md',root/'RELEASE_NOTES.md',root/'CLAIM_LEDGER.md'])
    manifest={}
    for f in fnames:
        rel=f.relative_to(root).as_posix()
        manifest[rel]={'bytes':f.stat().st_size,'sha256':sha(f)}
    info={'name':'PAL_R7p6_PAIR_ASSOCIATIVE_14000_full','format':data['format'],'joint_steps':data['joint_steps'],
          'optimizer_names':list(data['optimizers'].keys()),'artifact_count':len(fnames),
          'one_active_checkpoint':'checkpoint/pal_active.pt',
          'checkpoint_sha256':manifest['checkpoint/pal_active.pt']['sha256'],'files':manifest}
    sums=''.join(f'{v["sha256"]}  {k}\n' for k,v in sorted(manifest.items()))
    out.parent.mkdir(exist_ok=True,parents=True);partial=out.with_suffix('.zip.tmp')
    with zipfile.ZipFile(partial,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4,allowZip64=True) as z:
        for i,f in enumerate(fnames):
            z.write(f,f.relative_to(root).as_posix())
        z.writestr('MANIFEST.json',json.dumps(info,indent=2)+'\n')
        z.writestr('SHA256SUMS.txt',sums)
    with zipfile.ZipFile(partial) as z:
        bad=z.testzip();assert bad is None,bad
        assert z.namelist().count('checkpoint/pal_active.pt')==1
    os.replace(partial,out)
    print(json.dumps({'path':str(out),'bytes':out.stat().st_size,
                      'sha256':sha(out),'checkpoint_sha256':sha(ck),'files':len(fnames),'step':data['joint_steps']},indent=2))
if __name__=='__main__':
    main(sys.argv[1] if len(sys.argv)>1 else '.',sys.argv[2] if len(sys.argv)>2 else '/mnt/data/PAL_R7p6_PAIR_ASSOCIATIVE_14000_full.zip')
