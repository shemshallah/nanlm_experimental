#!/usr/bin/env python3
"""Create a self-verifying single-checkpoint complete PAL R7.7 release ZIP.

Only one accepted checkpoint, all original corpus/tokenizer/code/results and
negative experiment notes; no temporary/rejected/duplicated model archives.
"""
import hashlib,json,pathlib,zipfile,sys,os,datetime,torch

FORMAT='PAL_R7_7_SEMANTIC_ASSOCIATIVE_V1'
STEP=14416
CHECKPOINT_SHA='54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d'
OPTIMIZERS=['legacy_accum','adapter_adamw','mirror_adamw','outer_adamw','retrieval_adamw']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for data in iter(lambda:f.read(4<<20),b''):h.update(data)
    return h.hexdigest()

def main(root,out):
    root=pathlib.Path(root).resolve();out=pathlib.Path(out).resolve()
    ck=root/'checkpoint/pal_active.pt'
    saved=torch.load(ck,map_location='cpu',weights_only=False)
    assert saved['format']==FORMAT,saved['format']
    assert saved['joint_steps']==STEP,saved['joint_steps']
    assert list(saved['optimizers'])==OPTIMIZERS
    assert sha(ck)==CHECKPOINT_SHA,'checkpoint modified or wrong branch'
    assert any(x.startswith('retrieval.semantic.') for x in saved['model'])
    assert (root/'reports/R7p7_regression_final.json').exists()
    audit=[]
    for split in ('dev','test'):
        result=json.loads((root/f'reports/R7p7_step14416_{split}_historical_4096.json').read_text())
        assert result['joint_step']==STEP and result['n']==4096 and result['historical_only']
        assert result['metrics']['semantic_gain']>0 and result['metrics']['total_gain']>0
        audit.append(result)
    assert audit[0]['metrics']['active_nll']<5.15
    assert audit[1]['metrics']['active_nll']<5.8
    tests=json.loads((root/'reports/R7p7_regression_final.json').read_text())
    assert tests['passed']==12 and all(x=='PASS' for x in tests['checks'].values())
    keep_dirs=('source','corpus','tokenizer','reports','research')
    paths=[]
    for directory in keep_dirs:
        for f in sorted((root/directory).rglob('*')):
            if not f.is_file():continue
            if '__pycache__' in f.parts or f.suffix in ('.pyc','.tmp','.exit'):continue
            paths.append(f)
    paths.extend([ck,root/'README.md',root/'RELEASE_NOTES.md',root/'CLAIM_LEDGER.md'])
    assert len(paths)==len({f.relative_to(root).as_posix() for f in paths})
    assert len([f for f in paths if f.suffix in ('.pt','.pth','.ckpt','.pal7')])==1
    assert all('outeraux_rejected' not in str(f) for f in paths)
    manifest={}
    for f in paths:
        rel=f.relative_to(root).as_posix()
        manifest[rel]={'bytes':f.stat().st_size,'sha256':sha(f)}
    info={'name':'PAL_R7p7_SEMANTIC_ASSOCIATIVE_14416_FULL',
          'format':FORMAT,'joint_steps':STEP,
          'parent_checkpoint_sha256':'b48b405018024d5c3883351597b2ba8db4a50cd14df2ad1638fe584f6740f419',
          'checkpoint_sha256':CHECKPOINT_SHA,'optimizers':OPTIMIZERS,
          'active_checkpoint':'checkpoint/pal_active.pt','active_checkpoints_count':1,
          'historical_only_evaluation':True,'independent_natural_book_results_available':False,
          'rejected_outer_associative_auxiliary_trial_included_as_logs_only':True,
          'synthetic_external_book_data_are_smoke_tests_only':True,
          'report_4096':{a['split']:a['metrics'] for a in audit},
          'artifact_count_excluding_manifests':len(paths),'files':manifest}
    sums=''.join(f'{v["sha256"]}  {path}\n' for path,v in sorted(manifest.items()))
    for file,contents in [('MANIFEST.json',json.dumps(info,indent=2)+'\n'),('SHA256SUMS.txt',sums)]:
        (root/file).write_text(contents)
    out.parent.mkdir(parents=True,exist_ok=True);partial=out.with_suffix('.zip.tmp')
    with zipfile.ZipFile(partial,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4,allowZip64=True) as z:
        for f in paths:z.write(f,f.relative_to(root).as_posix())
        for name in ('MANIFEST.json','SHA256SUMS.txt'):z.write(root/name,name)
    with zipfile.ZipFile(partial) as z:
        assert z.testzip() is None,'bad CRC'
        members=z.namelist()
        assert len(members)==len(paths)+2 and len(set(members))==len(members)
        assert members.count('checkpoint/pal_active.pt')==1
        assert len([n for n in members if n.endswith('.pt')])==1
        assert z.read('MANIFEST.json')== (root/'MANIFEST.json').read_bytes()
    os.replace(partial,out)
    print(json.dumps({'path':str(out),'bytes':out.stat().st_size,'sha256':sha(out),
        'checkpoint_sha256':sha(ck),'step':saved['joint_steps'],
        'packaged_files':len(paths)+2,'zip_integrity':'PASS','regression_checks':12,
        'historical_test_nll':audit[1]['metrics']['active_nll']},indent=2))
if __name__=='__main__':
    main(sys.argv[1] if len(sys.argv)>1 else '.',sys.argv[2] if len(sys.argv)>2 else '/mnt/data/PAL_R7p7_SEMANTIC_ASSOCIATIVE_14416_FULL.zip')
