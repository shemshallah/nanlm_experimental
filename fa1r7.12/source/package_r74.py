#!/usr/bin/env python3
"""Reproducible single-lineage R7.4 ZIP with checksums and no nested archives."""
import hashlib,json,pathlib,sys,zipfile
import torch

SOURCES=['r7_core.py','r7_memory.py','r73_joint.py','r74_stream.py','train_r7.py',
         'build_lossless_subwords.py','infer_r74.py','infer_r73.py',
         'infer_unified_now.py','infer_longctx2048.py','infer_temporal_subword.py',
         'eval_r74.py','probe_retrieval.py','test_r74.py',
         'prepare_document_corpus.py','fetch_nanrouter_release.py','package_r74.py']
CORPUS=['tree.i32','token_heads.u16','token_counts.i64','vocab.words.txt','vocab.json']
for split in ('train','dev','test'):
    CORPUS += [f'{split}_temporal.i32',f'{split}_longctx2048.i8',f'{split}_char_offsets.u32',f'{split}_char_blob.bin']
REPORTS=['R7p4_crosswindow.json','R7p4_matched.json','R7p4_retrieval_opportunity.json',
         'R7p4_generation.json','R7p4_regressions.json','R7p4_train40.log','R7p4_train800.log',
         'R7p3_reference_matched.json','R7p3_reference_joint.json']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def package(root,out):
    root=pathlib.Path(root);out=pathlib.Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    names=[f'source/{x}' for x in SOURCES]
    names += [f'corpus/{x}' for x in CORPUS]
    names += ['tokenizer/gutenberg_r7.model','tokenizer/gutenberg_r7.vocab']
    names += [f'reports/{x}' for x in REPORTS]
    names += ['checkpoint/pal_active.pt','README.md','CLAIM_LEDGER.md','RELEASE_NOTES.md']
    assert len(names)==len(set(names)) and sum(n.startswith('checkpoint/') for n in names)==1
    payload=torch.load(root/'checkpoint/pal_active.pt',map_location='cpu',weights_only=False)
    assert payload['format']=='PAL_R7_4_CROSSWINDOW_V1' and payload['joint_steps']==3240
    assert set(payload['optimizers'])=={'legacy_accum','adapter_adamw','mirror_adamw','outer_adamw'}
    files=[]
    for name in names:
        path=root/name
        assert path.exists(),str(path)
        files.append({'path':name,'bytes':path.stat().st_size,'sha256':sha(path)})
    manifest={'project':'PAL','release':'R7p4_CROSSWINDOW_3240',
              'one_active_checkpoint':'checkpoint/pal_active.pt',
              'checkpoint_format':payload['format'],'joint_steps':payload['joint_steps'],
              'legacy_r6_steps':payload['base_legacy_steps'],
              'legacy_r7_adapter_steps':payload['adapter_legacy_steps'],
              'r73_joint_steps_before_migration':2400,
              'new_joint_steps_this_round':840,
              'historical_corpus_not_independent_document_holdout':True,
              'cuda_verified':False,
              'notes':'Outer-memory ablation not significant; no claim of learned long-context benefit.',
              'files':files}
    man=json.dumps(manifest,indent=2)+'\n'
    checks=''.join(f'{a["sha256"]}  {a["path"]}\n' for a in files)
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=2,allowZip64=True) as z:
        for item in files:z.write(root/item['path'],item['path'])
        z.writestr('MANIFEST.json',man)
        z.writestr('SHA256SUMS.txt',checks)
    with zipfile.ZipFile(out,'r') as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(files)+2
        assert [n for n in z.namelist() if n.startswith('checkpoint/')]==['checkpoint/pal_active.pt']
        assert not any(x.endswith(('.zip','.exe','.so','.o')) for x in z.namelist())
        for f in files:
            assert z.getinfo(f['path']).file_size==f['bytes']
        h=hashlib.sha256()
        with z.open('checkpoint/pal_active.pt') as s:
            for buf in iter(lambda:s.read(1024*1024),b''):h.update(buf)
        assert h.hexdigest()==next(x['sha256'] for x in files if x['path']=='checkpoint/pal_active.pt')
    print(json.dumps({'zip':str(out),'bytes':out.stat().st_size,'sha256':sha(out),
                      'archive_entries':len(files)+2,'one_checkpoint':True,
                      'checkpoint_sha256':sha(root/'checkpoint/pal_active.pt'),
                      'joint_steps':payload['joint_steps']},indent=2))

if __name__=='__main__':
    package(pathlib.Path(sys.argv[1]) if len(sys.argv)>1 else pathlib.Path('.'),
            pathlib.Path(sys.argv[2]) if len(sys.argv)>2 else pathlib.Path('/mnt/data/PAL_R7p4_CROSSWINDOW_3240_full.zip'))
