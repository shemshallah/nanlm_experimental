#!/usr/bin/env python3
"""Create a minimal R28 delta that fully restores the unified R27->R28 model.

The unchanged 11.9M parent tensors and 144 inherited Adam states are NOT
repeated. Strict audit compares full model and optimizer after cold replay.
"""
import argparse,copy,hashlib,json,os,pathlib,torch
from r27_restore_unified import restore as restore_r27,sha
from r28_recurrent import MirrorRecurrentResidual


def exact(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(exact(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b


def make(root,source='reports/fa2_r28_unified_candidate.pth',output='reports/fa2_r28_adapter_overlay.pt'):
    root=pathlib.Path(root).resolve()
    r27_path=root/'reports/fa2_r27_unified_candidate.pth'
    full=torch.load(root/source,weights_only=False,map_location='cpu')
    if full.get('format')!='FA2_R28_UNIFIED_CANDIDATE_V1' or full.get('accepted_model_promoted'):
        raise ValueError('R28 candidate format')
    if full['parent_r27_sha256']!=sha(r27_path):raise ValueError('R27 parent mismatch')
    model,opt,_,_,_,parent=restore_r27(root)
    before=model.state_dict();oldopt=opt.state_dict()
    selected=full['selected']
    for k,v in before.items():
        if not torch.equal(v,selected['model'][k]):raise ValueError('R27 tensor mutation: '+k)
    newopt=selected['optimizer']
    if newopt['param_groups'][:12]!=oldopt['param_groups']:raise ValueError('R27 optimizer groups mutated')
    if not exact({k:newopt['state'][k] for k in oldopt['state']},oldopt['state']):
        raise ValueError('R27 optimizer moments mutated')
    if not exact(selected['carry'],parent['selected']['carry']):raise ValueError('R27 stream mutation')
    ids=newopt['param_groups'][-1]['params'];otherids={i for g in newopt['param_groups'][:-1] for i in g['params']}
    if any(i in otherids for i in ids) or len(ids)!=9:raise ValueError('R28 optimizer group ownership error')
    overlay={'format':'FA2_R28_ADAPTER_DELTA_V1','accepted_model_promoted':False,
        'parent_r27_sha256':sha(r27_path),'r28_step':selected['r28_step'],
        'r28_state':{k.removeprefix('r28.'):v for k,v in selected['model'].items() if k.startswith('r28.')},
        'r28_optimizer_group':newopt['param_groups'][-1],
        'r28_optimizer_states':{k:copy.deepcopy(newopt['state'][k]) for k in ids},
        'inherited_adam_states':len(oldopt['state']),
        'inherited_model_tensors':len(before),'inherited_stream_count':len(selected['carry']),
        'selection_dev':full['selection_dev']}
    path=root/output;temp=path.with_suffix('.tmp')
    torch.save(overlay,temp);os.replace(temp,path)
    from r28_restore_unified import restore
    restored,ro,_,_,_,_,_=restore(root,output)
    if not exact(restored.state_dict(),selected['model']):raise AssertionError('overlay model cold replay differs')
    if not exact(ro.state_dict(),newopt):raise AssertionError('overlay Adam cold replay differs')
    report={'format':'FA2_R28_DELTA_EXACT_RESTORE_V1',
       'r27_sha256':sha(r27_path),'overlay_sha256':sha(path),
       'overlay_bytes':path.stat().st_size,'full_r28_bytes':(root/source).stat().st_size,
       'optimizer_groups':len(ro.param_groups),'optimizer_states':len(ro.state),
       'all_model_tensors_exact':len(restored.state_dict()),
       'restored_step':selected['r28_step'],'status':'candidate_not_promoted'}
    (root/'reports/r28_overlay_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('R28_DELTA_RESTORE_PASS',json.dumps(report))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--source',default='reports/fa2_r28_unified_candidate.pth');p.add_argument('--output',default='reports/fa2_r28_adapter_overlay.pt');a=p.parse_args()
    make(a.root,a.source,a.output)
