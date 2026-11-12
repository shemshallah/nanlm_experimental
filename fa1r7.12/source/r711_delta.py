#!/usr/bin/env python3
"""Compact R7.11 experimental adapter overlay: one accepted 91MB base checkpoint.

Extracts ONLY new directed writer parameters and optimizer moments from an
unpromoted full working candidate. Restores against pinned parent SHA exactly;
no second independent checkpoint or optimizer system is archived.
"""
import argparse,copy,hashlib,json,pathlib,os,tempfile
import torch,numpy as np
from r710_checkpoint_integrity import verify,deep_equal,sha256
from r711_selective_writer import restore,FMT
from r75_retrieval import OPT_NAMES
FORMAT='PAL_R7_11_DIRECTED_ADAPTER_OVERLAY_V1'

def create(root,candidate,output):
    root=pathlib.Path(root);source=root/'checkpoint/pal_active.pt'
    parent,pinfo=verify(source,root,expected_step=14416)
    child,cinfo=verify(candidate,root,expected_step=14416)
    if child['format']!=FMT or child['accepted_parent_sha256']!=pinfo['sha256']:
        raise ValueError('candidate has incorrect architecture or accepted parent')
    newkeys=sorted(set(child['model'])-set(parent['model']))
    if not newkeys or not all(k.startswith('r74.outer.dir_') for k in newkeys):raise ValueError('not directed writer-only delta')
    for k,v in parent['model'].items():
        if not torch.equal(child['model'][k],v):raise ValueError('legacy tensor was changed: '+k)
    for n in OPT_NAMES:
        old=parent['optimizers'][n];new=child['optimizers'][n]
        if n!='outer_adamw':
            if not deep_equal(old,new):raise ValueError('legacy optimizer changed: '+n)
        else:
            if not deep_equal(old['param_groups'],new['param_groups'][:-1]):raise ValueError('original outer param groups changed')
            if not all(k in new['state'] and deep_equal(v,new['state'][k]) for k,v in old['state'].items()):raise ValueError('original outer moments changed')
    last=child['optimizers']['outer_adamw']['param_groups'][-1]
    states={k:child['optimizers']['outer_adamw']['state'][k] for k in last['params'] if k in child['optimizers']['outer_adamw']['state']}
    delta={'format':FORMAT,'parent_sha256':pinfo['sha256'],'candidate_sha256':cinfo['sha256'],
      'joint_steps':14416,'directed_aux_updates':child.get('directed_aux_updates'),
      'new_model':{k:child['model'][k] for k in newkeys},
      'new_outer_param_group':last,'new_outer_optimizer_states':states,
      'torch_rng':child['torch_rng'],'numpy_rng':child['numpy_rng'],
      'python_rng':child['python_rng'],'batch_rng_state':child['batch_rng_state'],
      'auxiliary_rng_at_selected_step':child['metrics'].get('auxiliary_rng_at_selected_step'),
      'metrics':child['metrics']}
    output=pathlib.Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.r711_delta_',suffix='.pending',dir=output.parent)
    os.close(fd)
    try:
        torch.save(delta,temp)
        with open(temp,'rb') as f:os.fsync(f.fileno())
        os.replace(temp,output)
    finally:
        if os.path.exists(temp):os.unlink(temp)
    digest=sha256(output)
    output.with_suffix(output.suffix+'.sha256').write_text(digest+'  '+output.name+'\n')
    return {'parent_sha256':pinfo['sha256'],'candidate_sha256':cinfo['sha256'],
      'overlay_sha256':digest,'overlay_bytes':output.stat().st_size,
      'new_tensor_count':len(newkeys),'new_optimizer_moments':len(states),
      'original_tensors_and_five_optimizers_bitwise_unchanged':True}

def restore_overlay(root,overlay_path):
    root=pathlib.Path(root);overlay_path=pathlib.Path(overlay_path)
    expected=overlay_path.with_suffix(overlay_path.suffix+'.sha256')
    if not expected.exists() or sha256(overlay_path)!=expected.read_text().split()[0]:raise ValueError('overlay checksum mismatch')
    delta=torch.load(overlay_path,map_location='cpu',weights_only=False)
    if delta['format']!=FORMAT or sha256(root/'checkpoint/pal_active.pt')!=delta['parent_sha256']:
        raise ValueError('Wrong parent or overlay format')
    model,opts,base,comp,prior=restore(root,root/'checkpoint/pal_active.pt')
    sd=model.state_dict()
    if set(delta['new_model'])!={k for k in sd if k.startswith('r74.outer.dir_')}:
        raise ValueError('Overlay does not exactly match expected new parameters')
    for name,v in delta['new_model'].items():
        if v.shape!=sd[name].shape or not torch.isfinite(v).all():raise ValueError('Invalid tensor '+name)
        sd[name]=v
    model.load_state_dict(sd,strict=True)
    outerstate=opts[3].state_dict()
    if outerstate['param_groups'][-1]['params']!=delta['new_outer_param_group']['params']:
        raise ValueError('Overlay optimizer parameter identity mismatch')
    outerstate['param_groups'][-1]=delta['new_outer_param_group']
    outerstate['state'].update(delta['new_outer_optimizer_states'])
    opts[3].load_state_dict(outerstate)
    torch.set_rng_state(delta['torch_rng']);np.random.set_state(delta['numpy_rng']);__import__('random').setstate(delta['python_rng'])
    return model,opts,base,comp,delta

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    p.add_argument('--candidate',type=pathlib.Path)
    p.add_argument('--overlay',type=pathlib.Path,default=pathlib.Path('adaptation/r711_experimental_delta.pt'))
    a=p.parse_args()
    if a.candidate:print(json.dumps(create(a.root,a.candidate,a.overlay),indent=2))
    model,opts,base,comp,d=restore_overlay(a.root,a.overlay)
    print(json.dumps({'overlay_restored':True,'directed_updates':d['directed_aux_updates'],'model_params':len(model.state_dict()),'optimizers':len(opts)},indent=2))
