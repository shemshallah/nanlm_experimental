#!/usr/bin/env python3
"""Identity-preserving R7.23 -> R7.24 geometric migration, same five Adams."""
from __future__ import annotations
import copy,math,os,pathlib,tempfile
import torch
from fa1_r723_diamond import extend_r722,load_diamond,UnifiedR723Diamond
from fa1_r724_geometry import GeometricDiamond
from r710_checkpoint_integrity import sha256,deep_equal
from fa1_r721_unified import blank
from fa1_r722_unified import UnifiedR722

FORMAT='FA1_R7_24_ORIENTED_SIMPLICIAL_DIAMOND_V1'


def _legacy_geometry_upgrade(obj,model,opts):
    """Lossless 9->11 triangle-input migration for pre-meta R7.24 candidates.

    Original triangle columns and their Adam moments remain byte-identical.
    The two spherical columns start at zero, while the newly introduced apex
    specialist gate remains exactly closed. No inherited R7 optimizer resets.
    """
    old_state=obj['model']
    face='diamond.geometry.face_net.0.weight'
    if 'diamond.geometry.meta_gate' in old_state:
        return old_state,obj['optimizer_states']
    if tuple(old_state[face].shape)!=(32,9):
        raise ValueError('unknown geometric checkpoint layout; refusing migration')
    new_state=dict(old_state)
    current_state=model.state_dict()
    def widen(value,target):
        if value.shape!=(32,9) or target.shape!=(32,11):
            raise ValueError('unexpected triangle projection shape')
        result=value.new_zeros(target.shape)
        result[:,:9].copy_(value)
        return result
    new_state[face]=widen(old_state[face],current_state[face])
    # Pre-meta R7.24 files lack the five tensors below. Their random
    # initializer used to depend on unrelated global RNG state, so loading
    # the SAME candidate twice produced different model weights. Reconstruct
    # them deterministically without touching any saved learned parameter.
    rng=torch.Generator(device='cpu').manual_seed(724027)
    def missing_meta(name,reference):
        value=reference.new_empty(reference.shape)
        if name.endswith(('meta_up','meta_down')):
            return value.normal_(mean=0.,std=.02,generator=rng)
        if name.endswith('meta_gate'):
            return value.zero_()
        if name.endswith('tetra_meta.weight'):
            return torch.nn.init.kaiming_uniform_(value,a=math.sqrt(5),generator=rng)
        if name.endswith('tetra_meta.bias'):
            return value.uniform_(-1/math.sqrt(32),1/math.sqrt(32),generator=rng)
        raise ValueError('unknown absent geometry tensor '+name)
    added=[]
    for name,value in current_state.items():
        if name not in new_state:
            if not name.startswith('diamond.geometry.'):
                raise ValueError('missing inherited tensor '+name)
            new_state[name]=missing_meta(name,value)
            added.append(name)
    if set(added)!={
        'diamond.geometry.meta_up','diamond.geometry.meta_down',
        'diamond.geometry.meta_gate','diamond.geometry.tetra_meta.weight',
        'diamond.geometry.tetra_meta.bias'}:
        raise ValueError('unrecognized legacy geometry parameter inventory')
    # Map the old final optimizer group by the exact ordered names shared
    # between old and new modules; inserting new meta parameters changes the
    # positional order and must NOT silently shift historical Adam moments.
    states=copy.deepcopy(obj['optimizer_states'])
    last=states[4]
    previous_ids=last['param_groups'][-1]['params']
    named=list(model.diamond.named_parameters())
    old_names=[name for name,_ in named if 'diamond.'+name in old_state]
    if len(previous_ids)!=len(old_names):
        raise ValueError('legacy Adam group does not match learned Diamond')
    mapping=dict(zip(old_names,previous_ids))
    next_id=max(i for group in last['param_groups'] for i in group['params'])+1
    remapped=[]
    for name,_ in named:
        if name in mapping:
            remapped.append(mapping[name])
        else:
            remapped.append(next_id)
            next_id+=1
    face_name='geometry.face_net.0.weight'
    face_id=mapping[face_name]
    moment=last['state'].get(face_id,{})
    for key,value in list(moment.items()):
        if torch.is_tensor(value) and tuple(value.shape)==(32,9):
            moment[key]=widen(value,current_state[face])
    last['param_groups'][-1]['params']=remapped
    obj['legacy_geometry_migration']={
        'from_face_features':9,'to_face_features':11,
        'missing_meta_seed':724027,
        'added_meta_parameters':sorted(added),
        'old_adam_ids_preserved':True,'accepted_parent_unchanged':True}
    return new_state,states


def _install(model,optimizers):
    old=model.diamond
    old_params=list(old.parameters())
    old_state=old.state_dict()
    geo=GeometricDiamond()
    missing,unexpected=geo.load_state_dict(old_state,strict=False)
    assert missing and all(x.startswith('geometry.') for x in missing) and not unexpected
    new_params=list(geo.parameters())
    assert len(new_params)>len(old_params)
    opt=optimizers[4]
    assert len(opt.param_groups)==8
    current=opt.param_groups[-1]['params']
    assert all(x is y for x,y in zip(current,old_params)) and len(current)==len(old_params)
    # Preserve every Adam moment by parameter name/order, including the old
    # apex's own learned optimizer state. Fresh geometric parameters have no
    # fake/initialized optimizer history until their first genuine update.
    existing=dict(opt.state)
    for p0,p1 in zip(old_params,new_params):
        assert torch.equal(p0,p1)
        if p0 in existing:
            opt.state[p1]=opt.state.pop(p0)
    opt.param_groups[-1]['params']=new_params
    model.diamond=geo
    for k,v in old_state.items():assert torch.equal(v,model.diamond.state_dict()[k])
    return model,optimizers


def migrate(parent,root,from_r723=False,threads=2):
    source_sha=sha256(parent)
    if from_r723:
        model,opts,base,comp,blob=load_diamond(parent,root,threads=threads)
    else:
        model,opts,base,comp,blob=extend_r722(parent,root,threads=threads)
    prev_states=[copy.deepcopy(o.state_dict()) for o in opts]
    non_diamond={k:v.detach().clone() for k,v in model.state_dict().items() if not k.startswith('diamond.')}
    model,opts=_install(model,opts)
    for i in range(4):assert deep_equal(opts[i].state_dict(),prev_states[i])
    for k,v in non_diamond.items():assert torch.equal(model.state_dict()[k],v)
    assert len(opts)==5 and sha256(parent)==source_sha
    return model,opts,base,comp,{'parent_sha256':source_sha,
       'source_format':'R7.23 candidate' if from_r723 else 'R7.22 accepted ancestry',
       'source_meta':blob}


def save_geometric(path,model,opts,parent_digest,update=0,notes=None):
    if len(opts)!=5 or len(opts[4].param_groups)!=8:
        raise ValueError('must have five inherited optimizers')
    obj={'format':FORMAT,'accepted_step':14416,'parent_sha256':parent_digest,
         'optimizer_count':5,'model':model.state_dict(),
         'optimizer_states':[o.state_dict() for o in opts],
         'new_geometry_updates':int(update),'promoted':False,
         'notes':notes or {},'points':256,'pair_count':32640,
         'face_count':2763520,'possible_tetrahedra':174792640,
         'active_tetrahedra_per_token':70}
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.r724_',suffix='.pending',dir=path.parent)
    os.close(fd)
    try:
        torch.save(obj,tmp)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return {'sha256':sha256(path),'bytes':path.stat().st_size}


def load_geometric(path,root,threads=2):
    from fa1_r722_semantic import SemanticGate
    from fa1_r723_diamond import CombinatorialDiamond
    obj=torch.load(path,map_location='cpu',weights_only=False)
    assert obj['format']==FORMAT and obj['optimizer_count']==5 and obj['accepted_step']==14416 and not obj['promoted']
    old,opts,base,comp=blank(root,threads,embedded_state=obj['model'])
    old=UnifiedR722(old.core,old.calibration,old.near,old.far)
    opts[4].add_param_group({'params':list(old.semantic.parameters())+list(old.long_semantic.parameters()),'lr':5e-4,'weight_decay':0.})
    m=UnifiedR723Diamond(old.core,old.calibration,old.near,old.far,old.semantic,old.long_semantic,GeometricDiamond())
    opts[4].add_param_group({'params':list(m.diamond.parameters()),'lr':1e-3,'weight_decay':0.})
    weights,states=_legacy_geometry_upgrade(obj,m,opts)
    m.load_state_dict(weights,strict=True)
    for o,s in zip(opts,states):o.load_state_dict(s)
    assert len(opts)==5 and len(opts[4].param_groups)==8
    return m,opts,base,comp,obj
