#!/usr/bin/env python3
"""Identity-preserving R7.23 -> R7.24 geometric migration, same five Adams."""
from __future__ import annotations
import copy,os,pathlib,tempfile
import torch
from fa1_r723_diamond import extend_r722,load_diamond,UnifiedR723Diamond
from fa1_r724_geometry import GeometricDiamond
from r710_checkpoint_integrity import sha256,deep_equal
from fa1_r721_unified import blank
from fa1_r722_unified import UnifiedR722

FORMAT='FA1_R7_24_ORIENTED_SIMPLICIAL_DIAMOND_V1'


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
    m.load_state_dict(obj['model'],strict=True)
    for o,s in zip(opts,obj['optimizer_states']):o.load_state_dict(s)
    assert len(opts)==5 and len(opts[4].param_groups)==8
    return m,opts,base,comp,obj