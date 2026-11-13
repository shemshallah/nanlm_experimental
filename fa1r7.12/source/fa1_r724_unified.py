#!/usr/bin/env python3
"""FA1 R7.24 migration: one checkpoint, inherited R7.23 + causal S+/S- geometry.

Original R7.22 and R7.23 weights are never replaced, cloned, reinitialized,
or silently discarded. The fifth AdamW acquires exactly one additional group.
"""
from __future__ import annotations
import pathlib,tempfile,os,torch
from fa1_r724_curvature import CurvatureDiamond
from fa1_r723_diamond import load_diamond,CombinatorialDiamond,UnifiedR723Diamond
from fa1_r722_unified import UnifiedR722
from fa1_r721_unified import blank
from r710_checkpoint_integrity import sha256,deep_equal

FORMAT='FA1_R7_24_CURVED_DIAMOND_SINGLE_V1'

def extend_r723(seed_path,root,threads=2):
    checkpoint_sha=sha256(seed_path)
    m,opts,base,comp,obj=load_diamond(seed_path,root,threads=threads)
    assert len(opts)==5 and len(opts[4].param_groups)==8
    legacy_weights={k:v.clone() for k,v in m.state_dict().items()}
    legacy_states=[o.state_dict() for o in opts]
    m.diamond.geometry=CurvatureDiamond(width=m.diamond.d)
    opts[4].add_param_group({'params':list(m.diamond.geometry.parameters()),
                             'lr':5e-4,'weight_decay':0.})
    assert len(opts[4].param_groups)==9
    assert all(torch.equal(m.state_dict()[k],v) for k,v in legacy_weights.items())
    for i in range(4):assert deep_equal(opts[i].state_dict(),legacy_states[i])
    assert deep_equal(opts[4].state_dict()['state'],legacy_states[4]['state'])
    assert opts[4].state_dict()['param_groups'][:-1]==legacy_states[4]['param_groups']
    return m,opts,base,comp,{'seed_sha256':checkpoint_sha,
              'r722_ancestor_sha256':obj['r722_ancestor_sha256'],
              'r723_updates':obj['new_diamond_updates'],
              'accepted_step':14416}


def save_r724(path,model,opts,meta,steps,selection):
    if len(opts)!=5 or len(opts[4].param_groups)!=9:raise ValueError('optimizer lineage broken')
    data={'format':FORMAT,'accepted_step':14416,'r723_parent_sha256':meta['seed_sha256'],
          'r722_ancestor_sha256':meta['r722_ancestor_sha256'],
          'r723_inherited_updates':meta['r723_updates'],
          'r724_geometry_updates':int(steps),'optimizer_count':5,
          'promoted':False,'selection':selection,
          'topology':{'vertices':256,'edges':32640,'faces':2763520,
                      'tetrahedra_addressable':174792640,
                      'active_vertices':8,'active_edges':28,
                      'active_faces':56,'active_tetrahedra':70,
                      'mirror_channels':['S+','S-']},
          'model':model.state_dict(),
          'optimizer_states':[o.state_dict() for o in opts]}
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.r724_',suffix='.pending',dir=path.parent);os.close(fd)
    try:
        torch.save(data,tmp)
        os.replace(tmp,path)
    finally:
        if os.path.isfile(tmp):os.unlink(tmp)
    return {'sha256':sha256(path),'bytes':path.stat().st_size}


def load_r724(path,root,threads=2):
    obj=torch.load(path,map_location='cpu',weights_only=False)
    if obj.get('format')!=FORMAT or obj.get('accepted_step')!=14416 or obj.get('promoted') or obj.get('optimizer_count')!=5:
        raise ValueError('invalid R7.24 unified checkpoint lineage')
    old,opts,base,comp=blank(root,threads,embedded_state=obj['model'])
    old=UnifiedR722(old.core,old.calibration,old.near,old.far)
    opts[4].add_param_group({'params':list(old.semantic.parameters())+list(old.long_semantic.parameters()),
                             'lr':5e-4,'weight_decay':0.})
    model=UnifiedR723Diamond(old.core,old.calibration,old.near,old.far,
                             old.semantic,old.long_semantic,CombinatorialDiamond())
    opts[4].add_param_group({'params':list(model.diamond.parameters()),'lr':1e-3,'weight_decay':0.})
    model.diamond.geometry=CurvatureDiamond(width=model.diamond.d)
    opts[4].add_param_group({'params':list(model.diamond.geometry.parameters()),
                             'lr':5e-4,'weight_decay':0.})
    model.load_state_dict(obj['model'],strict=True)
    for o,state in zip(opts,obj['optimizer_states']):o.load_state_dict(state)
    assert len(opts)==5 and len(opts[4].param_groups)==9
    model.eval()
    return model,opts,base,comp,obj