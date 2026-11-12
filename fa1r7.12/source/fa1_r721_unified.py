#!/usr/bin/env python3
"""FA1 R7.21 unified architecture: one model, one candidate, five optimizer histories.

The *same* R7.7 trunk, R7.11 directed writer, R7.12 calibration state,
R7.16 near router, and R7.18 far router become one nn.Module and one resumable
checkpoint. The fifth inherited optimizer gets its original near and far AdamW
moments as two additional parameter groups (no sixth optimizer).

This is an unpromoted R7 continuation of the inherited R7.13 step-14416 model,
not an independently trained replacement or a new branch. Packed 128K memory is
causal non-parametric state used by the R7.20 runtime.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, pathlib, random, tempfile
import numpy as np
import torch
from torch import nn
from fa1_r716_router import TrustRouter, nll as near_nll, PARENT
from fa1_r718_far_residual import FarResidual, loss_far
from fa1_r720_runtime import FA1R720Runtime, NEAR_SHA, FAR_SHA
from r712_validate import restore as restore_parent
from r712_lm_bridge import CalibrationGate
from r77_semantic import build_model, PALR77
from r7_core import R6, V
from r7_memory import TokenCompositor
from r73_joint import UnifiedPAL
from r74_stream import PALR74, build_optimizers
from r711_selective_writer import upgrade
from r75_retrieval import OPT_NAMES
from r710_checkpoint_integrity import sha256, deep_equal

FORMAT='FA1_R7_21_UNIFIED_R7_CANDIDATE_V1'
STEP=14416


class UnifiedR7(nn.Module):
    """All learned R7 modules share this one nn.Module registration hierarchy."""
    def __init__(self,core,calibration,near,far):
        super().__init__()
        self.core=core
        self.calibration=calibration    # R7.12 historical path, disabled in R7.20 parity
        self.near=near
        self.far=far

    def joint_nll(self,data,near_enabled=True,far_enabled=True):
        """Differentiable, joint near/far mixture; the input target is used in CE only.

        Inputs are target-free features/counts plus target probabilities from
        causal history, exactly the same semantics as R7.20.
        """
        if near_enabled:
            close,_=near_nll(self.near,{
                'base':data['base'], 'features':data['near_features'],
                'prob':data['near_prob'],'count':data['near_count']})
        else:
            close=data['base']
        if far_enabled:
            final,_=loss_far(self.far,{
                'near':close,'far_features':data['far_features'],
                'far_prob':data['far_prob'],'far_count':data['far_count']})
        else:
            final=close
        return final,close

    def runtime(self,root,base,comp,split='test',corpus_dir=None):
        return FA1R720Runtime(root,self.core,base,comp,self.near,self.far,
                             split=split,corpus_dir=corpus_dir)


def _load_gate_states(root,near_path,far_path):
    root=pathlib.Path(root)
    if sha256(root/'checkpoint/pal_active.pt')!=PARENT:
        raise ValueError('Accepted parent SHA-256 mismatch')
    if sha256(near_path)!=NEAR_SHA or sha256(far_path)!=FAR_SHA:
        raise ValueError('R7 near/far source digest mismatch')
    near_data=torch.load(near_path,map_location='cpu',weights_only=False)
    far_data=torch.load(far_path,map_location='cpu',weights_only=False)
    if near_data['parent_sha256']!=PARENT or far_data['parent_sha256']!=PARENT:
        raise ValueError('R7 overlay parent mismatch')
    if far_data['r716_overlay_sha256']!=NEAR_SHA:
        raise ValueError('R7 far overlay not descended from near')
    if near_data['parent_joint_steps']!=STEP or far_data['step']!=48:
        raise ValueError('unexpected overlay training lineage')
    return near_data,far_data


def _append_moments(opts,near,far,near_data,far_data):
    """Map saved optimizer parameter IDs to the actual R7 fifth optimizer.

    Preserves every inherited optimizer param-group and state, including the
    original R7.12 calibration optimizer moments and all other R7 Adam slots.
    """
    if len(opts)!=5:raise ValueError('must restore five R7 optimizer histories')
    opt=opts[4]
    original=copy.deepcopy(opt.state_dict())
    old_groups=len(original['param_groups'])
    old_ids=set(original['state'])
    ng=near_data['optimizer_group']
    opt.add_param_group({'params':list(near.parameters()),
                         **{k:copy.deepcopy(v) for k,v in ng.items() if k!='params'}})
    merged=opt.state_dict()
    near_ids=merged['param_groups'][-1]['params']
    if near_ids!=ng['params']:
        raise AssertionError('near parameter IDs cannot be mapped losslessly')
    merged['param_groups'][-1]=copy.deepcopy(ng)
    if set(near_ids)!=set(near_data['optimizer_new_state']):
        raise AssertionError('near optimizer moment coverage incomplete')
    merged['state'].update(copy.deepcopy(near_data['optimizer_new_state']))
    opt.load_state_dict(merged)

    fg=far_data['far_optimizer']['param_groups']
    if len(fg)!=1:raise ValueError('far optimizer must have one existing group')
    opt.add_param_group({'params':list(far.parameters()),
                         **{k:copy.deepcopy(v) for k,v in fg[0].items() if k!='params'}})
    merged=opt.state_dict()
    far_ids=merged['param_groups'][-1]['params']
    far_original_ids=fg[0]['params']
    if len(far_ids)!=len(far_original_ids):raise AssertionError('far optimizer count mismatch')
    if set(far_original_ids)!=set(far_data['far_optimizer']['state']):
        raise AssertionError('far optimizer moment coverage incomplete')
    merged['state'].update({new_id:copy.deepcopy(far_data['far_optimizer']['state'][old_id])
                            for old_id,new_id in zip(far_original_ids,far_ids)})
    opt.load_state_dict(merged)
    final=opt.state_dict()
    assert len(final['param_groups'])==old_groups+2
    assert deep_equal(final['param_groups'][:old_groups],original['param_groups'])
    assert all(k in final['state'] and deep_equal(final['state'][k],v) for k,v in original['state'].items())
    assert not (set(near_ids)&set(far_ids))
    assert set(original['state'])==old_ids
    return {'legacy_fifth_groups':old_groups,'fifth_groups_after':len(final['param_groups']),
            'near_adam_slots':len(near_ids),'far_adam_slots':len(far_ids),
            'all_inherited_fifth_optimizer_slots_bitwise_preserved':True}


def bootstrap(root,near_path,far_path,threads=2):
    root=pathlib.Path(root)
    near_data,far_data=_load_gate_states(root,near_path,far_path)
    torch.set_num_threads(int(threads))
    core,opts,base,comp,calibration,_=restore_parent(root,root/'adaptation/r712_gate_delta.pt')
    near=TrustRouter();near.load_state_dict(near_data['gate'],strict=True)
    far=FarResidual();far.load_state_dict(far_data['far_gate'],strict=True)
    merged=_append_moments(opts,near,far,near_data,far_data)
    unified=UnifiedR7(core,calibration,near,far)
    unified.eval()
    return unified,opts,base,comp,merged


def _embedded_r6(root,states):
    """Load R6 vocabulary/tree and word tensors directly from this R7.21 file.

    Does not open the legacy accepted parent checkpoint or any R7 overlays.
    The existing tree/tokenizer artifacts are non-checkpoint corpus metadata.
    """
    root=pathlib.Path(root)
    prefix='core.r74.core.trunk.'
    required=('E','H','P','M','Q','NOW','W','B','L','T')
    obj=R6.__new__(R6)
    for name in required:
        obj.__dict__[name]=states[prefix+name].detach().cpu().numpy().copy()
    corpus=root/'corpus'
    obj.heads=np.fromfile(corpus/'token_heads.u16',dtype='<u2').reshape(V,2).astype(np.int32)
    obj.children=np.fromfile(corpus/'tree.i32',dtype='<i4').reshape(V-1,2)
    obj.vocab=(corpus/'vocab.words.txt').read_text().splitlines()
    if len(obj.vocab)!=V:raise ValueError('R7 checkpoint vocabulary size mismatch')
    parent=np.full(V*2-1,-1,dtype=np.int32)
    bits=np.zeros(V*2-1,dtype=np.float32)
    for j,(left,right) in enumerate(obj.children):
        parent[left]=parent[right]=V+j;bits[right]=1
    paths=[];targets=[];maxlen=0
    for word in range(V):
        cur=word;nodes=[];b=[]
        while parent[cur]>=0:
            pnode=parent[cur];nodes.append(pnode-V);b.append(bits[cur]);cur=pnode
        nodes.reverse();b.reverse();paths.append(nodes);targets.append(b);maxlen=max(maxlen,len(nodes))
    obj.maxpath=maxlen
    obj.pnodes=np.zeros((V,maxlen),dtype=np.int64)
    obj.pbits=np.zeros((V,maxlen),dtype=np.float32)
    obj.pmask=np.zeros((V,maxlen),dtype=np.float32)
    for i,(n,b) in enumerate(zip(paths,targets)):
        obj.pnodes[i,:len(n)]=n;obj.pbits[i,:len(n)]=b;obj.pmask[i,:len(n)]=1
    obj.Wt=torch.tensor(obj.W.copy(),dtype=torch.float32)
    obj.Bt=torch.tensor(obj.B.copy(),dtype=torch.float32)
    obj.pnodest=torch.tensor(obj.pnodes)
    obj.pbitst=torch.tensor(obj.pbits)
    obj.pmaskt=torch.tensor(obj.pmask)
    return obj


def blank(root,threads=2,embedded_state=None):
    """Build the same topology; with embedded_state the old checkpoint is unnecessary."""
    root=pathlib.Path(root);torch.set_num_threads(int(threads))
    if embedded_state is None:
        core,opts,base,comp=build_model(root,torch.device('cpu'),(2e-6,6e-5,1e-4,8e-5,.002))
    else:
        base=_embedded_r6(root,embedded_state)
        comp=TokenCompositor(root,base)
        core=PALR77(PALR74(UnifiedPAL(base,comp,root))).to(torch.device('cpu'))
        retrieval_params=dict(core.retrieval.named_parameters())
        groups=[{'params':[p for n,p in retrieval_params.items() if not n.startswith(('pair.','semantic.'))]},
                {'params':list(core.retrieval.pair.parameters()),'weight_decay':0.0},
                {'params':list(core.retrieval.semantic.parameters()),'weight_decay':0.0}]
        opts=(*build_optimizers(core.r74,2e-6,6e-5,1e-4,8e-5),
              torch.optim.AdamW(groups,lr=.002,weight_decay=.0005))
    upgrade(core,opts)
    calibration=CalibrationGate()
    opts[4].add_param_group({'params':list(calibration.parameters()),'lr':.008,'weight_decay':0.0})
    near=TrustRouter();far=FarResidual()
    opts[4].add_param_group({'params':list(near.parameters()),'lr':.002,'weight_decay':0.0})
    opts[4].add_param_group({'params':list(far.parameters()),'lr':.003,'weight_decay':0.0})
    return UnifiedR7(core,calibration,near,far),opts,base,comp


def _safe_save(path,obj):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.r721_',suffix='.pending',dir=path.parent);os.close(fd)
    try:
        torch.save(obj,tmp)
        with open(tmp,'rb') as f:os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return sha256(path)


def write_candidate(path,model,optimizers,root,near_path,far_path,candidate_updates=0,selection=None):
    if len(optimizers)!=5:raise ValueError('five optimizer histories required')
    if sha256(pathlib.Path(root)/'checkpoint/pal_active.pt')!=PARENT:
        raise ValueError('accepted checkpoint changed before candidate save')
    obj={
      'format':FORMAT,'series':'R7','branch_policy':'single master R7 lineage',
      'accepted_core_sha256':PARENT,'accepted_joint_steps':STEP,
      'source_overlay_sha256':{'r716':NEAR_SHA,'r718':FAR_SHA},
      'model':model.state_dict(),'optimizer_names':list(OPT_NAMES),
      'optimizer_states':[opt.state_dict() for opt in optimizers],
      'optimizer_count':5,'candidate_joint_adapter_updates':int(candidate_updates),
      'selection':selection or {'status':'identity-preserving unified migration','independent_books':False},
      'promotion':False,'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),
      'python_rng':random.getstate(),
      'model_architecture':'R7.7 trunk + R7.11 directed + R7.12 calibration (inactive) + near 8K/32K + far 32K/128K packed reader',
      'calibration_enabled':False,'packed_index_in_checkpoint':False,
    }
    return {'path':str(path),'sha256':_safe_save(path,obj),'bytes':pathlib.Path(path).stat().st_size,
            'unified_model_parameter_tensors':len(obj['model']),
            'optimizer_histories':len(obj['optimizer_states']),
            'candidate_adapter_updates':candidate_updates,'promoted':False}


def load_candidate(path,root,threads=2):
    obj=torch.load(path,map_location='cpu',weights_only=False)
    if obj.get('format')!=FORMAT or obj.get('series')!='R7':
        raise ValueError('wrong R7 candidate format')
    if obj['accepted_core_sha256']!=PARENT or obj['accepted_joint_steps']!=STEP:
        raise ValueError('wrong accepted R7 ancestry')
    if obj['optimizer_count']!=5 or obj['optimizer_names']!=list(OPT_NAMES):
        raise ValueError('missing/reordered R7 optimizer histories')
    if obj['source_overlay_sha256']!={'r716':NEAR_SHA,'r718':FAR_SHA}:
        raise ValueError('source lineage mismatch')
    model,opts,base,comp=blank(root,threads,embedded_state=obj['model'])
    model.load_state_dict(obj['model'],strict=True)
    for opt,state in zip(opts,obj['optimizer_states']):opt.load_state_dict(state)
    if len(opts[4].param_groups)!=6:raise AssertionError('near/far histories not loaded')
    model.eval()
    return model,opts,base,comp,obj


def trainable_memory_only(model):
    """Joint near+far training within the *same* R7 architecture and optimizer.

    Frozen inherited trunk/calibration protects the accepted step-14416 state;
    the existing fifth AdamW optimizer updates near and far together.
    A wider unfreeze requires independently validated books.
    """
    for p in model.core.parameters():p.requires_grad_(False)
    for p in model.calibration.parameters():p.requires_grad_(False)
    for p in model.near.parameters():p.requires_grad_(True)
    for p in model.far.parameters():p.requires_grad_(True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='/mnt/data/fa1_full')
    ap.add_argument('--near',default='/mnt/data/fa1r720/adaptation/fa1_r716_225.pt')
    ap.add_argument('--far',default='/mnt/data/fa1r720/adaptation/fa1_r718_full_candidate.pt')
    ap.add_argument('--output',default='/mnt/data/fa1r721/checkpoint/r7_unified_candidate.pt')
    ap.add_argument('--mode',choices=['bootstrap','load'],default='bootstrap')
    args=ap.parse_args()
    if args.mode=='bootstrap':
        model,opts,base,comp,info=bootstrap(args.root,args.near,args.far)
        report=write_candidate(args.output,model,opts,args.root,args.near,args.far)
        report['migration']=info
    else:
        model,opts,base,comp,blob=load_candidate(args.output,args.root)
        report={'loaded':True,'candidate_updates':blob['candidate_joint_adapter_updates'],
                'parameter_tensors':len(model.state_dict()),'optimizer_histories':len(opts)}
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()