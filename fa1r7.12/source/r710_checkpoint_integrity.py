#!/usr/bin/env python3
"""Durable, fail-closed PAL R7 checkpoint audit / candidate writer.

Research checkpoints are trusted PyTorch pickle files, not safe inputs from
unknown parties.  NEVER overwrite pal_active.pt for an unpromoted experiment.
All five optimizer histories, model tensors, RNG, step and source identity
travel as one atomic payload.  SHA sidecar verifies exact committed bytes.
"""
import argparse,hashlib,json,os,pathlib,tempfile,time
import numpy as np
import torch

OPT_NAMES=('legacy_accum','adapter_adamw','mirror_adamw','outer_adamw','retrieval_adamw')
ACCEPTED_FORMAT='PAL_R7_7_SEMANTIC_ASSOCIATIVE_V1'

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def validate(data,root,expected_step=None):
    if data.get('format') not in (ACCEPTED_FORMAT,'PAL_R7_11_SELECTIVE_DIRECTED_CANDIDATE_V1'):raise ValueError('Unexpected checkpoint format')
    if not isinstance(data.get('joint_steps'),int) or data['joint_steps']<1:raise ValueError('Invalid step')
    if expected_step is not None and data['joint_steps']!=expected_step:raise ValueError('Wrong joint step')
    for key,filename in [('tokenizer_sha256','tokenizer/gutenberg_r7.model'),
                          ('train_split_sha256','corpus/train_temporal.i32')]:
        if sha256(pathlib.Path(root)/filename)!=data.get(key):raise ValueError('Immutable input mismatch '+key)
    names=tuple(data['optimizers'])
    if set(names)!=set(OPT_NAMES):raise ValueError('Exactly five optimizer histories required')
    tensors=data.get('model',{})
    if not tensors or any(not torch.is_tensor(v) or not torch.isfinite(v).all() for v in tensors.values()):
        raise ValueError('Invalid model tensor')
    numstates={}
    for name,opt in data['optimizers'].items():
        if not isinstance(opt,dict) or not opt.get('param_groups') or not opt.get('state'):
            raise ValueError('Missing optimizer history: '+name)
        numstates[name]=len(opt['state'])
        for state in opt['state'].values():
            if isinstance(state,dict):
                for v in state.values():
                    if torch.is_tensor(v) and not torch.isfinite(v).all():raise ValueError('Nonfinite optimizer '+name)
    for key in ('torch_rng','numpy_rng','python_rng','batch_rng_state'):
        if key not in data: raise ValueError('RNG history missing '+key)
    return {'format':data['format'],'step':data['joint_steps'],
            'model_tensors':len(tensors),'optimizer_histories':numstates,
            'rng_keys_verified':['torch_rng','numpy_rng','python_rng','batch_rng_state']}

def verify(path,root,expected_step=None):
    path=pathlib.Path(path)
    if path.with_suffix(path.suffix+'.sha256').exists():
        expected=path.with_suffix(path.suffix+'.sha256').read_text().strip().split()[0]
        if sha256(path)!=expected:raise ValueError('SHA-256 sidecar mismatch')
    data=torch.load(path,map_location='cpu',weights_only=False)
    summary=validate(data,root,expected_step)
    summary['sha256']=sha256(path)
    return data,summary

def durable_save(data,out,root,expected_parent_sha256,force=False):
    """Write candidate and hash separately with atomic fsync; never write active."""
    out=pathlib.Path(out)
    if out.name=='pal_active.pt':raise PermissionError('Candidate must not overwrite accepted active checkpoint')
    if out.exists() and not force:raise FileExistsError(out)
    validate(data,root)
    if not isinstance(expected_parent_sha256,str) or len(expected_parent_sha256)!=64:
        raise ValueError('Exact parent SHA-256 is mandatory')
    out.parent.mkdir(parents=True,exist_ok=True)
    # Preserve original payload metadata exactly, retaining Adam + RNG states.
    fd,temp=tempfile.mkstemp(prefix='.'+out.name+'.',suffix='.pending',dir=out.parent)
    os.close(fd)
    try:
        torch.save(data,temp)
        with open(temp,'rb') as f:os.fsync(f.fileno())
        hexdigest=sha256(temp)
        # An interrupted write leaves previous committed checkpoint unchanged.
        os.replace(temp,out)
        directoryfd=os.open(str(out.parent),os.O_RDONLY)
        try:os.fsync(directoryfd)
        finally:os.close(directoryfd)
        # Sidecar is a second atomic commit; old sidecars are never silently trusted.
        sidecar=out.with_suffix(out.suffix+'.sha256')
        fd,sidetemp=tempfile.mkstemp(prefix='.'+sidecar.name+'.',suffix='.pending',dir=out.parent)
        with os.fdopen(fd,'w') as f:
            f.write(f'{hexdigest}  {out.name}\n');f.flush();os.fsync(f.fileno())
        os.replace(sidetemp,sidecar)
        directoryfd=os.open(str(out.parent),os.O_RDONLY)
        try:os.fsync(directoryfd)
        finally:os.close(directoryfd)
        return {'candidate_path':str(out),'candidate_sha256':hexdigest,
                'parent_sha256':expected_parent_sha256,'step':data['joint_steps']}
    finally:
        if os.path.exists(temp):os.unlink(temp)

def deep_equal(x,y):
    if torch.is_tensor(x):return torch.is_tensor(y) and torch.equal(x,y)
    if isinstance(x,np.ndarray):return isinstance(y,np.ndarray) and np.array_equal(x,y)
    if isinstance(x,dict):return isinstance(y,dict) and x.keys()==y.keys() and all(deep_equal(x[k],y[k]) for k in x)
    if isinstance(x,(tuple,list)):return isinstance(y,type(x)) and len(x)==len(y) and all(deep_equal(a,b) for a,b in zip(x,y))
    return type(x) is type(y) and x==y

def selftest(root,accepted,scratch):
    t=time.monotonic();accepted=pathlib.Path(accepted)
    orig,info=verify(accepted,root,expected_step=14416)
    scratch=pathlib.Path(scratch);scratch.mkdir(parents=True,exist_ok=True)
    candidate=scratch/'candidate_roundtrip.pt'
    commit=durable_save(orig,candidate,root,info['sha256'],force=True)
    restored,restored_info=verify(candidate,root,expected_step=14416)
    comparisons={k:deep_equal(orig[k],restored[k]) for k in
                 ('model','optimizers','torch_rng','numpy_rng','python_rng','batch_rng_state')}
    if not all(comparisons.values()):raise RuntimeError('roundtrip changed state')
    # A bogus checksum MUST block loading (test on temporary sidecar only).
    sidecar=candidate.with_suffix(candidate.suffix+'.sha256')
    true_sidecar=sidecar.read_text()
    try:
        sidecar.write_text('0'*64+'  candidate_roundtrip.pt\n')
        try:verify(candidate,root)
        except ValueError:corrupt_rejected=True
        else:corrupt_rejected=False
    finally:sidecar.write_text(true_sidecar)
    if not corrupt_rejected:raise RuntimeError('corruption not detected')
    # Refuse active checkpoint writes before touching it.
    try:durable_save(orig,accepted,root,info['sha256'],force=True)
    except PermissionError:accepted_write_blocked=True
    else:accepted_write_blocked=False
    if not accepted_write_blocked:raise RuntimeError('Accepted overwrite not blocked')
    # Hash must remain exact after the negative write test.
    if sha256(accepted)!=info['sha256']:raise RuntimeError('accepted checkpoint modified')
    return {'accepted':info,'roundtrip':restored_info,'parent_sha256':commit['parent_sha256'],
            'state_identical':comparisons,'corrupted_sidecar_rejected':corrupt_rejected,
            'accepted_overwrite_blocked':accepted_write_blocked,'accepted_unchanged':True,
            'new_training_updates':0,'elapsed_seconds':round(time.monotonic()-t,2)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    p.add_argument('--accepted',type=pathlib.Path,default=pathlib.Path('checkpoint/pal_active.pt'))
    p.add_argument('--scratch',type=pathlib.Path,default=pathlib.Path('/tmp/pal_r710_ckpt_audit'))
    p.add_argument('--output',type=pathlib.Path)
    a=p.parse_args();r=selftest(a.root,a.accepted,a.scratch)
    print(json.dumps({k:v for k,v in r.items() if k not in ('accepted','roundtrip')},indent=2))
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(r,indent=2)+'\n')
