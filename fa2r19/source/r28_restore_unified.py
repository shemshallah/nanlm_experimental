#!/usr/bin/env python3
"""Cold restore unpromoted R28 from immutable R19 ancestor + R27 + R28.

Never silently substitutes R28 for the accepted R27 checkpoint. Legacy Adam
states and the original 12 document streams remain in the restored model.
"""
import argparse,json,pathlib,torch
from r27_restore_unified import restore as restore_r27,sha
from r28_recurrent import MirrorRecurrentResidual


def restore(root, checkpoint='reports/fa2_r28_unified_candidate.pth', threads=2):
    root=pathlib.Path(root).resolve()
    model,opt,comp,r6,meta,r27=restore_r27(root,threads=threads)
    data=torch.load(root/checkpoint,map_location='cpu',weights_only=False)
    if data.get('format')!='FA2_R28_UNIFIED_CANDIDATE_V1' or data.get('accepted_model_promoted'):
        raise ValueError('R28 cannot masquerade as accepted checkpoint')
    if data.get('parent_r27_sha256')!=sha(root/'reports/fa2_r27_unified_candidate.pth'):
        raise ValueError('R28 parent R27 changed')
    weights=data['selected']['model']
    rank=weights['r28.up.weight'].shape[0]//2
    dim=model.model.memory.subword_cell.hidden_size
    model.add_module('r28',MirrorRecurrentResidual(dim=dim,rank=rank))
    opt.add_param_group({'params':list(model.r28.parameters()),'lr':.001,'weight_decay':1e-5})
    model.load_state_dict(weights,strict=True)
    opt.load_state_dict(data['selected']['optimizer'])
    if len(opt.param_groups)!=13 or len(opt.state)!=153:
        raise ValueError('R28 optimizer incomplete')
    if len(data['selected']['carry'])!=12:
        raise ValueError('R27 stream snapshots missing')
    return model,opt,comp,r6,meta,r27,data

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--checkpoint',default='reports/fa2_r28_unified_candidate.pth');args=p.parse_args()
    m,o,_,_,_,base,data=restore(args.root,args.checkpoint)
    print('R28_COLD_RESTORE_PASS',json.dumps({'step':data['selected']['r28_step'],
        'optimizer_groups':len(o.param_groups),'adam_states':len(o.state),
        'original_streams':len(data['selected']['carry']),
        'parent_r27_step':base['selected']['r27_step']}))
