#!/usr/bin/env python3
"""R12 continuity, exact-logit migration and strict autoregressive causality."""
import argparse, hashlib, json, pathlib, sys
import torch
from r12_experiment import load_for_arm,run_window,evaluate
from r11_raw import read_books,windows,batch,eval_raw


def run(args):
    torch.set_num_threads(2)
    root=pathlib.Path(args.root);ck=pathlib.Path(args.checkpoint)
    saved,r6,comp,model,br,ad,opt,rg=load_for_arm(root,ck,True)
    docs,manifest=read_books(args.books,comp,r6.vocab,max_books=1)
    picks=windows(docs,12,8,128)
    ex,lc,target,tag,_=batch(docs['dev'],picks['dev'][:1],12)
    with torch.no_grad():
        h=model.core.hidden(ex.reshape(-1,7),lc.reshape(-1,72)).reshape(1,ex.shape[1],-1)
        k=min(12,int((target!=-100).sum())-3)
        before=run_window(model,br,ad,h,target,tag,inspect_prefix=k)
        altered=target.clone();altered[:,k:]=torch.where(altered[:,k:]!=-100,torch.full_like(altered[:,k:],41),altered[:,k:])
        after=run_window(model,br,ad,h,altered,tag,inspect_prefix=k)
        logits_error=max(float((x-y).abs().max()) for x,y in zip(before['prefix_logits'],after['prefix_logits']))
        # At exactly zero gates, the R12 architecture must match the original
        # R11 logits bit-for-bit. This is the critical reversible expansion.
        orig_gains=(ad.attn_gain.clone(),ad.ffn_gain.clone())
        ad.attn_gain.zero_();ad.ffn_gain.zero_()
        muted=run_window(model,br,ad,h,target,tag,inspect_prefix=k)
        previous=run_window(model,br,None,h,target,tag,inspect_prefix=k)
        migrated=max(float((x-y).abs().max()) for x,y in zip(muted['prefix_logits'],previous['prefix_logits']))
        ad.attn_gain.copy_(orig_gains[0]);ad.ffn_gain.copy_(orig_gains[1])
    states=list(saved['stream_optimizer']['state'].values())
    steps=[]
    for s in states:
        if 'step' in s:steps.append(float(s['step']))
    checks={
        'single_checkpoint_format':saved['format']=='PAL12_CAUSAL_RESIDUAL_V1',
        'r10_updates_preserved':saved['r10_updates']==2020,
        'r11_history_preserved':saved['r11_updates']>=130,
        'optimizer_groups_preserved_plus_one':len(opt.param_groups)==4,
        'optimizer_moments_nonempty':len(states)>=18 and max(steps)>=130,
        'mirror_rank64_preserved':tuple(model.core.P.shape)==(32640,64) and tuple(model.core.M.shape)==(32640,64),
        'zero_gate_exact_prior_logits':migrated==0.,
        'no_future_target_leakage':logits_error==0.,
        'same_decoder':model.memory.subword_out.weight.shape[1]==96,
        'nonzero_learned_residual':bool(abs(float(ad.ffn_gain))+abs(float(ad.attn_gain))>0),
        'manifest_is_checked':bool(all(x.raw_sha for split in docs.values() for x in split)),
    }
    report={'passed':sum(checks.values()),'total':len(checks),'checks':checks,
            'future_mutation_max_prefix_logit_error':logits_error,
            'zero_gate_migration_max_logit_error':migrated,
            'active_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),
            'r10_updates':saved['r10_updates'],'r11_updates':saved['r11_updates'],
            'r12_updates':saved.get('r12_updates')}
    assert all(checks.values()),report
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='.')
    p.add_argument('--books',default='nanrouter')
    p.add_argument('--checkpoint',default='checkpoint/r12_active.pal12')
    run(p.parse_args())
