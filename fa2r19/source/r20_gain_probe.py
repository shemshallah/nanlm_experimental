#!/usr/bin/env python3
"""FA2.1 R20: read-only matched dev NLL sweep for causal soft admission.

This is a frozen-gain ablation, not trainable semantic retrieval. The accepted
checkpoint remains byte-identical. No held-out test is used for selection.
"""
import argparse,hashlib,json,math,pathlib,sys,time
import torch
from torch.nn import functional as F
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import r14_slim_base as pal
from r18_long_memory import R18LongMemory
from r18_joint_continue import integrated_logits
from r13_corpus_cache import read_books_cached
from r11_raw import batch,windows
from r20_soft_admission import soft_candidate_copy

@torch.no_grad()
def run(root,gains=(0.0,0.1,0.5,1.0),long_windows=8):
    root=pathlib.Path(root)
    parent=pal.sha(root/'checkpoint/r12_active.pal12')
    saved,r6,comp,model,bridge,adapter,opt,rng=pal.load_for_arm(root,root/'checkpoint/r12_active.pal12',True)
    net=R18LongMemory(model.memory.piece_emb.weight)
    net.load_state_dict(saved['r18_memory']);net.eval();model.eval();bridge.eval();adapter.eval()
    docs,_=read_books_cached(root/'nanrouter',comp,r6.vocab,0)
    choices=windows(docs,400,500,2000)['dev'][:long_windows]
    acc={str(x):{'nll':0.0,'other_nll':0.0,'tokens':0,'other_tokens':0,'bytes':0,'active':0} for x in gains}
    for pick in choices:
        ex,lc,targets,tags,rawbytes=batch(docs['dev'],[pick],400)
        n=int((targets[0]!=-100).sum());tokens=targets[0,:n]
        old=integrated_logits(model,bridge,adapter,net,ex,lc,tags,targets)[0,:n]
        mask=tokens!=32
        for x in gains:
            a=acc[str(x)]
            result,meta=soft_candidate_copy(net,old,tokens,torch.tensor(float(x)))
            ce=F.cross_entropy(result,tokens,reduction='none')
            a['nll']+=float(ce.sum());a['other_nll']+=float(ce[mask].sum())
            a['tokens']+=n;a['other_tokens']+=int(mask.sum());a['bytes']+=rawbytes;a['active']+=meta['enabled_positions']
    outcomes={}
    for g,a in acc.items():
        outcomes[g]={'bits_per_byte':a['nll']/a['bytes']/math.log(2),
                     'nonspace_nll':a['other_nll']/a['other_tokens'],
                     'tokens':a['tokens'],'bytes':a['bytes'],'enabled_queries':a['active']}
    assert pal.sha(root/'checkpoint/r12_active.pal12')==parent
    out={'checkpoint_sha256':parent,'task':'frozen-gain soft admission read-only dev ablation',
         'long_windows':len(choices),'gain_trials':outcomes,'promoted':False,
         'warnings':['No R20 weights trained; copied values only from older causal positions',
           'Historical development windows repeatedly used for experimental decisions',
           'Does not evaluate paraphrase accuracy, multi-hop reasoning, or external new books']}
    dest=root/'reports/fa2_r20_gain_probe.json';dest.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.')
    p.add_argument('--long-windows',type=int,default=8)
    p.add_argument('--threads',type=int,default=2)
    args=p.parse_args();torch.set_num_threads(args.threads);run(args.root,long_windows=args.long_windows)
