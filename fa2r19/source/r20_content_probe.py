#!/usr/bin/env python3
"""Read-only FA2.1 soft-admission content-value mask ablation."""
import argparse,json,math,pathlib,sys,torch
from torch.nn import functional as F
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import r14_slim_base as pal
from r18_long_memory import R18LongMemory
from r18_joint_continue import integrated_logits
from r13_corpus_cache import read_books_cached
from r11_raw import batch,windows
from r20_soft_admission import soft_candidate_copy

@torch.no_grad()
def run(root, count=8):
    root=pathlib.Path(root);ck=root/'checkpoint/r12_active.pal12';h=pal.sha(ck)
    saved,r6,comp,model,bridge,adapter,opt,rng=pal.load_for_arm(root,ck,True)
    net=R18LongMemory(model.memory.piece_emb.weight)
    net.load_state_dict(saved['r18_memory']);net.eval();model.eval();bridge.eval();adapter.eval()
    docs,_=read_books_cached(root/'nanrouter',comp,r6.vocab,0)
    choices=windows(docs,400,500,2000)['dev'][:count]
    configs=[(mode,g) for mode in ('all','content') for g in (0.,.05,.1,.2,.5,1.)]
    sums={f'{mode}:{g}':{'nats':0.,'other_nats':0.,'bytes':0,'other':0,'active':0} for mode,g in configs}
    for pick in choices:
        ex,lc,t,tag,bytes_=batch(docs['dev'],[pick],400)
        n=int((t[0]!=-100).sum());x=t[0,:n]
        logits=integrated_logits(model,bridge,adapter,net,ex,lc,tag,t)[0,:n]
        other=x!=32
        for mode,g in configs:
            out,meta=soft_candidate_copy(net,logits,x,torch.tensor(float(g)),content_only=(mode=='content'))
            ce=F.cross_entropy(out,x,reduction='none')
            z=sums[f'{mode}:{g}'];z['nats']+=float(ce.sum());z['other_nats']+=float(ce[other].sum());z['bytes']+=bytes_;z['other']+=int(other.sum());z['active']+=meta['enabled_positions']
    res={k:{'bpb':v['nats']/v['bytes']/math.log(2),'nonspace_nll':v['other_nats']/v['other'],'enabled_queries':v['active']} for k,v in sums.items()}
    assert h==pal.sha(ck)
    report={'checkpoint_sha256':h,'type':'R20 read-only nonspace-value admission ablation','positions':len(choices),'outcomes':res,'promoted':False,'warning':'Historical development data; no training or semantic generalization claims'}
    (root/'reports/r20_content_probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.')
    p.add_argument('--windows',type=int,default=8);args=p.parse_args()
    torch.set_num_threads(2);run(args.root,args.windows)
