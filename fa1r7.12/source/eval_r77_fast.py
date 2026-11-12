#!/usr/bin/env python3
"""Fast, exact (up to fp reduction order) R7.7 ablation audit over fixed positions.

Runs the expensive R7.4 trunk once per batch, not five times. Older-memory
state and history are shared by all ablations, and never include target tokens.
"""
import argparse,json,pathlib,time
import torch,numpy as np
from r77_semantic import restore
from r74_stream import OUTER_CONTEXT,OLDER
from r7_core import Corpus
from train_r7 import batch_data

@torch.no_grad()
def evaluate_positions(model,base,comp,corpus,positions,batch=8):
    model.eval();device=next(model.parameters()).device; n=len(positions)
    keys=('active','semantic_off','pair_off','one_word_only','retrieval_off','outer_off')
    sums={k:0. for k in keys};sums.update(n=n,semantic_coverage=0,semantic_target_hits=0,exact_key_coverage=0,pair_coverage=0)
    for index in np.array_split(positions,int(np.ceil(n/batch))):
        _,ctx,pc,ln,y,_,_=batch_data(base,corpus,comp,index)
        history=torch.as_tensor(corpus.get(index,OUTER_CONTEXT)[0][:,:OLDER].copy(),device=device)
        state=model.initial(history)
        a=torch.as_tensor(corpus.data[index].copy(),device=device)
        l=torch.as_tensor(corpus.long[index].copy(),device=device)
        ctx,pc,ln,y=(z.to(device) for z in (ctx,pc,ln,y))
        now,_=model.r74(a,l,ctx,pc,ln,state=state)
        base_loss=model.trunk.nll(now,y)
        q=ctx[:,-1]
        p,exists,feat,_=model.retrieval.lookup(history,q,y)
        g=model.retrieval.gate(now,q,feat,model.trunk.E,exists)
        one=model.retrieval.mixed_nll(base_loss,p,g)
        p2,e2,f2=model.pair_lookup(history,ctx,y)
        g2=model.pair_gate(now,ctx,f2,e2)
        both=model.retrieval.mixed_nll(one,p2,g2)
        ps,es,fs,_,_=model.retrieval.semantic.lookup(history,ctx,model.trunk.E,y)
        gs=model.retrieval.semantic.gate(now,ctx,fs,model.trunk.E,es)
        active=model.retrieval.mixed_nll(both,ps,gs)
        # Pair_off leaves semantic active. One_word_only switches off both new paths.
        pair_off=model.retrieval.mixed_nll(one,ps,gs)
        # Compute outer_off with fixed history but reduced NOW (all other paths still enabled).
        now0,_=model.r74(a,l,ctx,pc,ln,state=state,outer_off=True)
        b0=model.trunk.nll(now0,y)
        g0=model.retrieval.gate(now0,q,feat,model.trunk.E,exists)
        g20=model.pair_gate(now0,ctx,f2,e2)
        gs0=model.retrieval.semantic.gate(now0,ctx,fs,model.trunk.E,es)
        b0=model.retrieval.mixed_nll(b0,p,g0)
        b0=model.retrieval.mixed_nll(b0,p2,g20)
        outer=model.retrieval.mixed_nll(b0,ps,gs0)
        for k,loss in zip(keys,(active,both,pair_off,one,base_loss,outer)):
            sums[k]+=float(loss.sum())
        sums['semantic_coverage']+=int(es.sum());sums['semantic_target_hits']+=int((ps>0).sum())
        sums['exact_key_coverage']+=int(exists.sum());sums['pair_coverage']+=int(e2.sum())
    for k in keys:sums[k+'_nll']=sums.pop(k)/n
    sums['semantic_gain']=sums['semantic_off_nll']-sums['active_nll']
    sums['pair_gain']=sums['one_word_only_nll']-sums['semantic_off_nll']
    sums['exact_gain']=sums['retrieval_off_nll']-sums['one_word_only_nll']
    sums['total_gain']=sums['retrieval_off_nll']-sums['active_nll']
    sums['outer_gain']=sums['outer_off_nll']-sums['active_nll']
    return sums

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    p.add_argument('--split',choices=('dev','test'),default='test')
    p.add_argument('--count',type=int,default=4096)
    p.add_argument('--start',type=int,default=0,help='index within fixed count')
    p.add_argument('--size',type=int,default=128,help='shard size')
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--output',type=pathlib.Path)
    args=p.parse_args();torch.set_num_threads(args.threads)
    t=time.monotonic();m,o,b,c,d=restore(args.root,args.root/'checkpoint/pal_active.pt',torch.device('cpu'))
    dataset=Corpus(args.root,args.split)
    positions=(np.arange(args.count,dtype=np.int64)*(len(dataset.seq)-1)//args.count)[args.start:args.start+args.size]
    r={'split':args.split,'step':d['joint_steps'],'total_fixed_count':args.count,'shard_start':args.start,
       'shard_count':len(positions),'historical_only':True,'independent_books':False,
       'results':evaluate_positions(m,b,c,dataset,positions,batch=args.batch),
       'elapsed_seconds':round(time.monotonic()-t,2)}
    if args.output:args.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
if __name__=='__main__':main()
