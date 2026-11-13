#!/usr/bin/env python3
"""FA1 R7.22 real 1M CPU addressability, identity and causal range audit."""
import argparse,json,pathlib,time
import numpy as np
import torch
from fa1_r721_unified import load_candidate
from fa1_r722_unified import load_unified
from r710_checkpoint_integrity import sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='/mnt/data/fa1_full')
    p.add_argument('--ancestor',default='/mnt/data/fa1r721/checkpoint/r7_unified_candidate.pt')
    p.add_argument('--checkpoint',default='/mnt/data/fa1r722/checkpoint/r7_unified_candidate.pt')
    p.add_argument('--positions',type=int,default=64);p.add_argument('--start',type=int,default=1350000)
    p.add_argument('--output',default='/mnt/data/fa1r722/reports/R722_REAL_1M_MEMORY.json')
    a=p.parse_args();torch.set_num_threads(2);begin=time.monotonic()
    par=sha256(a.ancestor)
    old,o,base,comp,_=load_candidate(a.ancestor,a.root)
    model,opts,newbase,newcomp,meta=load_unified(a.checkpoint,a.root)
    ps=np.arange(a.start,a.start+a.positions,dtype=np.int64)
    ref=old.runtime(a.root,base,comp,split='train').score_positions(ps,mode='packed_index',batch_size=16)
    rt=model.runtime(a.root,newbase,newcomp,split='train')
    got=rt.score_positions(ps,mode='packed_index',batch_size=16,return_details=True,probe_memory=True)
    errs={k:max(abs(x-y) for x,y in zip(ref[k],got[k])) for k in ('baseline_nll','near_nll','far_nll')}
    assert max(errs.values())==0.
    assert got['long_semantic_nll']==got['far_nll']
    assert got['semantic_nll']==got['far_nll']
    assert all(x==0. for x in got['semantic_gate'])
    assert all(x==0. for x in got['long_semantic_gate'])
    hits=got['long_semantic_eligible_positions']
    assert hits>0,'1M old tier did not retrieve any real training history'
    # Rebuild at last position using only the historical stream, with strict
    # exclusion of targets and no cross-document access.
    idx=rt.long_index;rec=idx.query(target=int(rt.corpus.seq[int(ps[-1])]))
    assert all(int(q)+1<int(ps[-1])-131072 for q in rec.get('positions',()))
    assert all(int(q)>=int(ps[-1])-1048576 for q in rec.get('positions',()))
    assert sha256(a.ancestor)==par
    out={'round':'FA1 R7.22 unified 1M memory','split':'historical train',
         'positions':len(ps),'first':int(ps[0]),'last':int(ps[-1]),
         'full_model_parity_errors':errs,
         'short_semantic_eligible':got['semantic_eligible_positions'],
         'long_semantic_eligible':hits,
         'new_gates_disabled_exactly':True,
         'source_1M_memory_causal':True,
         'selected_updates':0,'independent_books':False,'promoted':False,
         'elapsed_seconds':round(time.monotonic()-begin,3)}
    pathlib.Path(a.output).write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()