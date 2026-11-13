#!/usr/bin/env python3
"""FA1 R7.22 CPU accounting: inherited vs gate-closed vs explicit 1M probe.

No model-quality claim. Two execution orders reduce misleading benchmark bias.
"""
import argparse,json,pathlib,time
import numpy as np,torch
from fa1_r721_unified import load_candidate
from fa1_r722_unified import load_unified

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='/mnt/data/fa1_full')
 p.add_argument('--ancestor',default='/mnt/data/fa1r721/checkpoint/r7_unified_candidate.pt')
 p.add_argument('--checkpoint',default='/mnt/data/fa1r722/checkpoint/r7_unified_candidate.pt')
 p.add_argument('--start',type=int,default=1350000);p.add_argument('--n',type=int,default=64)
 p.add_argument('--split',choices=['train','dev','test'],default='train')
 p.add_argument('--output',default='/mnt/data/fa1r722/reports/R722_1M_CPU_BENCHMARK.json')
 a=p.parse_args();torch.set_num_threads(2)
 old,_,base,comp,_=load_candidate(a.ancestor,a.root)
 cur,_,b2,c2,_=load_unified(a.checkpoint,a.root)
 pos=np.arange(a.start,a.start+a.n,dtype=np.int64)
 runs=[]
 for order in [('r721','r722_closed','r722_probe'),('r722_probe','r722_closed','r721')]:
  times={};results={}
  for name in order:
   obj=(old,base,comp) if name=='r721' else (cur,b2,c2)
   runtime=obj[0].runtime(a.root,obj[1],obj[2],split=a.split)
   t=time.perf_counter();results[name]=runtime.score_positions(pos,mode='packed_index',batch_size=16,
       **({'probe_memory':name=='r722_probe'} if name!='r721' else {}))
   times[name]=time.perf_counter()-t
  assert results['r721']['far_nll']==results['r722_closed']['long_semantic_nll']
  assert results['r721']['far_nll']==results['r722_probe']['long_semantic_nll']
  assert results['r722_closed']['semantic_indices_skipped_in_closed_gate']
  assert results['r722_probe']['long_semantic_eligible_positions']>0
  runs.append({'order':order,'runtime_s':times,
    'closed_factor_vs_r721':times['r722_closed']/times['r721'],
    'probe_factor_vs_r721':times['r722_probe']/times['r721']})
 report={'round':'FA1 R7.22','positions':a.n,'split':a.split,'start':a.start,
    'comparison':'full model R7.21 vs single unified R7.22 closed (lazy) and 1M probe',
    'runs':runs,'prediction_parity_exact':True,'newbook_results':False,
    'semantic_capability_gain_proven':False,'promoted':False}
 pathlib.Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()