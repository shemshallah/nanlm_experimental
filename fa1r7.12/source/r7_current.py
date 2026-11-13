#!/usr/bin/env python3
"""Canonical FA1 R7 entrypoint: one R7.22 checkpoint, master only."""
import argparse,json,pathlib
import numpy as np
from fa1_r722_unified import load_unified

def main():
 p=argparse.ArgumentParser(description='FA1 single unified R7 prediction evaluator')
 p.add_argument('--root',type=pathlib.Path,required=True)
 p.add_argument('--checkpoint',type=pathlib.Path,required=True)
 p.add_argument('--split',choices=['train','dev','test'],default='dev')
 p.add_argument('--start',type=int,default=134000)
 p.add_argument('--positions',type=int,default=128)
 p.add_argument('--batch',type=int,default=16)
 p.add_argument('--mode',choices=['auto','packed_index','exact_reference'],default='auto')
 p.add_argument('--probe-memory',action='store_true',help='Inspect inactive 128K+1M associative indices')
 p.add_argument('--report',type=pathlib.Path)
 a=p.parse_args()
 if a.positions<=0:raise ValueError('positions must be positive')
 m,opts,base,comp,blob=load_unified(a.checkpoint,a.root)
 if blob['promotion'] or blob['accepted_step']!=14416:raise RuntimeError('invalid lineage')
 result=m.runtime(a.root,base,comp,split=a.split).score_positions(
   np.arange(a.start,a.start+a.positions,dtype=np.int64),mode=a.mode,
   batch_size=a.batch,probe_memory=a.probe_memory)
 d={'round':'FA1 R7.22 one unified architecture','git_branch':'master',
    'accepted_parent_step':blob['accepted_step'],'semantic_updates':blob['semantic_updates'],
    'model_tensors':len(m.state_dict()),'optimizers':len(opts),
    'split':a.split,'positions':a.positions,'probe_memory':a.probe_memory,
    'baseline_nll':float(np.mean(result['baseline_nll'])),
    'near_nll':float(np.mean(result['near_nll'])),
    'far_nll':float(np.mean(result['far_nll'])),
    'associative_128k_nll':float(np.mean(result['semantic_nll'])),
    'associative_1m_nll':float(np.mean(result['long_semantic_nll'])),
    'semantic_128k_eligible_positions':result['semantic_eligible_positions'],
    'semantic_1m_eligible_positions':result['long_semantic_eligible_positions'],
    'promotion':False,'independent_new_books':False}
 if a.report:
  a.report.parent.mkdir(parents=True,exist_ok=True)
  a.report.write_text(json.dumps(d,indent=2)+'\n')
 print(json.dumps(d,indent=2))
if __name__=='__main__':main()