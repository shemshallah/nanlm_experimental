#!/usr/bin/env python3
"""Canonical FA1 R7 entrypoint: one unified checkpoint, no overlay selection.

Use the packed causal R7 memory reader, with an automatic rescan fallback for
short/sparse sequences. The inherited R7 model is one registered nn.Module.
"""
import argparse,json,pathlib
import numpy as np
import torch
from fa1_r721_unified import load_candidate


def main():
    p=argparse.ArgumentParser(description='FA1 R7 unified CPU inference evaluator')
    p.add_argument('--root',type=pathlib.Path,required=True)
    p.add_argument('--checkpoint',type=pathlib.Path,required=True)
    p.add_argument('--split',choices=['train','dev','test'],default='test')
    p.add_argument('--start',type=int,default=134000)
    p.add_argument('--positions',type=int,default=128)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--mode',choices=['auto','packed_index','exact_reference'],default='auto')
    p.add_argument('--report',type=pathlib.Path)
    a=p.parse_args()
    unified,opts,base,comp,metadata=load_candidate(a.checkpoint,a.root)
    if metadata['promotion'] or metadata['accepted_joint_steps']!=14416:
        raise RuntimeError('R7 candidate provenance is invalid')
    runtime=unified.runtime(a.root,base,comp,split=a.split)
    res=runtime.score_positions(np.arange(a.start,a.start+a.positions),mode=a.mode,batch_size=a.batch)
    report={'architecture':'FA1 R7.21 UNIFIED','checkpoint':str(a.checkpoint),
            'original_optimizer_histories':len(opts),
            'accepted_parent_step':metadata['accepted_joint_steps'],
            'candidate_adapter_steps':metadata['candidate_joint_adapter_updates'],
            'split':a.split,'positions':a.positions,'mode_used':res['mode'],
            'mean_baseline_nll':float(np.mean(res['baseline_nll'])),
            'mean_near_nll':float(np.mean(res['near_nll'])),
            'mean_far_nll':float(np.mean(res['far_nll'])),
            'promotion':False,'new_book_validation':False}
    if a.report:
        a.report.parent.mkdir(parents=True,exist_ok=True)
        a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()