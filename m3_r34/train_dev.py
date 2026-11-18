"""One-lineage R34 CPU FIT/GATE/DEV on disjoint future NaNROUTER books.

Only 540..571 are opened; independent TEST 572..595 stays sealed. The
immutable R32 checkpoint supplies all trained prior weights and history.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from m3_default.predictor import R32_CHECKPOINT
from m3_r16.counterfactual_wave_gate import whole_book_deltas
from m3_r17.locked_test import WARMUP
from m3_r23.train_dev import scoring_start
from m3_r32.anchor import load_r32_checkpoint
from m3_r32.train_dev import selective_books
from m3_r34.feedback import M3R34Predictor,feedback_readout
from m3_r34.optimizer import R34ContinuationAdam

FIT=(540,556)
GATE=(556,560)
DEV=(560,572)
SEALED_TEST=(572,596)
ARMS=("r32","r34","anchor_only","local_only")
BATCH=64


def fit(books:list[bytes],prior:np.ndarray,anchor,r32_theta:np.ndarray,
        opt:R34ContinuationAdam)->dict:
    positions=batch=0
    grad=np.zeros(4,dtype=np.float64)
    start=time.perf_counter()
    for i,book in enumerate(books):
        model=M3R34Predictor(prior,anchor,r32_theta,opt.theta)
        warmup=scoring_start(book)
        for pos,target in enumerate(book):
            if pos>=warmup:
                base,f=model.base_and_feedback()
                logp=feedback_readout(base,f,opt.theta)
                error=np.exp(logp)
                error[target]-=1.
                grad+=f@error+.0001*opt.theta
                positions+=1
                batch+=1
                if batch==BATCH:
                    opt.update(grad/batch)
                    model.feedback_gains[:]=opt.theta
                    grad.fill(0.)
                    batch=0
            model.observe(int(target))
        print(f"R34 FIT {i+1}/{len(books)} updates={opt.steps} elapsed={time.perf_counter()-start:.1f}s",
              file=sys.stderr,flush=True)
    if batch:
        opt.update(grad/batch)
    return {"updates":opt.steps,"positions":positions,"theta":opt.theta.tolist(),
            "inherited_adam_groups":32,"new_adam_groups":4,
            "elapsed_s":time.perf_counter()-start}


def evaluate(books:list[bytes],names:list[str],prior:np.ndarray,
             anchor,r32_theta:np.ndarray,feedback_theta:np.ndarray)->dict:
    if len(books)<2 or len(books)!=len(names):
        raise ValueError("at least two disjoint books required")
    totals={key:{"bits":0.,"nonspace":0.,"top1":0,"top5":0} for key in ARMS}
    losses={key:[] for key in ARMS}
    positions=nonspace=0
    rows=[]
    start=time.perf_counter()
    for i,(name,book) in enumerate(zip(names,books)):
        model=M3R34Predictor(prior,anchor,r32_theta,feedback_theta)
        local={key:[] for key in ARMS}
        # One-way ablations prove whether information flow in BOTH directions helps.
        anchor_theta=feedback_theta*np.array((1.,0.,0.,1.))
        local_theta=feedback_theta*np.array((0.,1.,1.,0.))
        warmup=scoring_start(book)
        for pos,target in enumerate(book):
            if pos>=warmup:
                base,feat=model.base_and_feedback()
                predictions=(base,feedback_readout(base,feat,feedback_theta),
                             feedback_readout(base,feat,anchor_theta),
                             feedback_readout(base,feat,local_theta))
                for key,logp in zip(ARMS,predictions):
                    bits=-float(logp[target])/np.log(2.)
                    local[key].append(bits)
                    totals[key]["bits"]+=bits
                    if target!=32:
                        totals[key]["nonspace"]+=bits
                    totals[key]["top1"]+=int(np.argmax(logp)==target)
                    totals[key]["top5"]+=int(target in np.argpartition(logp,-5)[-5:])
                positions+=1
                nonspace+=int(target!=32)
            model.observe(int(target))
        row={"filename":name,"sha256":hashlib.sha256(book).hexdigest(),
             "positions":len(local["r32"])}
        for key in ARMS:
            losses[key].append(np.asarray(local[key]))
            row[key+"_bpb"]=float(np.mean(local[key]))
        row["gain_vs_r32"]=row["r32_bpb"]-row["r34_bpb"]
        rows.append(row)
        print(f"R34 EVAL {i+1}/{len(books)} r32={row['r32_bpb']:.6f} r34={row['r34_bpb']:.6f}",
              file=sys.stderr,flush=True)
    metrics={key:{"bpb":data["bits"]/positions,
                  "nonspace_bpb":data["nonspace"]/nonspace,
                  "top1":data["top1"]/positions,"top5":data["top5"]/positions,
                  "positions":positions} for key,data in totals.items()}
    for key in ARMS[1:]:
        metrics[key]["gain_vs_r32"]=metrics["r32"]["bpb"]-metrics[key]["bpb"]
        metrics[key].update(whole_book_deltas(np.concatenate(losses["r32"]),
                                               np.concatenate(losses[key]),books,WARMUP))
    return {"books":len(books),"positions":positions,"arms":metrics,
            "per_book":rows,"elapsed_s":time.perf_counter()-start}


def gate(result:dict,min_books:int)->dict:
    arms=result["arms"]
    old,new=arms["r32"],arms["r34"]
    return {"lower_bpb":bool(new["bpb"]<old["bpb"]),
            "nonspace_not_worse":bool(new["nonspace_bpb"]<=old["nonspace_bpb"]),
            "top1_not_worse":bool(new["top1"]>=old["top1"]),
            "top5_not_worse":bool(new["top5"]>=old["top5"]),
            "books_improved":bool(new["books_improved"]>=min_books),
            "both_feedback_directions_contribute":bool(new["bpb"]<min(
                arms["anchor_only"]["bpb"],arms["local_only"]["bpb"]))}


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    args=p.parse_args()
    parent=load_r32_checkpoint(R32_CHECKPOINT)
    opt=R34ContinuationAdam(R32_CHECKPOINT)
    needed=set(range(FIT[0],DEV[1]))
    paths,read=selective_books(args.corpus_root,DEV[1],needed)
    hashes=[hashlib.sha256(read[i]).hexdigest() for i in sorted(needed)]
    if len(set(hashes))!=len(needed):
        raise ValueError("new FIT/GATE/DEV documents overlap each other")
    metrics_fit=fit([read[i] for i in range(*FIT)],parent.prior,parent.anchor,
                    parent.anchor_theta,opt)
    metrics_gate=evaluate([read[i] for i in range(*GATE)],
                          [paths[i].name for i in range(*GATE)],
                          parent.prior,parent.anchor,parent.anchor_theta,opt.theta)
    metrics_dev=evaluate([read[i] for i in range(*DEV)],
                         [paths[i].name for i in range(*DEV)],
                         parent.prior,parent.anchor,parent.anchor_theta,opt.theta)
    decisions={"gate":gate(metrics_gate,3),"dev":gate(metrics_dev,9)}
    result={"lineage":"M3 R34 same R32 causal state", "parent_r32_certified":True,
            "fit_indices":FIT,"gate_indices":GATE,"dev_indices":DEV,
            "sealed_test_indices":SEALED_TEST,"sealed_test_read":False,
            "fitted":metrics_fit,"gate":metrics_gate,"dev":metrics_dev,
            "acceptance":decisions,"candidate_eligible":all(all(x.values()) for x in decisions.values()),
            "promotion":"R32 remains default; R34 unpromoted candidate"}
    opt.save(args.checkpoint,FIT,GATE,DEV,SEALED_TEST)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"acceptance":decisions,"dev":metrics_dev["arms"]},indent=2))


if __name__=="__main__":
    main()
