"""R35 CPU train + disjoint GATE/DEV; immutable R32 stays the M3 default.

R34 FIT/GATE are reused with explicit disclosure. R35 DEV 596–607 is fresh,
never scored by R34, and the R34 independent TEST 572–595 remains sealed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from m3_default.predictor import R32_CHECKPOINT,R32_TRAINING_CHECKPOINT_SHA256
from m3_r16.counterfactual_wave_gate import whole_book_deltas
from m3_r17.locked_test import WARMUP
from m3_r23.train_dev import scoring_start
from m3_r32.anchor import load_r32_checkpoint
from m3_r32.train_dev import selective_books
from m3_r35.local_feedback import M3R35Predictor,local_readout
from m3_r35.optimizer import R35ContinuationAdam

FIT=(540,556)
GATE=(556,560)
DEV=(596,608)
R34_SEALED_TEST=(572,596)
ARMS=("r32","r34_local_only","r35")
BATCH=64


def get_r34_reference(path:Path)->np.ndarray:
    if hashlib.sha256(path.read_bytes()).hexdigest()!="0d63bff7ce91d6444d1c2eaf4de15021f981401ba210e0208e08e5ac43bf8b5c":
        raise ValueError("R34 ablation reference does not match frozen recovery artifact")
    with np.load(path,allow_pickle=False) as data:
        theta=data["theta"]
        if (theta.shape!=(36,) or
            str(data["r34_parent_checkpoint_sha256"])!=R32_TRAINING_CHECKPOINT_SHA256):
            raise ValueError("R34 reference is not the matched R32 descendant")
        with np.load(R32_CHECKPOINT,allow_pickle=False) as parent:
            for key in ("theta","adam_m","adam_v","adam_steps"):
                if not np.array_equal(data[key][:32],parent[key]):
                    raise ValueError("R34 reference lost inherited optimizer parity")
        return theta[33:35].copy()


def fit(books:list[bytes],prior:np.ndarray,anchor,r32_theta:np.ndarray,
        opt:R35ContinuationAdam)->dict:
    positions=batch=0
    grad=np.zeros(2,dtype=np.float64)
    start=time.perf_counter()
    for i,book in enumerate(books):
        model=M3R35Predictor(prior,anchor,r32_theta,opt.theta)
        warmup=scoring_start(book)
        for pos,target in enumerate(book):
            if pos>=warmup:
                base,features=model.base_and_local_feedback()
                logp=local_readout(base,features,opt.theta)
                error=np.exp(logp)
                error[target]-=1.
                grad+=features@error+.0001*opt.theta
                positions+=1
                batch+=1
                if batch==BATCH:
                    opt.update(grad/batch)
                    model.local_gains[:]=opt.theta
                    grad.fill(0.)
                    batch=0
            model.observe(int(target))
        print(f"R35 FIT {i+1}/{len(books)} steps={opt.steps} elapsed={time.perf_counter()-start:.1f}s",
              file=sys.stderr,flush=True)
    if batch:
        opt.update(grad/batch)
    return {"updates":opt.steps,"positions":positions,"theta":opt.theta.tolist(),
            "inherited_adam_groups":32,"new_adam_groups":2,
            "elapsed_s":time.perf_counter()-start}


def evaluate(books:list[bytes],names:list[str],prior:np.ndarray,anchor,
             r32_theta:np.ndarray,r34_local:np.ndarray,r35_theta:np.ndarray)->dict:
    if len(books)<2 or len(books)!=len(names):
        raise ValueError("evaluation requires at least two named independent books")
    totals={a:{"bits":0.,"nonspace":0.,"top1":0,"top5":0} for a in ARMS}
    losses={a:[] for a in ARMS}
    positions=nonspace=0
    rows=[]
    start=time.perf_counter()
    for i,(name,book) in enumerate(zip(names,books)):
        model=M3R35Predictor(prior,anchor,r32_theta,r35_theta)
        local={a:[] for a in ARMS}
        warmup=scoring_start(book)
        for pos,target in enumerate(book):
            if pos>=warmup:
                base,features=model.base_and_local_feedback()
                predictions=(base,local_readout(base,features,r34_local),
                             local_readout(base,features,r35_theta))
                for arm,logp in zip(ARMS,predictions):
                    bits=-float(logp[target])/np.log(2.)
                    local[arm].append(bits)
                    totals[arm]["bits"]+=bits
                    if target!=32:
                        totals[arm]["nonspace"]+=bits
                    totals[arm]["top1"]+=int(np.argmax(logp)==target)
                    totals[arm]["top5"]+=int(target in np.argpartition(logp,-5)[-5:])
                positions+=1
                nonspace+=int(target!=32)
            model.observe(int(target))
        row={"filename":name,"sha256":hashlib.sha256(book).hexdigest(),
             "positions":len(local["r32"])}
        for arm in ARMS:
            losses[arm].append(np.asarray(local[arm]))
            row[arm+"_bpb"]=float(np.mean(local[arm]))
        row["gain_vs_r32"]=row["r32_bpb"]-row["r35_bpb"]
        rows.append(row)
        print(f"R35 EVAL {i+1}/{len(books)} r32={row['r32_bpb']:.6f} "
              f"r34local={row['r34_local_only_bpb']:.6f} r35={row['r35_bpb']:.6f}",
              file=sys.stderr,flush=True)
    metrics={a:{"bpb":v["bits"]/positions,"nonspace_bpb":v["nonspace"]/nonspace,
                "top1":v["top1"]/positions,"top5":v["top5"]/positions,
                "positions":positions} for a,v in totals.items()}
    for arm in ARMS[1:]:
        metrics[arm]["gain_vs_r32"]=metrics["r32"]["bpb"]-metrics[arm]["bpb"]
        metrics[arm].update(whole_book_deltas(np.concatenate(losses["r32"]),
                                               np.concatenate(losses[arm]),books,WARMUP))
    return {"books":len(books),"positions":positions,"arms":metrics,
            "per_book":rows,"elapsed_s":time.perf_counter()-start}


def gate(result:dict,min_books:int,require_bootstrap:bool=False)->dict:
    arms=result["arms"]
    old,new=arms["r32"],arms["r35"]
    conditions={"lower_bpb":bool(new["bpb"]<old["bpb"]),
                "nonspace_not_worse":bool(new["nonspace_bpb"]<=old["nonspace_bpb"]),
                "top1_not_worse":bool(new["top1"]>=old["top1"]),
                "top5_not_worse":bool(new["top5"]>=old["top5"]),
                "books_improved":bool(new["books_improved"]>=min_books)}
    if require_bootstrap:
        # Whole-book resampling: reject if the 95% gain interval includes zero.
        bounds=new.get("whole_book_bootstrap_95pct_bits")
        if bounds is None or len(bounds)!=2:
            raise ValueError(f"missing whole-book bootstrap interval: {list(new)}")
        conditions["book_bootstrap_positive"]=bool(bounds[0]>0)
    return conditions


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--r34-reference",type=Path,
                        default=Path("m3_r34/results/R34_36_group_candidate.npz"))
    args=parser.parse_args()
    parent=load_r32_checkpoint(R32_CHECKPOINT)
    old_local=get_r34_reference(args.r34_reference)
    opt=R35ContinuationAdam(R32_CHECKPOINT)
    needed=set(range(*FIT))|set(range(*GATE))|set(range(*DEV))
    paths,read=selective_books(args.corpus_root,DEV[1],needed)
    hashes=[hashlib.sha256(read[i]).hexdigest() for i in sorted(needed)]
    if len(set(hashes))!=len(needed):
        raise ValueError("FIT/GATE/DEV documents overlap")
    stats_fit=fit([read[i] for i in range(*FIT)],parent.prior,parent.anchor,
                  parent.anchor_theta,opt)
    stats_gate=evaluate([read[i] for i in range(*GATE)],
                        [paths[i].name for i in range(*GATE)],
                        parent.prior,parent.anchor,parent.anchor_theta,old_local,opt.theta)
    stats_dev=evaluate([read[i] for i in range(*DEV)],
                       [paths[i].name for i in range(*DEV)],
                       parent.prior,parent.anchor,parent.anchor_theta,old_local,opt.theta)
    decisions={"gate":gate(stats_gate,3),"dev":gate(stats_dev,9,True)}
    candidate_passes=all(all(g.values()) for g in decisions.values())
    result={"lineage":"M3 R35 local-only feedback on immutable R32",
            "parent_r32_certified":True,"fit_indices":FIT,"gate_indices":GATE,
            "fresh_dev_indices":DEV,"r34_sealed_test_indices":R34_SEALED_TEST,
            "r34_sealed_test_read":False,"fit":stats_fit,"gate":stats_gate,
            "dev":stats_dev,"acceptance":decisions,
            "eligible_for_independent_test":candidate_passes,
            "promoted":False,"production_default":"r32",
            "note":"R35 experimental; no independent R35 TEST until a prelocked new corpus is reserved"}
    opt.save(args.checkpoint,FIT,GATE,DEV)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"acceptance":decisions,"dev_arms":stats_dev["arms"],
                      "eligible_for_independent_test":candidate_passes},indent=2))


if __name__=="__main__":
    main()
