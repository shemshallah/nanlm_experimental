"""R32 exact-checkpoint independent TEST, prelocked before 516..539 is opened.

The sealed documents are scored once, causally, using only the original R32
checkpoint and TRAIN-only background. No training or threshold changes occur.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from m3_r17.locked_test import BYTES_PER_BOOK, SEED
from m3_r27.locked_test import digest
from m3_r32.anchor import load_r32_checkpoint
from m3_r32.train_dev import DEV, SEALED_TEST, evaluate, selective_books

LOCK_PATH = Path(__file__).with_name("r32_frozen_candidate.json")


def report_json(data:dict)->str:
    """Export NumPy metric scalars without modifying evaluation results."""
    def normalize(value):
        if isinstance(value,np.generic):
            return value.item()
        raise TypeError(f"unsupported metric type: {type(value).__name__}")
    return json.dumps(data,indent=2,sort_keys=True,default=normalize)


def evaluate_acceptance(measured:dict,locked:dict)->tuple[bool,dict]:
    requirements=locked["test_acceptance"]
    arms=measured["arms"]
    candidate=arms["r32"]
    conditions={
        "all_bpb_controls":all(candidate["bpb"]<arms[key]["bpb"]
                               for key in requirements["require_bpb_better_than"]),
        "nonspace_preserved":candidate["nonspace_bpb"]<=arms["r31"]["nonspace_bpb"],
        "top1_preserved":candidate["top1"]>=max(arms[key]["top1"]
                             for key in requirements["top1_not_worse_than"]),
        "top5_preserved":candidate["top5"]>=arms["r31"]["top5"],
        "books_improved":candidate["books_improved"]>=requirements["min_books_improved_over_r31"],
        "positive_book_bootstrap_lower":candidate["whole_book_bootstrap_95pct_bits"][0]>0,
        "exact_test_books":measured["books"]==requirements["test_books"]
    }
    return all(conditions.values()),conditions


def locked_test(corpus_root:Path,checkpoint:Path,training_report:Path)->dict:
    lock=json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if (digest(checkpoint)!=lock["checkpoint_sha256"]
            or digest(training_report)!=lock["training_dev_report_sha256"]):
        raise ValueError("R32 checkpoint/report changed since TEST lock")
    parent=json.loads(training_report.read_text(encoding="utf-8"))
    if (not parent["gate_pass"] or not parent["dev_gate_pass"]
            or parent["sealed_test_read"]
            or parent["parent_checkpoint_sha256"]!=lock["parent_r31_checkpoint_sha256"]
            or parent["dev_indices"]!=list(DEV)
            or parent["sealed_test_indices"]!=list(SEALED_TEST)):
        raise ValueError("invalid R32 locked GATE/DEV ancestry")
    with np.load(checkpoint,allow_pickle=False) as x:
        theta=np.asarray(x["theta"],dtype=np.float64)
        ages=np.asarray(x["adam_steps"],dtype=np.int64)
    if (theta.shape!=(lock["optimizer_groups"],)
            or not np.array_equal(theta[-4:],np.asarray(lock["coefficients"]))
            or ages.tolist()!=[1536]*3+[768]*29):
        raise ValueError("R32 trained coefficients or optimizer ages differ from lock")
    model=load_r32_checkpoint(checkpoint)  # embedded anchor and hashes validated
    paths,opened=selective_books(corpus_root,SEALED_TEST[1],set(range(*SEALED_TEST)))
    test_books=[opened[i] for i in range(*SEALED_TEST)]
    test_names=[paths[i].name for i in range(*SEALED_TEST)]
    test_hashes=[hashlib.sha256(x).hexdigest() for x in test_books]
    used=set(parent["background_sha256"]+parent["fit_sha256"]+
             parent["gate_sha256"]+parent["dev_sha256"])
    if (len(set(test_hashes))!=len(test_books) or set(test_hashes)&used
            or len(set(test_names))!=len(test_names)):
        raise ValueError("overlapping, duplicated or modified independent TEST books")
    measured=evaluate(test_books,test_names,model.prior,model.anchor,model.anchor_theta)
    accepted,conditions=evaluate_acceptance(measured,lock)
    return {"architecture":"M3 R32 train-only anchor: independent TEST",
            "locked_candidate_sha256":lock["checkpoint_sha256"],
            "locked_dev_report_sha256":lock["training_dev_report_sha256"],
            "training_commit":lock["selection_commit"],
            "test_indices":list(SEALED_TEST),"test_sha256":test_hashes,
            "test_opened_after_lock":True,"trained_weights_unchanged":True,
            "test":measured,"precommitted_acceptance":conditions,
            "independent_test_pass":bool(accepted),
            "promotion":"eligible for separately verified master default promotion" if accepted
                        else "not eligible: R31 directed default remains"}


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus-root",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--training-report",type=Path,required=True)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    result=locked_test(args.corpus_root,args.checkpoint,args.training_report)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(report_json(result)+"\n",encoding="utf-8")
    print(report_json({"acceptance":result["precommitted_acceptance"],
                      "independent_test_pass":result["independent_test_pass"],
                      "test_arms":result["test"]["arms"]}))


if __name__=="__main__":
    main()
