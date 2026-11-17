#!/usr/bin/env python3
"""FA2 R63 real-checkpoint sequential book-disjoint DEV scorer (TEST sealed).

One model, no training, no checkpoint overwrites. Candidate controls use
the same frozen model outputs and real causal C(256,2) pair addresses.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from book_stream_eval import ARMS,METRICS,evaluate_causal_book
from poincare_frontier import Config,MirrorLM,load_v25_compatible,file_hash
from train_r55_pair_successor import checked_corpus
from probe_r47_selective_emission import book_comparison
from fa2_active import resolve_active_checkpoint, ACTIVE_MANIFEST


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,default=None,
                   help="Defaults to the hash-verified unified FA2_ACTIVE checkpoint")
    p.add_argument("--expected-parent-sha256",default=None)
    p.add_argument("--active-manifest",type=Path,default=ACTIVE_MANIFEST)
    p.add_argument("--data-dir",type=Path,required=True)
    p.add_argument("--output-report",type=Path,required=True)
    p.add_argument("--dev-books",type=int,default=12)
    p.add_argument("--max-book-positions",type=int,default=512)
    p.add_argument("--warmup-positions",type=int,default=0,
                   help="Observe this many causal bytes per DEV book before scoring")
    p.add_argument("--context",type=int,default=64)
    p.add_argument("--chunk",type=int,default=16)
    p.add_argument("--history",type=int,default=32768)
    p.add_argument("--bootstrap",type=int,default=2000)
    p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--key-mode",default="hierarchical",
                   choices=("pair_byte","pair_bigram","head_byte","hierarchical"))
    p.add_argument("--zero-init-joint",action="store_true",
                   help="R58 source-only parity diagnostic; not a trained R62 candidate")
    p.add_argument("--space-rank-guard",action="store_true",
                   help="R67: preserve inherited space/EOS top-1 at maximum safe memory strength")
    args=p.parse_args()
    if (min(args.dev_books,args.max_book_positions,args.context,args.chunk,
            args.history,args.bootstrap,args.threads)<1 or
            args.chunk>args.context or args.warmup_positions<0):
        p.error("Invalid workload quantities")
    if args.output_report.exists():
        p.error("Refuse to overwrite an existing report")
    if args.checkpoint is None:
        args.checkpoint,active=resolve_active_checkpoint(manifest_path=args.active_manifest)
        if args.expected_parent_sha256 is None:
            args.expected_parent_sha256=active['checkpoint_sha256']
    elif args.expected_parent_sha256 is None:
        p.error('Explicit --checkpoint requires --expected-parent-sha256')
    current_sha=file_hash(args.checkpoint)
    if current_sha!=args.expected_parent_sha256:
        raise ValueError("Requested FA2 checkpoint SHA-256 does not match")
    torch.set_num_threads(args.threads)
    saved=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    meta,arrays=checked_corpus(args.data_dir,saved)  # TRAIN/DEV only; never opens TEST
    model=MirrorLM(Config(**saved["config"])).eval()
    load_v25_compatible(model,saved["model"])
    if model.pair_successor is None:
        raise ValueError("Checkpoint lacks FA2 pair-successor model lineage")
    if model.pair_successor.joint_weight is None:
        if not args.zero_init_joint:
            raise ValueError("Checkpoint lacks trained joint rail; --zero-init-joint is diagnostic only")
        model.enable_joint_pair_successor(key_mode=args.key_mode)
    n=min(args.dev_books,len(meta["valid_doc_lengths"]))
    all_rows={arm:[] for arm in ARMS}
    per_book={}
    offset=0
    for book_idx,length in enumerate(meta["valid_doc_lengths"]):
        if book_idx>=n:
            break
        # Keep two extra true targets for +2/+3 without scoring them as
        # additional query positions. The causal warm-up only updates memory.
        scored=min(args.warmup_positions+args.max_book_positions+3,int(length))
        if scored<=args.warmup_positions+3:
            raise ValueError("Book too short for requested warm-up and multi-horizon scoring")
        book=np.asarray(arrays["valid"][offset:offset+scored],dtype=np.int64)
        bytes_tensor=torch.from_numpy(book.copy())
        identifier="DEV:"+str(book_idx)
        rows=evaluate_causal_book(
            model,bytes_tensor,identifier,context=args.context,chunk=args.chunk,
            history=args.history,key_mode=args.key_mode,
            score_start=args.warmup_positions,
            score_positions=args.max_book_positions,
            space_rank_guard=args.space_rank_guard)
        per_book[identifier]=rows
        for arm in ARMS:
            all_rows[arm].append({metric:rows[arm][metric] for metric in METRICS})
        offset+=int(length)
    comparisons={}
    for control in ("neutral","wrong_address","wrong_byte","wrong_successor"):
        comparisons["exact_vs_"+control]=book_comparison(
            all_rows[control],all_rows["exact"],args.bootstrap,args.seed,4)
    summary={}
    for arm in ARMS:
        summary[arm]={metric:float(np.mean([row[metric] for row in all_rows[arm]]))
                      for metric in METRICS}
        summary[arm]["bits_per_byte_or_eos_target"]=summary[arm]["ce_1"]/math.log(2)
        summary[arm]["mean_book_bits_per_raw_byte"]=float(np.mean(
            [r[arm]["bits_per_raw_byte"] for r in per_book.values()]))
        summary[arm]["mean_positions_per_second"]=float(np.mean(
            [r[arm]["evaluated_tokens_per_second"] for r in per_book.values()]))
    parent=comparisons["exact_vs_neutral"]
    accepted=(all(c["ce_1"]["ci95_bonferroni"][0]>0 and
                  c["ce_nonspace_1"]["ci95_bonferroni"][0]>0
                  for c in comparisons.values()) and
              parent["acc_1"]["mean_gain"]>=0 and
              parent["top5_1"]["mean_gain"]>=0 and
              all(parent[k]["mean_gain"]>=0 for k in ("ce_2","ce_3")))
    report={
        "round":"FA2_R63_BOOK_STREAM_SEQUENTIAL_DEV",
        "status":"DEV_screen_only_no_promotion",
        "parent_step":saved.get("step"),
        "parent_sha256":current_sha,
        "parent_config":asdict(model.cfg),
        "zero_initialized_joint_diagnostic":args.zero_init_joint,
        "space_rank_guard":bool(args.space_rank_guard or model.pair_successor.space_rank_guard),
        "space_rank_guard_source":("checkpoint" if model.pair_successor.space_rank_guard
                                   else "evaluation_ablation" if args.space_rank_guard
                                   else "disabled"),
        "dev_books_scored":n,
        "requested_positions_per_book":args.max_book_positions,
        "warmup_positions_per_book":args.warmup_positions,
        "actual_scored_positions_per_book":{
            book_id:rows["exact"]["positions_scored"]
            for book_id,rows in per_book.items()},
        "context":args.context,"chunk":args.chunk,"history":args.history,
        "key_mode":args.key_mode,
        "test_opened":False,"checkpoint_modified":False,
        "bootstrap_unit":"whole independent DEV book",
        "familywise_controls":4,
        "summary":summary,"comparisons":comparisons,
        "book_rows":per_book,"dev_screen_pass":bool(accepted),"promoted":False,
    }
    args.output_report.parent.mkdir(parents=True,exist_ok=True)
    args.output_report.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":report["status"],"summary":summary,
                      "dev_screen_pass":accepted,"report":str(args.output_report)},indent=2))


if __name__=="__main__":
    main()
