#!/usr/bin/env python3
"""FA2 R27: CPU trust calibration with NO inherited parameter or Adam resets.

Input: immutable R19 ancestor overlaid by the R26 latest recovery ZIP.
Selection: R26-relative dev BPB, nonspace, unrelated-book donor and 132K guard.
No historical test is consulted during fitting or checkpoint selection.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import pathlib
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from r13_corpus_cache import read_books_cached
from r26_restore_unified import restore
from r26_unified_cpu import extract_split
from r27_trust import CounterfactualTrustGate, near_only_target


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(4 << 20), b""):
            h.update(part)
    return h.hexdigest()


def row_scores(unified, row, wrong=False):
    if wrong:
        base, prob, feats = row["wrong_y"], row["wp_y"], row["wfeatures"]
    else:
        base, prob, feats = row["base_y"], row["p_y"], row["features"]
    full = unified.r26.target_logprob(base, prob, feats)
    near = near_only_target(unified.r26, base, prob, feats)
    blended = unified.r27(full, near, feats)
    return full, near, blended


def score(unified, rows):
    total = {"reference":0.0, "selected":0.0, "donor":0.0,
             "nonspace":0.0, "nonspace_n":0, "bytes":0, "tokens":0}
    bands = {}
    books = {}
    with torch.no_grad():
        for row in rows:
            ref, near, selected = row_scores(unified, row)
            _, _, unrelated = row_scores(unified, row, wrong=True)
            n = len(selected)
            if not n:
                continue
            total["reference"] -= float(ref.sum())
            total["selected"] -= float(selected.sum())
            total["donor"] -= float(unrelated.sum())
            total["bytes"] += int(row["byte_count"])
            total["tokens"] += n
            nonspace = row["targets"] != 32
            total["nonspace"] -= float(selected[nonspace].sum())
            total["nonspace_n"] += int(nonspace.sum())
            band = "132K" if row["start"] >= 131072 else (
                "65K" if row["start"] >= 65536 else (
                    "33K" if row["start"] >= 32768 else "near"))
            for target, key in ((bands,band),(books,row["book_id"])):
                a = target.setdefault(key, [0.0,0.0,0])
                a[0] -= float(ref.sum())
                a[1] -= float(selected.sum())
                a[2] += row["byte_count"]
    if not total["bytes"] or not total["nonspace_n"]:
        raise ValueError("missing scored windows")
    d = math.log(2.0)
    return {
        "r26_bpb":total["reference"]/total["bytes"]/d,
        "r27_bpb":total["selected"]/total["bytes"]/d,
        "gain_bpb":(total["reference"]-total["selected"])/total["bytes"]/d,
        "correct_vs_donor_nats":(total["donor"]-total["selected"])/total["tokens"],
        "nonspace_nll":total["nonspace"]/total["nonspace_n"],
        "bytes":total["bytes"], "tokens":total["tokens"],
        "books_improved":sum(v[1]<v[0] for v in books.values()),
        "books":len(books),
        "bands":{k:{"r26_bpb":v[0]/v[2]/d,
                     "r27_bpb":v[1]/v[2]/d,
                     "gain_bpb":(v[0]-v[1])/v[2]/d,
                     "bytes":v[2]} for k,v in bands.items()}
    }


def assert_inheritance(before_model, before_opt, now_model, now_opt):
    current = now_model.state_dict()
    for name, orig in before_model.items():
        if not torch.equal(orig,current[name]):
            raise AssertionError("R26 weight changed: "+name)
    new = now_opt.state_dict()
    if len(new["param_groups"]) != len(before_opt["param_groups"])+1:
        raise AssertionError("wrong count of optimizer groups")
    if new["param_groups"][:-1] != before_opt["param_groups"]:
        raise AssertionError("inherited optimizer groups mutated")
    for key, values in before_opt["state"].items():
        if key not in new["state"]:
            raise AssertionError("lost inherited Adam state: "+str(key))
        for field, orig in values.items():
            actual = new["state"][key][field]
            if not (torch.equal(orig, actual) if torch.is_tensor(orig)
                    else orig == actual):
                raise AssertionError("inherited Adam changed: "+str((key,field)))
    return len(before_model),len(before_opt["state"])


def run(args):
    root=pathlib.Path(args.root).resolve()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    source=root/"reports/fa2_r26_unified_candidate.pth"
    original_digest=digest(source)
    unified,opt,comp,r6,meta,ancestor=restore(root)
    if ancestor["selected"]["r26_step"] != 90:
        raise ValueError("expected corrected R26 step-90 predecessor")
    if len(opt.param_groups)!=11 or len(opt.state)!=137:
        raise ValueError("R26 optimizer state mismatch")
    before_model={k:v.detach().clone() for k,v in unified.state_dict().items()}
    before_opt=copy.deepcopy(opt.state_dict())
    before_carry=copy.deepcopy(ancestor["selected"]["carry"])
    unified.add_module("r27",CounterfactualTrustGate())
    opt.add_param_group({"params":list(unified.r27.parameters()),
                         "lr":args.lr,"weight_decay":1e-4})
    for name,param in unified.named_parameters():
        param.requires_grad_(name.startswith("r27."))
    all_books,_=read_books_cached(root/"nanrouter",comp,r6.vocab,0)
    offsets=[int(o) for o in args.offsets.split(",")]
    train=extract_split(unified,all_books["train"][:args.train_books],
                        offsets,args.chunk_span,args.windows,"r27_training")
    dev_books=list(all_books["dev"][:args.dev_books])
    anchor=next((b for b in all_books["dev"] if str(b.book_id)==str(args.anchor_book)),None)
    if anchor is not None and all(b.book_id!=anchor.book_id for b in dev_books):
        dev_books.append(anchor)
    dev=extract_split(unified,dev_books,offsets,
                      args.chunk_span,args.windows,"r27_selection")
    if not train or not dev:
        raise ValueError("no causal data extracted")
    unified.eval()
    baseline=score(unified,dev)
    if not args.allow_no_132k and "132K" not in baseline["bands"]:
        raise ValueError("No 132K development windows: no frontier trust promotion allowed")
    best=(baseline["r27_bpb"],0,copy.deepcopy(unified.r27.state_dict()),
          copy.deepcopy(opt.state_dict()),baseline)
    report={"format":"FA2_R27_TRUST_CPU_V1","r26_checkpoint_sha256":original_digest,
            "inherited_r26_step":90,"selected_updates":0,
            "attempted_updates":args.steps,"baseline":baseline,
            "status":"no_promotion","settings":vars(args)}
    params=list(unified.r27.parameters())
    for step in range(1,args.steps+1):
        unified.train()
        batch=random.sample(train,min(args.batch,len(train)))
        losses=[]
        for row in batch:
            full, near, pred=row_scores(unified,row)
            _, _, wrong=row_scores(unified,row,wrong=True)
            loss=-pred.mean()+args.donor_weight*F.softplus(
                args.margin+wrong.mean()-pred.mean())
            losses.append(loss)
        opt.zero_grad(set_to_none=True)
        loss=torch.stack(losses).mean()
        loss.backward()
        grad=float(torch.nn.utils.clip_grad_norm_(params,2.0))
        if not math.isfinite(grad):
            raise ValueError("non-finite R27 gradient")
        opt.step()
        if step%args.eval_every==0 or step==args.steps:
            unified.eval()
            m=score(unified,dev)
            long_base=baseline["bands"].get("132K")
            long_new=m["bands"].get("132K")
            far_safe=(not long_base or not long_new or
                long_new["r27_bpb"]<=long_base["r26_bpb"]+args.band_slack)
            near_base=baseline["bands"].get("near")
            near_new=m["bands"].get("near")
            near_safe=(not near_base or not near_new or
                near_new["r27_bpb"]<=near_base["r26_bpb"]+args.band_slack)
            eligible=(m["r27_bpb"]<best[0]-args.min_gain and
                      m["nonspace_nll"]<=baseline["nonspace_nll"]+args.nonspace_slack and
                      m["correct_vs_donor_nats"]>0 and far_safe and near_safe)
            print("R27_DEV",json.dumps({"step":step,"gain_bpb":m["gain_bpb"],
                "r27_bpb":m["r27_bpb"],"far_safe":far_safe,
                "near_safe":near_safe,"eligible":eligible}),flush=True)
            if eligible:
                best=(m["r27_bpb"],step,copy.deepcopy(unified.r27.state_dict()),
                      copy.deepcopy(opt.state_dict()),m)
    unified.r27.load_state_dict(best[2])
    opt.load_state_dict(best[3])
    unified.eval()
    inherits=assert_inheritance(before_model,before_opt,unified,opt)
    if len(ancestor["selected"]["carry"])!=len(before_carry):
        raise AssertionError("inherited training streams changed")
    if digest(source)!=original_digest:
        raise AssertionError("R26 ancestor checkpoint mutated")
    report.update({"selected_updates":best[1],"selected":score(unified,dev),
                   "inheritance":{"inherited_model_tensors":inherits[0],
                                  "inherited_Adam_states":inherits[1],
                                  "stream_count":len(before_carry)},
                   "status":"R27 experimental candidate" if best[1] else "no_promotion"})
    destination=root/"reports/r27_trust_trial.json"
    destination.write_text(json.dumps(report,indent=2)+"\n")
    if best[1]:
        payload={"format":"FA2_R27_UNIFIED_CANDIDATE_V1",
                 "parent_r26_sha256":original_digest,
                 "accepted_model_promoted":False,
                 "selected":{"r27_step":best[1],
                             "model":unified.state_dict(),
                             "optimizer":opt.state_dict(),
                             "carry":before_carry},
                 "selection_dev":report["selected"]}
        path=root/"reports/fa2_r27_unified_candidate.pth"
        temp=path.with_suffix(".tmp.pth")
        torch.save(payload,temp)
        os.replace(temp,path)
        report["candidate_checkpoint_sha256"]=digest(path)
        destination.write_text(json.dumps(report,indent=2)+"\n")
    print("R27_FINAL",json.dumps({"selected_step":best[1],
        "r26_bpb":baseline["r26_bpb"],
        "r27_bpb":report["selected"]["r27_bpb"],
        "checkpoint_written":bool(best[1]),"legacy_states":inherits[1]}),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--steps",type=int,default=100)
    ap.add_argument("--lr",type=float,default=.004)
    ap.add_argument("--seed",type=int,default=27027)
    ap.add_argument("--threads",type=int,default=2)
    ap.add_argument("--train-books",type=int,default=4)
    ap.add_argument("--dev-books",type=int,default=5)
    ap.add_argument("--anchor-book",default="1444")
    ap.add_argument("--offsets",default="4096,8192,33000,65000,132000")
    ap.add_argument("--chunk-span",type=int,default=20)
    ap.add_argument("--windows",type=int,default=2)
    ap.add_argument("--batch",type=int,default=12)
    ap.add_argument("--eval-every",type=int,default=10)
    ap.add_argument("--donor-weight",type=float,default=.025)
    ap.add_argument("--margin",type=float,default=.01)
    ap.add_argument("--min-gain",type=float,default=1e-6)
    ap.add_argument("--nonspace-slack",type=float,default=1e-4)
    ap.add_argument("--band-slack",type=float,default=1e-5)
    ap.add_argument("--allow-no-132k",action="store_true")
    run(ap.parse_args())
