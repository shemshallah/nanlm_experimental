"""M3 R31 class-calibrated feedback: causal CPU TRAIN/GATE/DEV continuation.

Full R30 25-group checkpoint is SHA-verified and copied unchanged. Three new
Adam groups are optimized on books 412..427. GATE 428..431 and DEV 432..443
are book-disjoint; TEST 444..467 remains sealed and unread.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from m3_default.predictor import R26_THETA, R29_THETA
from m3_r14.predictive_probe import resolve_books, train_unigram_prior
from m3_r16.counterfactual_wave_gate import whole_book_deltas
from m3_r17.locked_test import SEED, TRAIN_BOOKS, BYTES_PER_BOOK, WARMUP
from m3_r18.locked_replication import MANIFEST, verify_manifest
from m3_r22.evaluate import R21_TEST_HASHES
from m3_r23.train_dev import scoring_start
from m3_r27.locked_test import digest
from m3_r29.lexical import WordSuccessorMemory, pack_lexical, lexical_readout
from m3_r30.router import router_features, router_readout, support
from m3_r31.calibration import (M3R31Predictor, R30_DELTA,
                                CALIBRATION_WEIGHTS, class_features, calibrated_readout)

PARENT_SHA = "a757433f6b72b4b444763de2f3158eea960a530c94fe613af58de247647b3421"
FIT = (412, 428)
GATE = (428, 432)
DEV = (432, 444)
SEALED_TEST = (444, 468)
ARMS = ("r29", "r30", "r31", "wrong_successor", "unrelated_matched", "unrelated_full")
BATCH = 64
LR = .004


class FrozenR30Adam:
    """25 inherited readout and Adam histories + 3 new group moments/ages."""

    def __init__(self, path: Path):
        if not path.is_file() or digest(path) != PARENT_SHA:
            raise ValueError("missing/altered full R30 checkpoint")
        with np.load(path, allow_pickle=False) as x:
            self.parent_theta = np.asarray(x["theta"], dtype=np.float64).copy()
            self.parent_m = np.asarray(x["adam_m"], dtype=np.float64).copy()
            self.parent_v = np.asarray(x["adam_v"], dtype=np.float64).copy()
            self.parent_steps = np.asarray(x["adam_steps"], dtype=np.int64).copy()
            self.prior = np.asarray(x["prior"], dtype=np.float64).copy()
            self.parent_fit = tuple(x["fit_indices"].tolist())
        if (self.parent_theta.shape != (25,) or self.parent_m.shape != (25,)
                or self.parent_v.shape != (25,) or self.parent_steps.shape != (25,)
                or self.prior.shape != (256,) or self.parent_fit != (356, 372)
                or self.parent_steps.tolist() != [1536]*3 + [768]*22
                or not np.array_equal(self.parent_theta[:14], R26_THETA)
                or not np.array_equal(self.parent_theta[14:22], R29_THETA)
                or not np.array_equal(self.parent_theta[22:], R30_DELTA)):
            raise ValueError("R30 parent weights, prior or optimizer state changed")
        self.theta = np.zeros(CALIBRATION_WEIGHTS, dtype=np.float64)
        self.m = np.zeros_like(self.theta)
        self.v = np.zeros_like(self.theta)
        self.step = 0

    def update(self,gradient:np.ndarray)->None:
        if gradient.shape!=(CALIBRATION_WEIGHTS,):
            raise ValueError("expected three class-gate gradients")
        if not np.isfinite(gradient).all():
            raise ValueError("non-finite calibration gradient")
        self.step+=1
        g=np.clip(gradient,-2,2)
        self.m=.9*self.m+.1*g
        self.v=.999*self.v+.001*g*g
        self.theta-=LR*(self.m/(1-.9**self.step))/(np.sqrt(self.v/(1-.999**self.step))+1e-8)
        np.clip(self.theta,-1.5,1.5,out=self.theta)

    def save(self,path:Path,hashes:list[str])->None:
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez(path,theta=np.concatenate((self.parent_theta,self.theta)),
                 adam_m=np.concatenate((self.parent_m,self.m)),
                 adam_v=np.concatenate((self.parent_v,self.v)),
                 adam_steps=np.concatenate((self.parent_steps,np.full(3,self.step))),
                 prior=self.prior,parent_checkpoint_sha256=PARENT_SHA,
                 fit_indices=np.array(FIT),fit_sha256=np.array(hashes[FIT[0]:FIT[1]]),
                 gate_indices=np.array(GATE),dev_indices=np.array(DEV),
                 sealed_test_indices=np.array(SEALED_TEST))


def train(books:list[bytes],prior:np.ndarray,opt:FrozenR30Adam)->dict:
    np.testing.assert_array_equal(prior,opt.prior)
    grad=np.zeros(CALIBRATION_WEIGHTS)
    queued=positions=0
    bits=0.
    started=time.perf_counter()
    for index,book in enumerate(books):
        model=M3R31Predictor(prior)
        warmup=scoring_start(book)
        for pos,target in enumerate(book):
            if pos>=warmup:
                base,features=model.r29_base_and_calibration()
                logp=calibrated_readout(base,features,opt.theta)
                error=np.exp(logp)
                error[target]-=1.
                importance=.8 if target==32 else 1.2
                grad+=importance*(features@error)+.0001*opt.theta
                positions+=1
                queued+=1
                bits-=float(logp[target])/np.log(2)
                if queued==BATCH:
                    opt.update(grad/queued)
                    grad.fill(0)
                    queued=0
            model.observe(int(target))
        print(f"R31 FIT {index+1}/{len(books)} steps={opt.step} "
              f"online_bpb={bits/positions:.6f} elapsed={time.perf_counter()-started:.1f}s",
              file=sys.stderr,flush=True)
    if queued:
        opt.update(grad/queued)
    return {"positions":positions,"new_updates":opt.step,
            "training_online_bpb":bits/positions,"new_theta":opt.theta.tolist(),
            "adam_m":opt.m.tolist(),"adam_v":opt.v.tolist(),
            "inherited_optimizer_groups":25,"new_optimizer_groups":3,
            "elapsed_s":time.perf_counter()-started}


def alternative(r26:np.ndarray,memory:WordSuccessorMemory,theta:np.ndarray,
                *,wrong:bool=False,query:WordSuccessorMemory|None=None)->np.ndarray:
    lex=pack_lexical(memory.features(r26,wrong_values=wrong,query=query))
    r29=lexical_readout(r26,lex,R29_THETA)
    return calibrated_readout(r29,class_features(lex,support(memory,query=query)),theta)


def evaluate(books:list[bytes],names:list[str],prior:np.ndarray,theta:np.ndarray)->dict:
    if len(books)<2 or len(books)!=len(names) or theta.shape!=(CALIBRATION_WEIGHTS,):
        raise ValueError("require named disjoint books and three coefficients")
    sums={arm:{"bits":0.,"nonspace":0.,"top1":0,"top5":0} for arm in ARMS}
    losses={arm:[] for arm in ARMS}
    count=nonspace=characters=0
    began=time.perf_counter()
    rows=[]
    for index,(name,book) in enumerate(zip(names,books)):
        model=M3R31Predictor(prior)
        unrelated=books[(index+1)%len(books)]
        matched=WordSuccessorMemory()
        full=WordSuccessorMemory()
        for b in unrelated:
            full.observe(int(b))
        warmup=scoring_start(book)
        characters+=len(book[warmup:].decode("utf-8","strict"))
        local={arm:[] for arm in ARMS}
        for pos,target in enumerate(book):
            if pos>=warmup:
                r26,lex=model.base_and_lexical_features()
                r29=lexical_readout(r26,lex,R29_THETA)
                evidence=support(model.lexical)
                r30=router_readout(r29,router_features(lex,evidence),R30_DELTA)
                r31=calibrated_readout(r29,class_features(lex,evidence),theta)
                predictions=(r29,r30,r31,
                             alternative(r26,model.lexical,theta,wrong=True),
                             alternative(r26,matched,theta,query=model.lexical),
                             alternative(r26,full,theta,query=model.lexical))
                for arm,lp in zip(ARMS,predictions):
                    value=-float(lp[target])/np.log(2)
                    local[arm].append(value)
                    sums[arm]["bits"]+=value
                    if target!=32:
                        sums[arm]["nonspace"]+=value
                    sums[arm]["top1"]+=int(np.argmax(lp)==target)
                    sums[arm]["top5"]+=int(target in np.argpartition(lp,-5)[-5:])
                count+=1
                nonspace+=int(target!=32)
            model.observe(int(target))
            if pos<len(unrelated):
                matched.observe(int(unrelated[pos]))
        row={"filename":name,"sha256":hashlib.sha256(book).hexdigest(),
             "positions":len(local["r29"])}
        for arm,v in local.items():
            losses[arm].append(np.asarray(v))
            row[arm+"_bpb"]=float(np.mean(v))
        row["gain_vs_r29"]=row["r29_bpb"]-row["r31_bpb"]
        rows.append(row)
        print(f"R31 EVAL {index+1}/{len(books)} "
              f"r29={row['r29_bpb']:.6f} r30={row['r30_bpb']:.6f} "
              f"r31={row['r31_bpb']:.6f}",file=sys.stderr,flush=True)
    measured={}
    for arm,v in sums.items():
        measured[arm]={"bpb":v["bits"]/count,"bpc":v["bits"]/characters,
                       "nonspace_bpb":v["nonspace"]/nonspace,
                       "top1":v["top1"]/count,"top5":v["top5"]/count,
                       "positions":count}
    for arm in ARMS[1:]:
        measured[arm]["gain_bpb_vs_r29"]=measured["r29"]["bpb"]-measured[arm]["bpb"]
        measured[arm].update(whole_book_deltas(np.concatenate(losses["r29"]),
                                              np.concatenate(losses[arm]),books,WARMUP))
    return {"arms":measured,"per_book":rows,"books":len(books),
            "positions":count,"characters":characters,
            "unrelated_control_matched_prefix":True,
            "elapsed_s":time.perf_counter()-began}


def gate_pass(data:dict,minimum_books:int)->bool:
    arms=data["arms"]
    base,trial=arms["r29"],arms["r31"]
    return bool(trial["bpb"]<min(base["bpb"],arms["r30"]["bpb"],
                                     arms["wrong_successor"]["bpb"],
                                     arms["unrelated_matched"]["bpb"])
                and trial["nonspace_bpb"]<=base["nonspace_bpb"]
                and trial["top1"]>=base["top1"]
                and trial["top5"]>=base["top5"]
                and trial["books_improved"]>=minimum_books)


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus-root",type=Path,required=True)
    p.add_argument("--parent-checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path)
    p.add_argument("--checkpoint",type=Path)
    args=p.parse_args()
    opt=FrozenR30Adam(args.parent_checkpoint)
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
    paths,books=resolve_books(args.corpus_root,BYTES_PER_BOOK,DEV[1],SEED)
    verify_manifest(paths[:52],books[:52],manifest)
    hashes=[hashlib.sha256(b).hexdigest() for b in books]
    if (len(set(hashes))!=DEV[1] or len(set(p.name for p in paths))!=DEV[1]
            or tuple(hashes[132:156])!=R21_TEST_HASHES):
        raise ValueError("duplicate or modified NaNROUTER corpus identities")
    prior=train_unigram_prior(books[:TRAIN_BOOKS])
    np.testing.assert_array_equal(prior,opt.prior)
    fit=train(books[FIT[0]:FIT[1]],prior,opt)
    names=[p.name for p in paths]
    gate=evaluate(books[GATE[0]:GATE[1]],names[GATE[0]:GATE[1]],prior,opt.theta)
    gate_ok=gate_pass(gate,minimum_books=3)
    dev=evaluate(books[DEV[0]:DEV[1]],names[DEV[0]:DEV[1]],prior,opt.theta)
    dev_ok=gate_ok and gate_pass(dev,minimum_books=9) and (
        dev["arms"]["r31"]["whole_book_bootstrap_95pct_bits"][0]>0)
    report={"architecture":"M3 R31 class-selective feedback on frozen R29/R30",
            "parent_checkpoint_sha256":PARENT_SHA,
            "parent_25_optimizer_groups_untouched":True,
            "fit_indices":list(FIT),"gate_indices":list(GATE),
            "dev_indices":list(DEV),"sealed_test_indices":list(SEALED_TEST),
            "sealed_test_read":False,"fit_sha256":hashes[FIT[0]:FIT[1]],
            "gate_sha256":hashes[GATE[0]:GATE[1]],
            "dev_sha256":hashes[DEV[0]:DEV[1]],
            "fit":fit,"gate":gate,"gate_pass":gate_ok,"dev":dev,
            "dev_gate_pass":bool(dev_ok),
            "promotion":"none; R29 remains default until full independent verification"}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if args.checkpoint:
        opt.save(args.checkpoint,hashes)
    print(json.dumps({"fit":fit,"gate_pass":gate_ok,"gate":gate["arms"],
                      "dev_gate_pass":bool(dev_ok),"dev":dev["arms"]},indent=2,sort_keys=True))


if __name__=="__main__":
    main()
