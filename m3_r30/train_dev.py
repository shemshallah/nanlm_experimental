"""M3 R30 CPU optimizer continuation from exact 22-state R29 checkpoint.

R30 TRAIN 356..371, GATE 372..375, DEV 376..387, TEST 388..411 sealed.
The causal unrelated-book control observes the SAME number of bytes as the
actual history. A separate full-unrelated-book arm records the historical
control with greater context support. No future target is accessed by
either matched history or the candidate before scoring its current byte.
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
from m3_r30.router import (M3R30Predictor, ROUTER_FEATURES,
                           router_features, router_readout, support)

R29_PARENT_HASH = "02c340e82ec77ae2c6e1d20ddd808ca4f63e3c242fe6fc94ff8f2235d8ab9386"
FIT = (356, 372)
GATE = (372, 376)
DEV = (376, 388)
SEALED_TEST = (388, 412)
LR, BATCH = .004, 64
ARMS = ("r29", "r30", "wrong_successor", "unrelated_matched", "unrelated_full")


class FrozenR29Adam:
    """Explicitly retain all 22 parent moments/ages; train three new groups."""

    def __init__(self, checkpoint: Path):
        if not checkpoint.is_file() or digest(checkpoint) != R29_PARENT_HASH:
            raise ValueError("complete R29 checkpoint missing or altered")
        with np.load(checkpoint, allow_pickle=False) as x:
            self.parent_theta = np.asarray(x["theta"], dtype=np.float64).copy()
            self.parent_m = np.asarray(x["adam_m"], dtype=np.float64).copy()
            self.parent_v = np.asarray(x["adam_v"], dtype=np.float64).copy()
            self.parent_steps = np.asarray(x["adam_steps"], dtype=np.int64).copy()
            self.prior = np.asarray(x["prior"], dtype=np.float64).copy()
            self.parent_fit = tuple(x["fit_indices"].tolist())
        if (self.parent_theta.shape != (22,) or self.parent_m.shape != (22,)
                or self.parent_v.shape != (22,) or self.parent_steps.shape != (22,)
                or self.prior.shape != (256,) or self.parent_fit != (300, 316)
                or self.parent_steps.tolist() != [1536]*3 + [768]*19
                or not np.array_equal(self.parent_theta[:14], R26_THETA)
                or not np.array_equal(self.parent_theta[14:], R29_THETA)):
            raise ValueError("R29 trained readout, prior, or optimizer history modified")
        self.delta = np.zeros(ROUTER_FEATURES, dtype=np.float64)
        self.m = np.zeros_like(self.delta)
        self.v = np.zeros_like(self.delta)
        self.step = 0

    def update(self, gradient: np.ndarray) -> None:
        if gradient.shape != (ROUTER_FEATURES,):
            raise ValueError("incorrect three-gain gradient")
        self.step += 1
        g = np.clip(gradient, -2, 2)
        self.m = .9*self.m+.1*g
        self.v = .999*self.v+.001*g*g
        self.delta -= LR*(self.m/(1-.9**self.step))/(np.sqrt(self.v/(1-.999**self.step))+1e-8)
        np.clip(self.delta, -1., 1., out=self.delta)

    def save(self, checkpoint: Path, hashes: list[str]) -> None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        np.savez(checkpoint,
                 theta=np.concatenate((self.parent_theta,self.delta)),
                 adam_m=np.concatenate((self.parent_m,self.m)),
                 adam_v=np.concatenate((self.parent_v,self.v)),
                 adam_steps=np.concatenate((self.parent_steps,np.full(3,self.step))),
                 prior=self.prior,parent_checkpoint_sha256=R29_PARENT_HASH,
                 fit_indices=np.array(FIT),fit_sha256=np.array(hashes[FIT[0]:FIT[1]]),
                 gate_indices=np.array(GATE),dev_indices=np.array(DEV),
                 sealed_test_indices=np.array(SEALED_TEST))


def train(books: list[bytes], prior: np.ndarray, optimizer: FrozenR29Adam) -> dict:
    np.testing.assert_array_equal(prior,optimizer.prior)
    gradient = np.zeros(ROUTER_FEATURES)
    queued = positions = 0
    bits = 0.
    began = time.perf_counter()
    for index,book in enumerate(books):
        model = M3R30Predictor(prior)
        warmup = scoring_start(book)
        for pos,target in enumerate(book):
            if pos >= warmup:
                base,features = model.r29_base_and_router_features()
                logp = router_readout(base,features,optimizer.delta)
                error=np.exp(logp)
                error[target]-=1
                gradient += (features@error) + .0001*optimizer.delta
                queued += 1
                positions += 1
                bits -= float(logp[target])/np.log(2)
                if queued == BATCH:
                    optimizer.update(gradient/queued)
                    gradient.fill(0)
                    queued=0
            model.observe(int(target))
        print(f"R30 FIT {index+1}/{len(books)} step={optimizer.step} "
              f"online_bpb={bits/positions:.6f} elapsed={time.perf_counter()-began:.1f}s",
              file=sys.stderr,flush=True)
    if queued:
        optimizer.update(gradient/queued)
    return {"scored_positions":positions,"new_updates":optimizer.step,
            "online_bpb":bits/positions,"new_delta":optimizer.delta.tolist(),
            "adam_m":optimizer.m.tolist(),"adam_v":optimizer.v.tolist(),
            "inherited_groups_preserved":22,"new_groups":ROUTER_FEATURES,
            "elapsed_s":time.perf_counter()-began}


def alternate(model: M3R30Predictor, r26: np.ndarray,
              memory: WordSuccessorMemory, delta: np.ndarray,
              *,wrong:bool=False, query:WordSuccessorMemory|None=None) -> np.ndarray:
    lex=pack_lexical(memory.features(r26,wrong_values=wrong,query=query))
    r29=lexical_readout(r26,lex,R29_THETA)
    features=router_features(lex,support(memory,query=query))
    return router_readout(r29,features,delta)


def evaluate(books:list[bytes],names:list[str],prior:np.ndarray,
             delta:np.ndarray) -> dict:
    if len(books)<2 or len(books)!=len(names) or delta.shape!=(ROUTER_FEATURES,):
        raise ValueError("at least two disjoint books and three gains required")
    scores={a:{"bits":0.,"nonspace":0.,"top1":0,"top5":0} for a in ARMS}
    losses={a:[] for a in ARMS}
    count=nonspace=characters=0
    began=time.perf_counter()
    book_rows=[]
    for index,(name,book) in enumerate(zip(names,books)):
        model=M3R30Predictor(prior)
        other=books[(index+1)%len(books)]
        matched=WordSuccessorMemory()
        full=WordSuccessorMemory()
        for b in other:
            full.observe(int(b))
        start=scoring_start(book)
        characters+=len(book[start:].decode("utf-8","strict"))
        local={arm:[] for arm in ARMS}
        for pos,target in enumerate(book):
            if pos>=start:
                r26,lex=model.base_and_lexical_features()
                r29=lexical_readout(r26,lex,R29_THETA)
                features=router_features(lex,support(model.lexical))
                r30=router_readout(r29,features,delta)
                predictions=(r29,r30,
                             alternate(model,r26,model.lexical,delta,wrong=True),
                             alternate(model,r26,matched,delta,query=model.lexical),
                             alternate(model,r26,full,delta,query=model.lexical))
                for arm,lp in zip(ARMS,predictions):
                    value=-float(lp[target])/np.log(2)
                    local[arm].append(value)
                    scores[arm]["bits"]+=value
                    if target!=32:
                        scores[arm]["nonspace"]+=value
                    scores[arm]["top1"]+=int(np.argmax(lp)==target)
                    scores[arm]["top5"]+=int(target in np.argpartition(lp,-5)[-5:])
                count+=1
                nonspace+=int(target!=32)
            # Both timelines receive exactly one observed byte AFTER scoring.
            model.observe(int(target))
            if pos<len(other):
                matched.observe(int(other[pos]))
        row={"filename":name,"sha256":hashlib.sha256(book).hexdigest(),
             "positions":len(local["r29"])}
        for arm,vals in local.items():
            losses[arm].append(np.asarray(vals))
            row[arm+"_bpb"]=float(np.mean(vals))
        row["gain_vs_r29"]=row["r29_bpb"]-row["r30_bpb"]
        book_rows.append(row)
        print(f"R30 EVAL {index+1}/{len(books)} "
              f"r29={row['r29_bpb']:.6f} r30={row['r30_bpb']:.6f}",
              file=sys.stderr,flush=True)
    measured={}
    for arm,v in scores.items():
        measured[arm]={"bpb":v["bits"]/count,
                       "bpc":v["bits"]/characters,
                       "nonspace_bpb":v["nonspace"]/nonspace,
                       "top1":v["top1"]/count,"top5":v["top5"]/count,
                       "positions":count}
    for arm in ARMS[1:]:
        measured[arm]["gain_bpb_vs_r29"]=measured["r29"]["bpb"]-measured[arm]["bpb"]
        measured[arm].update(whole_book_deltas(np.concatenate(losses["r29"]),
                                              np.concatenate(losses[arm]),books,WARMUP))
    return {"arms":measured,"per_book":book_rows,"books":len(books),
            "positions":count,"characters":characters,
            "elapsed_s":time.perf_counter()-began,
            "unrelated_control_matched_prefix":True}


def gate_pass(data:dict,minimum_books:int)->bool:
    a=data["arms"]
    parent,candidate=a["r29"],a["r30"]
    return bool(candidate["bpb"]<min(parent["bpb"],a["wrong_successor"]["bpb"],
                                     a["unrelated_matched"]["bpb"])
                and candidate["nonspace_bpb"]<=parent["nonspace_bpb"]
                and candidate["top1"]>=parent["top1"]
                and candidate["top5"]>=parent["top5"]
                and candidate["books_improved"]>=minimum_books)


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus-root",type=Path,required=True)
    p.add_argument("--parent-checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path)
    p.add_argument("--checkpoint",type=Path)
    args=p.parse_args()
    optimizer=FrozenR29Adam(args.parent_checkpoint)
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
    paths,books=resolve_books(args.corpus_root,BYTES_PER_BOOK,DEV[1],SEED)
    verify_manifest(paths[:52],books[:52],manifest)
    hashes=[hashlib.sha256(b).hexdigest() for b in books]
    if (len(set(hashes))!=DEV[1] or len(set(p.name for p in paths))!=DEV[1]
            or tuple(hashes[132:156])!=R21_TEST_HASHES):
        raise ValueError("NaNROUTER ancestry or disjoint book split changed")
    prior=train_unigram_prior(books[:TRAIN_BOOKS])
    np.testing.assert_array_equal(prior,optimizer.prior)
    fit=train(books[FIT[0]:FIT[1]],prior,optimizer)
    names=[p.name for p in paths]
    gate=evaluate(books[GATE[0]:GATE[1]],names[GATE[0]:GATE[1]],prior,optimizer.delta)
    gate_ok=gate_pass(gate,minimum_books=3)
    dev=evaluate(books[DEV[0]:DEV[1]],names[DEV[0]:DEV[1]],prior,optimizer.delta)
    dev_ok=gate_ok and gate_pass(dev,minimum_books=9) and (
        dev["arms"]["r30"]["whole_book_bootstrap_95pct_bits"][0]>0)
    result={"architecture":"M3 R30 selective lexical trust on frozen R29",
            "parent_checkpoint_sha256":R29_PARENT_HASH,
            "inherited_optimizer_untouched":True,"fit_indices":list(FIT),
            "gate_indices":list(GATE),"dev_indices":list(DEV),
            "sealed_test_indices":list(SEALED_TEST),"sealed_test_read":False,
            "fit_sha256":hashes[FIT[0]:FIT[1]],
            "gate_sha256":hashes[GATE[0]:GATE[1]],
            "dev_sha256":hashes[DEV[0]:DEV[1]],
            "fit":fit,"gate":gate,"gate_pass":gate_ok,
            "dev":dev,"dev_gate_pass":bool(dev_ok),
            "promotion":"none; R29 remains user-directed default"}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if args.checkpoint:
        optimizer.save(args.checkpoint,hashes)
    print(json.dumps({"fit":fit,"gate_pass":gate_ok,"gate":gate["arms"],
                      "dev_gate_pass":bool(dev_ok),"dev":dev["arms"]},indent=2,sort_keys=True))


if __name__=="__main__":
    main()
