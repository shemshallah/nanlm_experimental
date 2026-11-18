"""M3 R32 CPU training and independent book-disjoint history controls.

Background 468..483, FIT 484..499, GATE 500..503, DEV 504..515; TEST
516..539 remains sealed. No background, training, or evaluation arm reads a
pending target before scoring that target. All 28 inherited R31 groups are
copied exactly; four new Adam moments/ages are stored in the checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

from m3_default.predictor import R26_THETA, R29_THETA, R30_DELTA, R31_THETA
from m3_r14.predictive_probe import train_unigram_prior
from m3_r16.counterfactual_wave_gate import whole_book_deltas
from m3_r17.locked_test import SEED, TRAIN_BOOKS, BYTES_PER_BOOK, WARMUP
from m3_r18.locked_replication import MANIFEST, verify_manifest
from m3_r22.evaluate import R21_TEST_HASHES
from m3_r23.train_dev import scoring_start
from m3_r27.locked_test import digest
from m3_r29.lexical import WordSuccessorMemory, lexical_readout
from m3_r30.router import support
from m3_r31.calibration import class_features, calibrated_readout
from m3_r32.anchor import (GAIN_COUNT, M3R32Predictor, anchor_features,
                            anchored_readout, make_anchor)

PARENT_SHA = "dd8fbaa93a17e5568bf2dc774b5599e5761824dee74db8f71242039f743a12ff"
BACKGROUND = (468, 484)
FIT = (484, 500)
GATE = (500, 504)
DEV = (504, 516)
SEALED_TEST = (516, 540)
ARMS = ("r29", "r31", "r32", "wrong_anchor", "unrelated_matched", "unrelated_full")
BATCH = 64
LR = .003


def selective_books(root:Path, count:int, indices:set[int])->tuple[list[Path],dict[int,bytes]]:
    """Use R14's exact seeded path order; never open sealed predecessor TESTs.

    The legacy resolve_books helper opens every selected book from index zero
    onward. R32 instead opens only original TRAIN ancestry, R21 hash ancestry,
    and its own new splits. In particular R31 TEST 444..467 is not read.
    """
    files=sorted(p for p in root.rglob("*.txt") if p.is_file() and p.stat().st_size>=BYTES_PER_BOOK)
    random.Random(SEED).shuffle(files)
    if len(files)<count:
        raise ValueError(f"need {count} sufficiently long books; found {len(files)}")
    selected=files[:count]
    loaded={}
    for i in sorted(indices):
        if not 0<=i<count:
            raise ValueError("selected index exceeds book count")
        data=selected[i].read_bytes()[:BYTES_PER_BOOK]
        while data:
            try:
                data.decode("utf-8","strict")
                break
            except UnicodeDecodeError as exc:
                if exc.start<len(data)-4:
                    raise ValueError(f"invalid UTF-8 inside {selected[i].name}") from exc
                data=data[:exc.start]
        loaded[i]=data
    return selected,loaded


class FrozenR31Adam:
    def __init__(self, path: Path):
        if not path.is_file() or digest(path) != PARENT_SHA:
            raise ValueError("missing or altered R31 full checkpoint")
        with np.load(path,allow_pickle=False) as x:
            self.parent_theta = np.asarray(x["theta"],dtype=np.float64).copy()
            self.parent_m = np.asarray(x["adam_m"],dtype=np.float64).copy()
            self.parent_v = np.asarray(x["adam_v"],dtype=np.float64).copy()
            self.parent_steps = np.asarray(x["adam_steps"],dtype=np.int64).copy()
            self.prior = np.asarray(x["prior"],dtype=np.float64).copy()
            self.parent_fit = tuple(x["fit_indices"].tolist())
        if (self.parent_theta.shape != (28,) or self.parent_m.shape != (28,)
                or self.parent_v.shape != (28,) or self.parent_steps.shape != (28,)
                or self.prior.shape != (256,) or self.parent_fit != (412,428)
                or self.parent_steps.tolist() != [1536]*3+[768]*25
                or not np.array_equal(self.parent_theta[:14],R26_THETA)
                or not np.array_equal(self.parent_theta[14:22],R29_THETA)
                or not np.array_equal(self.parent_theta[22:25],R30_DELTA)
                or not np.array_equal(self.parent_theta[25:],R31_THETA)):
            raise ValueError("R31 weights or inherited optimizer history modified")
        self.theta = np.zeros(GAIN_COUNT,dtype=np.float64)
        self.m = np.zeros_like(self.theta)
        self.v = np.zeros_like(self.theta)
        self.step = 0

    def update(self,g:np.ndarray)->None:
        if g.shape != (GAIN_COUNT,) or not np.isfinite(g).all():
            raise ValueError("four finite R32 gradients required")
        self.step += 1
        grad=np.clip(g,-2.,2.)
        self.m=.9*self.m+.1*grad
        self.v=.999*self.v+.001*grad*grad
        self.theta-=LR*(self.m/(1-.9**self.step))/(np.sqrt(self.v/(1-.999**self.step))+1e-8)
        np.clip(self.theta,-1.5,1.5,out=self.theta)

    def save(self,path:Path,hashes:list[str],background:list[bytes])->None:
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez(path,
            theta=np.concatenate((self.parent_theta,self.theta)),
            adam_m=np.concatenate((self.parent_m,self.m)),
            adam_v=np.concatenate((self.parent_v,self.v)),
            adam_steps=np.concatenate((self.parent_steps,np.full(GAIN_COUNT,self.step))),
            prior=self.prior,
            parent_checkpoint_sha256=PARENT_SHA,
            background_indices=np.array(BACKGROUND),
            background_lengths=np.array([len(x) for x in background],dtype=np.int32),
            background_bytes=np.frombuffer(b"".join(background),dtype=np.uint8).copy(),
            background_sha256=np.array(hashes[BACKGROUND[0]:BACKGROUND[1]]),
            fit_indices=np.array(FIT),fit_sha256=np.array(hashes[FIT[0]:FIT[1]]),
            gate_indices=np.array(GATE),dev_indices=np.array(DEV),
            sealed_test_indices=np.array(SEALED_TEST))


def train(books:list[bytes],prior:np.ndarray,anchor:WordSuccessorMemory,
          opt:FrozenR31Adam)->dict:
    np.testing.assert_array_equal(prior,opt.prior)
    g=np.zeros(GAIN_COUNT,dtype=np.float64)
    batch=positions=0
    bits=0.
    start=time.perf_counter()
    for i,book in enumerate(books):
        model=M3R32Predictor(prior,anchor)
        warmup=scoring_start(book)
        for pos,target in enumerate(book):
            if pos>=warmup:
                base,feat=model.r31_base_and_anchor_features()
                lp=anchored_readout(base,feat,opt.theta)
                e=np.exp(lp)
                e[target]-=1.
                g+=feat@e+.0001*opt.theta
                bits-=float(lp[target])/np.log(2)
                batch+=1
                positions+=1
                if batch==BATCH:
                    opt.update(g/batch)
                    g.fill(0.)
                    batch=0
            model.observe(int(target))
        print(f"R32 FIT {i+1}/{len(books)} steps={opt.step} "
              f"online_bpb={bits/positions:.6f} elapsed={time.perf_counter()-start:.1f}s",
              file=sys.stderr,flush=True)
    if batch:
        opt.update(g/batch)
    return {"positions":positions,"updates":opt.step,
            "online_bpb":bits/positions,"theta":opt.theta.tolist(),
            "adam_m":opt.m.tolist(),"adam_v":opt.v.tolist(),
            "inherited_optimizer_groups":28,"new_optimizer_groups":GAIN_COUNT,
            "elapsed_s":time.perf_counter()-start}


def evaluate(books:list[bytes],names:list[str],prior:np.ndarray,
             anchor:WordSuccessorMemory,theta:np.ndarray)->dict:
    if len(books)<2 or len(books)!=len(names) or theta.shape!=(GAIN_COUNT,):
        raise ValueError("at least two named books and four gains required")
    sums={arm:{"bits":0.,"nonspace":0.,"top1":0,"top5":0} for arm in ARMS}
    losses={arm:[] for arm in ARMS}
    count=nonspace=characters=0
    start=time.perf_counter()
    rows=[]
    for index,(name,book) in enumerate(zip(names,books)):
        model=M3R32Predictor(prior,anchor,theta)
        other=books[(index+1)%len(books)]
        matched=WordSuccessorMemory()
        full=WordSuccessorMemory()
        for b in other:
            full.observe(int(b))
        warmup=scoring_start(book)
        characters+=len(book[warmup:].decode("utf-8","strict"))
        local={arm:[] for arm in ARMS}
        for pos,target in enumerate(book):
            if pos>=warmup:
                r26,local_lex=model.base_and_lexical_features()
                r29=lexical_readout(r26,local_lex,R29_THETA)
                r31=calibrated_readout(r29,class_features(local_lex,support(model.lexical)),R31_THETA)
                f=anchor_features(r26,local_lex,model.lexical,anchor)
                predictions=(r29,r31,anchored_readout(r31,f,theta),
                             anchored_readout(r31,anchor_features(r26,local_lex,model.lexical,
                                                                 anchor,wrong_values=True),theta),
                             anchored_readout(r31,anchor_features(r26,local_lex,model.lexical,
                                                                 matched),theta),
                             anchored_readout(r31,anchor_features(r26,local_lex,model.lexical,
                                                                 full),theta))
                for arm,lp in zip(ARMS,predictions):
                    bits=-float(lp[target])/np.log(2)
                    local[arm].append(bits)
                    sums[arm]["bits"]+=bits
                    if target!=32:
                        sums[arm]["nonspace"]+=bits
                    sums[arm]["top1"]+=int(np.argmax(lp)==target)
                    sums[arm]["top5"]+=int(target in np.argpartition(lp,-5)[-5:])
                count+=1
                nonspace+=int(target!=32)
            model.observe(int(target))
            if pos<len(other):
                matched.observe(int(other[pos]))
        row={"filename":name,"sha256":hashlib.sha256(book).hexdigest(),
             "positions":len(local["r29"])}
        for arm,vals in local.items():
            losses[arm].append(np.asarray(vals))
            row[arm+"_bpb"]=float(np.mean(vals))
        row["gain_vs_r31"]=row["r31_bpb"]-row["r32_bpb"]
        rows.append(row)
        print(f"R32 EVAL {index+1}/{len(books)} "
              f"r29={row['r29_bpb']:.6f} r31={row['r31_bpb']:.6f} "
              f"r32={row['r32_bpb']:.6f}",file=sys.stderr,flush=True)
    metrics={}
    for arm,counts in sums.items():
        metrics[arm]={"bpb":counts["bits"]/count,
                      "bpc":counts["bits"]/characters,
                      "nonspace_bpb":counts["nonspace"]/nonspace,
                      "top1":counts["top1"]/count,
                      "top5":counts["top5"]/count,"positions":count}
    for arm in ARMS[2:]:
        metrics[arm]["gain_bpb_vs_r31"]=metrics["r31"]["bpb"]-metrics[arm]["bpb"]
        metrics[arm].update(whole_book_deltas(np.concatenate(losses["r31"]),
                                               np.concatenate(losses[arm]),books,WARMUP))
    return {"arms":metrics,"per_book":rows,"books":len(books),
            "positions":count,"characters":characters,
            "elapsed_s":time.perf_counter()-start,"anchor_is_fit_only":True}


def gate_pass(data:dict,minimum_books:int)->bool:
    a=data["arms"]
    old,new=a["r31"],a["r32"]
    return bool(new["bpb"]<min(a["r29"]["bpb"],old["bpb"],
                                 a["wrong_anchor"]["bpb"],a["unrelated_matched"]["bpb"])
                and new["nonspace_bpb"]<=old["nonspace_bpb"]
                and new["top1"]>=max(old["top1"],a["r29"]["top1"])
                and new["top5"]>=old["top5"]
                and new["books_improved"]>=minimum_books)


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus-root",type=Path,required=True)
    p.add_argument("--parent-checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path)
    p.add_argument("--checkpoint",type=Path)
    args=p.parse_args()
    opt=FrozenR31Adam(args.parent_checkpoint)
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
    needed=(set(range(TRAIN_BOOKS))|set(range(132,156))|
            set(range(BACKGROUND[0],DEV[1])))
    paths,loaded=selective_books(args.corpus_root,DEV[1],needed)
    hashes=[hashlib.sha256(loaded[i]).hexdigest() if i in loaded else "UNREAD"
            for i in range(DEV[1])]
    verify_manifest(paths[:52],[loaded[i] for i in range(52)],manifest)
    if (len({hashes[i] for i in needed})!=len(needed)
            or len({paths[i].name for i in needed})!=len(needed)
            or tuple(hashes[132:156])!=R21_TEST_HASHES):
        raise ValueError("NaNROUTER identities or seeded splits changed")
    prior=train_unigram_prior([loaded[i] for i in range(TRAIN_BOOKS)])
    np.testing.assert_array_equal(prior,opt.prior)
    anchor_books=[loaded[i] for i in range(*BACKGROUND)]
    anchor=make_anchor(anchor_books)
    trained=train([loaded[i] for i in range(*FIT)],prior,anchor,opt)
    names=[p.name for p in paths]
    gate=evaluate([loaded[i] for i in range(*GATE)],names[GATE[0]:GATE[1]],prior,anchor,opt.theta)
    gate_ok=gate_pass(gate,3)
    dev=evaluate([loaded[i] for i in range(*DEV)],names[DEV[0]:DEV[1]],prior,anchor,opt.theta)
    dev_ok=gate_ok and gate_pass(dev,9) and dev["arms"]["r32"]["whole_book_bootstrap_95pct_bits"][0]>0
    report={"architecture":"M3 R32 train-only immutable anchor lexical memory",
            "parent_checkpoint_sha256":PARENT_SHA,
            "inherited_28_optimizer_groups_unchanged":True,
            "background_indices":list(BACKGROUND),"fit_indices":list(FIT),
            "gate_indices":list(GATE),"dev_indices":list(DEV),
            "sealed_test_indices":list(SEALED_TEST),"sealed_test_read":False,
            "r31_sealed_test_indices":[444,468],"r31_sealed_test_read":False,
            "background_sha256":hashes[BACKGROUND[0]:BACKGROUND[1]],
            "fit_sha256":hashes[FIT[0]:FIT[1]],
            "gate_sha256":hashes[GATE[0]:GATE[1]],
            "dev_sha256":hashes[DEV[0]:DEV[1]],
            "fit":trained,"gate":gate,"gate_pass":gate_ok,"dev":dev,
            "dev_gate_pass":bool(dev_ok),
            "promotion":"none; R31 remains the directed default until independent TEST"}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if args.checkpoint:
        opt.save(args.checkpoint,hashes,anchor_books)
    print(json.dumps({"fit":trained,"gate_pass":gate_ok,"gate":gate["arms"],
                      "dev_gate_pass":bool(dev_ok),"dev":dev["arms"]},sort_keys=True,indent=2))


if __name__=="__main__":
    main()
