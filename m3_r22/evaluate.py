"""M3 R22: fresh-book DEV and capacity-matched conventional n-gram audit.

R21's TEST documents 132..155 remain sealed to R22 fitting and DEV.
Only new seeded NaNROUTER books 156..187 can select or evaluate R22.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np

from m3_r14.predictive_probe import resolve_books, train_unigram_prior, metrics
from m3_r16.counterfactual_wave_gate import whole_book_deltas
from m3_r17.locked_test import SEED, BYTES_PER_BOOK, WARMUP, TRAIN_BOOKS
from m3_r18.locked_replication import MANIFEST, verify_manifest
from m3_r20.signed_mirror_memory import SignedMirrorSuccessorMemory
from m3_r20.successor_dev import collect_candidate, combined_logp, SHIFT
from m3_r22.independent_feedback import (
    ConventionalCountNgram, feedback_at_prefix, candidate_logp,
)

FIT_START, GATE_START, DEV_START, TOTAL_BOOKS = 156, 172, 176, 188
R21_TEST_HASHES = ("8779fbc3e5e1688e910cbbf20c17e18d3a82676b66259ac46ff3f654aaa53e54","9bcadd1cd395cdc8f1951b35d38f8f95bb3dab6407799a7db8fcdd4dca4edda3","0692fbdaaa40d31da128d80f43655d14010c6eb2f67d5cb8a4822a84e5ed8a28","91ed539170f64391495037c8609779043676d88f7174a989074a97691af0c7e6","30afca55980f7b00008d1189e89a98a3f847f1a4c8582118f71dbc9d716885c4","8f4015a336709beaae4be485df2c3fd10d0b8340c235837c868753d5830be7fd","7daf0023673e9cb9c00ee464400c08d9b437ec5d41ab592b9f60636ccbb09ece","adef96e53a6f44b422537093e3c6769d0da636b27ca057c601675a53150e6f1e","bc27552abdc6db5df8bd64772937a71168b2409327aa47a5c702bcdd03e3361b","4141f1471eec0421cb41f0136dfb016c86608c28e7b68f5faa554c8e6ac06f36","ba438a1ffc10714daf72d82b3059384925419862a2f054989a89f8009c83d4dd","76c924be80710e1426a08c0a877680e0c50e63809740715f08b2a3829efcfa86","19c9de9f89c278dde3dc78decf89fbfc5325897f7aca0d8b238038bfed811a0f","7a3e73ecf3095192fa9c3a60e89e4dcb3e9b5352186228241a78818582fe07da","7ac43fbd7c3776734711715686072d78d086f9ca8e00905b15bfb54d976a3ff0","640935da44f871e182f614e555a27e4f2e4dc21c8de82582090895873c85ed16","20d89da5de32a27484dec8f3ac6b5d623971df8d0444ab646cf453333bf120d9","bc8320787ca998ba618a4266de144d8bfad8e4475509df1ec6244de387aba6dd","7514bed950a15369c65871eda25053d8816f96ad899ce1871bf54c2f8fea8666","7925198dc00f15ec1dadc05a985ec94905d81e52ef5e845aa3e5e04b58e02560","bcdb1ae483896169388b4a86d30a0dc36d556d627eb347c15532d76ca225d29f","d5ecf850dcef900418f99fe1300f8383735209f5b725ea6a03fbd7ee83678c69","acbd34927bd43792722cc1484179ec62a5076f253bd5a446f8cc8e63660ed2b0","e022f99c9ecfbe02dce21d078bead38226a272f485a2011c3bcdccb8091d69ad",)
GRID = tuple(itertools.product((-0.5,0.0,0.5),(-0.5,0.0,0.5),(-0.25,0.0,0.25)))
ARMS = ("r19","r20_default","conventional_trigram","conventional_mixture",
        "adaptive","without_flux","wrong_flux","wrong_pair")


def collect_book(book: bytes, prior: np.ndarray) -> dict:
    data, r19, pair_delta, tri_delta, invariants = collect_candidate(book,prior)
    r20 = combined_logp(r19,pair_delta,tri_delta,0.5,0.5)
    geo, std = SignedMirrorSuccessorMemory(prior), ConventionalCountNgram(prior)
    keys = ("pair_residual","triangle_residual","flux")
    features = {key:[] for key in keys}
    conv_tri = []
    max_ngram = max_frozen = 0.0
    start = WARMUP
    while start<len(book) and (book[start]&0xC0)==0x80:
        start+=1
    for pos in range(1,len(book)):
        a,b = book[pos-1],book[pos]
        p = book[pos-2] if pos>=2 else None
        if pos>=start:
            row=pos-start
            f=feedback_at_prefix(geo,p,a,data["base_logp"][row])
            std_pair,std_tri=std.predict(p,a)
            max_ngram=max(max_ngram,float(np.max(np.abs(std_pair-f["pair"]))),
                          float(np.max(np.abs(std_tri-f["triangle"]))))
            max_frozen=max(max_frozen,
                float(np.max(np.abs(pair_delta[row]-
                                  (np.log(f["pair"])-data["base_logp"][row])))),
                float(np.max(np.abs(tri_delta[row]-
                                  (np.log(f["triangle"])-np.log(f["pair"]))))))
            for key in keys:
                features[key].append(f[key])
            conv_tri.append(np.log(std_tri))
        geo.observe(p,a,b)
        std.observe(p,a,b)
    if max(max_ngram,max_frozen)>2e-10:
        raise ValueError("count-isomorphism or frozen R20 alignment failed")
    if len(conv_tri)!=len(data["targets"]):
        raise ValueError("R22 byte-score alignment failed")
    if not invariants["pair_nonnegative"]:
        raise ValueError("R20 conservation invariant broke")
    return {"data":data,"r19":r19,"r20":r20,"conventional_trigram":np.asarray(conv_tri),
            **{key:np.asarray(features[key]) for key in keys},
            "ngram_max_abs_diff":max_ngram,"frozen_alignment_max_abs_diff":max_frozen}


def score_grid(books: list[bytes], prior: np.ndarray) -> list[dict]:
    scores={k:[0.0,0.0] for k in GRID}
    base=[0.0,0.0]
    n=n_nonspace=0
    for book in books:
        x=collect_book(book,prior)
        data=x["data"]
        target=data["targets"]
        rows=np.arange(len(target))
        mask=data["nonspace"]
        b=-x["r20"][rows,target]/np.log(2)
        base[0]+=float(b.sum())
        base[1]+=float(b[mask].sum())
        n+=len(target)
        n_nonspace+=int(mask.sum())
        for scales,total in scores.items():
            lp=candidate_logp(x["r20"],x["pair_residual"],
                              x["triangle_residual"],x["flux"],scales)
            bits=-lp[rows,target]/np.log(2)
            total[0]+=float(bits.sum())
            total[1]+=float(bits[mask].sum())
    return [{"scales":list(k),"bpb":v[0]/n,"nonspace_bpb":v[1]/n_nonspace,
             "gain_bpb_vs_r20":(base[0]-v[0])/n,
             "nonspace_gain_bpb_vs_r20":(base[1]-v[1])/n_nonspace}
            for k,v in scores.items()]


def select_train_only(fit: list[dict], gate: list[dict]) -> dict:
    if len(fit)!=len(GRID) or len(gate)!=len(GRID):
        raise ValueError("incomplete TRAIN-only scale grid")
    eligible=[r for r in fit if r["gain_bpb_vs_r20"]>=-1e-12 and
              r["nonspace_gain_bpb_vs_r20"]>=-1e-12]
    shortlist={tuple(r["scales"]) for r in sorted(
        eligible,key=lambda x:-x["gain_bpb_vs_r20"])[:6]}
    shortlist.add((0.0,0.0,0.0))
    valid=[r for r in gate if tuple(r["scales"]) in shortlist and
           r["gain_bpb_vs_r20"]>=-1e-12 and
           r["nonspace_gain_bpb_vs_r20"]>=-1e-12]
    if not valid:
        raise ValueError("zero-residual R20 fallback missing")
    winner=min(valid,key=lambda x:(x["bpb"],sum(abs(z) for z in x["scales"])))
    return {"scales":winner["scales"],"gate_bpb":winner["bpb"],
            "gate_gain_bpb_vs_r20":winner["gain_bpb_vs_r20"],
            "fit_shortlist":[list(x) for x in sorted(shortlist)]}


def evaluate_dev(books: list[bytes], names: list[str], prior: np.ndarray,
                 scales: tuple[float,float,float]) -> dict:
    if len(books)!=12 or len(names)!=12:
        raise ValueError("R22 requires 12 fresh DEV books")
    total={arm:{"bits":0.0,"nonspace":0.0,"top1":0.0,"top5":0.0} for arm in ARMS}
    loss_vectors={arm:[] for arm in ARMS}
    per_book=[]
    positions=nonspace=characters=0
    max_ngram=max_frozen=0.0
    for book,name in zip(books,names):
        x=collect_book(book,prior)
        data=x["data"]
        scales_no_flux=(scales[0],scales[1],0.0)
        lp={
            "r19":x["r19"],
            "r20_default":x["r20"],
            "conventional_trigram":x["conventional_trigram"],
            "conventional_mixture":x["r20"],  # exact isomorphism, separately checked
            "adaptive":candidate_logp(x["r20"],x["pair_residual"],
                        x["triangle_residual"],x["flux"],scales),
            "without_flux":candidate_logp(x["r20"],x["pair_residual"],
                        x["triangle_residual"],x["flux"],scales_no_flux),
            "wrong_flux":candidate_logp(x["r20"],x["pair_residual"],
                        x["triangle_residual"],np.roll(x["flux"],SHIFT,axis=1),scales),
            "wrong_pair":candidate_logp(x["r20"],np.roll(x["pair_residual"],SHIFT,axis=1),
                        x["triangle_residual"],x["flux"],scales),
        }
        n=len(data["targets"])
        positions+=n
        nonspace+=int(data["nonspace"].sum())
        characters+=data["unicode_chars"]
        max_ngram=max(max_ngram,x["ngram_max_abs_diff"])
        max_frozen=max(max_frozen,x["frozen_alignment_max_abs_diff"])
        row={"filename":name,"sha256":hashlib.sha256(book).hexdigest(),"positions":n}
        for arm,probs in lp.items():
            bits=-probs[np.arange(n),data["targets"]]/np.log(2)
            loss_vectors[arm].append(bits)
            metric=metrics(probs,data)
            total[arm]["bits"]+=float(bits.sum())
            total[arm]["nonspace"]+=float(bits[data["nonspace"]].sum())
            total[arm]["top1"]+=metric["top1"]*n
            total[arm]["top5"]+=metric["top5"]*n
            row[arm+"_bpb"]=float(bits.mean())
        row["gain_bpb_vs_r20"]=row["r20_default_bpb"]-row["adaptive_bpb"]
        per_book.append(row)
    measured={}
    for arm,vals in total.items():
        measured[arm]={"positions":positions,"bpb":vals["bits"]/positions,
                       "bpc":vals["bits"]/characters,"nonspace_bpb":vals["nonspace"]/nonspace,
                       "top1":vals["top1"]/positions,"top5":vals["top5"]/positions}
    baseline=measured["r20_default"]
    for arm in ARMS:
        if arm!="r20_default":
            measured[arm]["gain_bpb_vs_r20"]=baseline["bpb"]-measured[arm]["bpb"]
            measured[arm]["nonspace_gain_bpb_vs_r20"]=(
                baseline["nonspace_bpb"]-measured[arm]["nonspace_bpb"])
    best=measured["adaptive"]
    best.update(whole_book_deltas(
        np.concatenate(loss_vectors["r20_default"]),
        np.concatenate(loss_vectors["adaptive"]),books,WARMUP))
    best["dev_gate_pass"]=(best["gain_bpb_vs_r20"]>0 and
        best["nonspace_gain_bpb_vs_r20"]>=0 and
        best["whole_book_bootstrap_95pct_bits"][0]>0)
    return {"dev":measured,"dev_books":len(books),"dev_positions":positions,
            "max_conventional_ngram_diff":max_ngram,
            "max_frozen_r20_alignment_diff":max_frozen,"per_book":per_book}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
    paths,books=resolve_books(args.corpus_root,BYTES_PER_BOOK,TOTAL_BOOKS,SEED)
    verify_manifest(paths[:52],books[:52],manifest)
    hashes=[hashlib.sha256(b).hexdigest() for b in books]
    if tuple(hashes[132:156])!=R21_TEST_HASHES:
        raise ValueError("prior R21 independent TEST corpus fingerprints changed")
    if len(set(hashes))!=TOTAL_BOOKS or len(set(p.name for p in paths))!=TOTAL_BOOKS:
        raise ValueError("duplicate R22 corpus documents")
    prior=train_unigram_prior(books[:TRAIN_BOOKS])
    started=time.perf_counter()
    fit=score_grid(books[FIT_START:GATE_START],prior)
    gate=score_grid(books[GATE_START:DEV_START],prior)
    chosen=select_train_only(fit,gate)
    evaluated=evaluate_dev(books[DEV_START:],
                           [p.name for p in paths[DEV_START:]],prior,
                           tuple(chosen["scales"]))
    evaluated.update({
        "architecture":"M3 R22 independent pair/triangle/flux feedback and conventional controls",
        "r21_default_frozen":True,
        "r21_test_scored_as_r22_train_or_dev":False,
        "r21_test_run":"https://github.com/shemshallah/nanlm_experimental/actions/runs/43977992731",
        "new_fit_indices":[FIT_START,GATE_START],
        "new_gate_indices":[GATE_START,DEV_START],
        "new_dev_indices":[DEV_START,TOTAL_BOOKS],
        "new_dev_sha256":hashes[DEV_START:],
        "train_selected":chosen,"fit_grid":fit,"gate_grid":gate,
        "conventional_count_mixture_is_isomorphic_to_geometric_r20":True,
        "r12_checkpoint_loaded":False,"seed":SEED,"warmup":WARMUP,
        "bytes_per_book":BYTES_PER_BOOK,"corpus":manifest["corpus"],
        "elapsed_seconds":time.perf_counter()-started,
        "promotion":"not promoted without stronger DEV and fresh independent TEST",
    })
    output=json.dumps(evaluated,sort_keys=True,indent=2)+"\n"
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(output,encoding="utf-8")


if __name__=="__main__":
    main()
