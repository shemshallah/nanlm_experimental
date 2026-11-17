"""M3 R23 trained recurrent wave/apex candidate against frozen R21 and null.

All fitting uses disjoint NaNROUTER indices 188..203; selection uses 204..207;
fresh development uses 208..219. Earlier R21 independent TEST is NEVER used
to train, select or evaluate R23. Readout is 5 scalar parameters with Adam;
the original R16/R19/R20 weights and causal memory remain frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from m3_r14.predictive_probe import resolve_books, train_unigram_prior
from m3_r16.counterfactual_wave_gate import whole_book_deltas
from m3_r17.locked_test import SEED, BYTES_PER_BOOK, TRAIN_BOOKS, WARMUP
from m3_r18.locked_replication import MANIFEST, verify_manifest
from m3_r22.evaluate import R21_TEST_HASHES
from m3_r23.apex import AdamReadout, M3R23Predictor

FIT_START, GATE_START, DEV_START, TOTAL_BOOKS = 188, 204, 208, 220
TRAIN_BATCH = 48
TRAIN_LR = .003
ARMS = ("r20", "apex_torus", "apex_shuffled", "copy_only", "wave_only", "wrong_copy")


def scoring_start(book: bytes) -> int:
    start = WARMUP
    while start < len(book) and (book[start] & 0xC0) == 0x80:
        start += 1
    return start


def train_readout(books: list[bytes], prior: np.ndarray,
                  geometry: str) -> tuple[np.ndarray, AdamReadout, dict]:
    """Observed byte determines supervised loss; observation advances state afterward."""
    adam = AdamReadout(lr=TRAIN_LR)
    grad = np.zeros(5)
    batch = 0
    scored = 0
    training_loss = 0.0
    for book in books:
        model = M3R23Predictor(prior,geometry=geometry,theta=adam.theta)
        start = scoring_start(book)
        for position,byte in enumerate(book):
            if position >= start:
                base, features = model.base_and_features()
                logp = model.readout(base,features,adam.theta)
                distribution = np.exp(logp)
                error = distribution.copy()
                error[byte] -= 1.0
                grad += features @ error + .0001 * adam.theta
                training_loss -= float(logp[byte]) / np.log(2)
                scored += 1
                batch += 1
                if batch == TRAIN_BATCH:
                    adam.update(grad / batch)
                    grad.fill(0.0)
                    batch = 0
            model.observe(byte)
    if batch:
        adam.update(grad / batch)
    return adam.theta.copy(), adam, {
        "train_positions":scored,"updates":adam.steps,
        "final_train_bpb_during_training":training_loss / scored,
        "theta":adam.theta.tolist(),"adam_m":adam.m.tolist(),
        "adam_v":adam.v.tolist(),"adam_step":adam.steps,
    }


def evaluate_books(books: list[bytes], names: list[str],prior: np.ndarray,
                   torus_theta: np.ndarray, null_theta: np.ndarray) -> dict:
    if len(books) != len(names):
        raise ValueError("name/document mismatch")
    sums={arm:{"bits":0.,"nonspace":0.,"top1":0,"top5":0}
          for arm in ARMS}
    losses={arm:[] for arm in ARMS}
    n=n_nonspace=n_chars=0
    per_book=[]
    for name,book in zip(names,books):
        torus=M3R23Predictor(prior,geometry="torus",theta=torus_theta)
        shuffled=M3R23Predictor(prior,geometry="shuffled",theta=null_theta)
        start=scoring_start(book)
        chars=len(book[start:].decode("utf-8","strict"))
        n_chars+=chars
        local={arm:[] for arm in ARMS}
        for position,byte in enumerate(book):
            if position>=start:
                base,features=torus.base_and_features()
                shuffled_base,shuffled_features=shuffled.base_and_features()
                if not np.array_equal(base,shuffled_base):
                    raise ValueError("matched geometry null changed frozen R21")
                wrong_base,wrong_features=torus.base_and_features(wrong_values=True)
                if not np.array_equal(base,wrong_base):
                    raise ValueError("wrong-value control changed frozen R21")
                selected={
                    "r20":base,
                    "apex_torus":torus.readout(base,features,torus_theta),
                    "apex_shuffled":shuffled.readout(base,shuffled_features,null_theta),
                    "copy_only":torus.readout(base,features,
                                     torus_theta * np.array([1.,0,0,0,0])),
                    "wave_only":torus.readout(base,features,
                                     torus_theta * np.array([0.,1,1,1,1])),
                    "wrong_copy":torus.readout(base,wrong_features,torus_theta),
                }
                for arm,logp in selected.items():
                    bits=-float(logp[byte])/np.log(2)
                    local[arm].append(bits)
                    acc=sums[arm]
                    acc["bits"]+=bits
                    if byte !=32:
                        acc["nonspace"]+=bits
                    acc["top1"]+=int(np.argmax(logp)==byte)
                    acc["top5"]+=int(byte in np.argpartition(logp,-5)[-5:])
                n+=1
                n_nonspace+=int(byte!=32)
            torus.observe(byte)
            shuffled.observe(byte)
        row={"filename":name,"sha256":hashlib.sha256(book).hexdigest(),
             "positions":len(local["r20"])}
        for arm,vals in local.items():
            v=np.asarray(vals)
            losses[arm].append(v)
            row[arm+"_bpb"]=float(v.mean())
        row["gain_torus_vs_r20"]=row["r20_bpb"]-row["apex_torus_bpb"]
        per_book.append(row)
    if not (0<n_nonspace<n and n_chars>0):
        raise ValueError("inconsistent scored positions")
    measured={}
    for arm,v in sums.items():
        measured[arm]={"bpb":v["bits"]/n,"bpc":v["bits"]/n_chars,
            "nonspace_bpb":v["nonspace"]/n_nonspace,
            "top1":v["top1"]/n,"top5":v["top5"]/n,"positions":n}
    for arm in ARMS[1:]:
        measured[arm]["gain_bpb_vs_r20"]=measured["r20"]["bpb"]-measured[arm]["bpb"]
        measured[arm]["nonspace_gain_bpb_vs_r20"]=(
            measured["r20"]["nonspace_bpb"]-measured[arm]["nonspace_bpb"])
        measured[arm].update(whole_book_deltas(
            np.concatenate(losses["r20"]),np.concatenate(losses[arm]),books,WARMUP))
    return {"arms":measured,"per_book":per_book,"positions":n,
            "unicode_chars":n_chars,"books":len(books)}


def choose_gate(gate: dict) -> dict:
    """Train-only decision: never replace default for a weaker or null geometry."""
    arms=gate["arms"]
    baseline=arms["r20"]
    torus=arms["apex_torus"]
    shuffled=arms["apex_shuffled"]
    copy=arms["copy_only"]
    geometry_ok=(torus["bpb"] < baseline["bpb"]
                 and torus["bpb"] < shuffled["bpb"]
                 and torus["bpb"] < copy["bpb"]
                 and torus["nonspace_bpb"] <= baseline["nonspace_bpb"])
    copy_ok=(copy["bpb"] < baseline["bpb"]
             and copy["nonspace_bpb"] <= baseline["nonspace_bpb"])
    return {"chosen":"apex_torus" if geometry_ok else ("copy_only" if copy_ok else "r20"),
            "geometry_beats_shuffled_on_gate":bool(torus["bpb"]<shuffled["bpb"]),
            "geometry_beats_copy_on_gate":bool(torus["bpb"]<copy["bpb"]),
            "geometry_gate_pass":bool(geometry_ok),"copy_gate_pass":bool(copy_ok)}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--checkpoint",type=Path)
    args=parser.parse_args()
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
    paths,books=resolve_books(args.corpus_root,BYTES_PER_BOOK,TOTAL_BOOKS,SEED)
    verify_manifest(paths[:52],books[:52],manifest)
    hashes=[hashlib.sha256(book).hexdigest() for book in books]
    if tuple(hashes[132:156]) != R21_TEST_HASHES:
        raise ValueError("R21 TEST fingerprints mismatch; cannot train")
    if len(set(hashes))!=TOTAL_BOOKS or len(set(p.name for p in paths))!=TOTAL_BOOKS:
        raise ValueError("R23 corpus duplicate book or hash")
    prior=train_unigram_prior(books[:TRAIN_BOOKS])
    start=time.perf_counter()
    fitted,adam,torus_train=train_readout(books[FIT_START:GATE_START],prior,"torus")
    null_theta,null_adam,null_train=train_readout(books[FIT_START:GATE_START],prior,"shuffled")
    names=[p.name for p in paths]
    gate=evaluate_books(books[GATE_START:DEV_START],names[GATE_START:DEV_START],
                        prior,fitted,null_theta)
    decision=choose_gate(gate)
    dev=evaluate_books(books[DEV_START:],names[DEV_START:],prior,fitted,null_theta)
    arm=dev["arms"][decision["chosen"]]
    accepted=(decision["chosen"]!="r20" and
              arm["gain_bpb_vs_r20"]>0 and arm["nonspace_gain_bpb_vs_r20"]>=0
              and arm["whole_book_bootstrap_95pct_bits"][0]>0)
    result={
        "architecture":"M3 R23 trained recurrent triangle apex plus long-prefix copy",
        "r21_default_unmodified":True,"r21_test_train_or_dev":False,
        "r12_checkpoint_loaded":False,
        "fit_indices":[FIT_START,GATE_START],"gate_indices":[GATE_START,DEV_START],
        "dev_indices":[DEV_START,TOTAL_BOOKS],"dev_sha256":hashes[DEV_START:],
        "train_torus":torus_train,"train_shuffled_null":null_train,
        "gate":gate,"selection":decision,"dev":dev,"selected_dev_gate_pass":bool(accepted),
        "time_s":time.perf_counter()-start,"promotion":"none until independent TEST",
        "corpus":manifest["corpus"],"seed":SEED,"warmup":WARMUP,
    }
    output=json.dumps(result,sort_keys=True,indent=2)+"\n"
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(output,encoding="utf-8")
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True,exist_ok=True)
        np.savez(args.checkpoint,theta=fitted,adam_m=adam.m,adam_v=adam.v,
                 adam_step=adam.steps,shuffled_theta=null_theta,
                 shuffled_m=null_adam.m,shuffled_v=null_adam.v,
                 shuffled_step=null_adam.steps,prior=prior,
                 chosen=decision["chosen"],train_sha256=np.array(hashes[FIT_START:GATE_START]))


if __name__=="__main__":
    main()
