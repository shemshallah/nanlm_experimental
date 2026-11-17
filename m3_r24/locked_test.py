"""Pre-registered independent TEST of R23's FIT/GATE-selected causal copy.

R23's wave geometry lost its shuffled-place control and is EXCLUDED from the
candidate. The frozen readout scalar and Adam moments are recorded in the
adjacent immutable JSON. No parameter is optimized or selected on TEST.
TEST uses 24 genuinely new NaNROUTER books, indices 220..243, never used by
R16..R23 TRAIN, gates, DEV, or the earlier R21 independent TEST.
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
from m3_r23.apex import M3R23Predictor
from m3_r23.train_dev import scoring_start

LOCK = Path(__file__).with_name("r23_frozen_candidate.json")
ARMS = ("r21_default", "copy_only", "wrong_copy")
TEST_FIRST, TEST_END = 220, 244


def load_locked_candidate() -> tuple[np.ndarray, dict]:
    metadata = json.loads(LOCK.read_text(encoding="utf-8"))
    if metadata["selection"] != "copy_only" or metadata["locked_test_indices"] != [TEST_FIRST, TEST_END]:
        raise ValueError("candidate selection or sealed TEST indices changed")
    theta = np.asarray(metadata["readout_theta"], dtype=np.float64)
    trained = np.asarray(metadata["trained_torus_theta"], dtype=np.float64)
    if theta.shape != (5,) or trained.shape != (5,) or not np.isfinite(theta).all():
        raise ValueError("invalid frozen R23 readout")
    if theta[0] != trained[0] or np.any(theta[1:]):
        raise ValueError("only the GATE-selected exact copy scalar may be active")
    if metadata["adam_step"] != 1024:
        raise ValueError("checkpoint optimizer step mismatch")
    return theta, metadata


def score_locked_test(books: list[bytes], names: list[str], prior: np.ndarray,
                      theta: np.ndarray) -> dict:
    if len(books) != TEST_END - TEST_FIRST or len(names) != len(books):
        raise ValueError("R24 requires all 24 predetermined new TEST books")
    totals = {k:{"bits":0.0,"nonspace":0.0,"top1":0,"top5":0} for k in ARMS}
    loss_vectors = {k:[] for k in ARMS}
    n = nonspace = unicode_chars = copy_hits = 0
    per_book = []
    started = time.perf_counter()
    for name,book in zip(names,books):
        model = M3R23Predictor(prior,geometry="torus",theta=theta)
        start = scoring_start(book)
        unicode_chars += len(book[start:].decode("utf-8","strict"))
        local = {k:[] for k in ARMS}
        for pos,observed in enumerate(book):
            if pos >= start:
                base = model.predict_logp("r20")
                residual = model.suffix.residual(base)
                wrong_residual = model.suffix.residual(base,wrong_values=True)
                zero = np.zeros((4,256),dtype=np.float64)
                candidate = model.readout(base,np.vstack((residual,zero)),theta)
                wrong = model.readout(base,np.vstack((wrong_residual,zero)),theta)
                if not np.any(residual):
                    # R23 readout re-normalizes the already-normalized R21
                    # logits even when its copied feature is zero. That is
                    # mathematically the identity, but can move the final
                    # floating-point bit by one ULP. Keep exact R23 execution
                    # and check numerical parity instead of bitwise identity.
                    if not np.allclose(base,candidate,atol=1e-12,rtol=0):
                        raise AssertionError("missing suffix changed default beyond roundoff")
                else:
                    copy_hits += 1
                for arm,lp in (("r21_default",base),("copy_only",candidate),
                               ("wrong_copy",wrong)):
                    bits = -float(lp[observed])/np.log(2)
                    local[arm].append(bits)
                    t = totals[arm]
                    t["bits"] += bits
                    if observed != 32:
                        t["nonspace"] += bits
                    t["top1"] += int(np.argmax(lp)==observed)
                    t["top5"] += int(observed in np.argpartition(lp,-5)[-5:])
                n += 1
                nonspace += int(observed!=32)
            model.observe(int(observed))  # update only AFTER scoring current byte
        row = {"filename":name,"sha256":hashlib.sha256(book).hexdigest(),
               "positions":len(local["r21_default"])}
        for arm,values in local.items():
            loss_vectors[arm].append(np.asarray(values,dtype=np.float64))
            row[arm+"_bpb"] = float(np.mean(values))
        row["gain_bpb"] = row["r21_default_bpb"]-row["copy_only_bpb"]
        per_book.append(row)
    if not (0 < nonspace < n and unicode_chars > 0):
        raise ValueError("missing or invalid TEST positions")
    metrics = {k:{"bpb":v["bits"]/n,"bpc":v["bits"]/unicode_chars,
                  "nonspace_bpb":v["nonspace"]/nonspace,
                  "top1":v["top1"]/n,"top5":v["top5"]/n,
                  "positions":n} for k,v in totals.items()}
    for arm in ARMS[1:]:
        row = metrics[arm]
        row["gain_bpb_vs_r21"] = metrics["r21_default"]["bpb"]-row["bpb"]
        row["nonspace_gain_bpb_vs_r21"] = (
            metrics["r21_default"]["nonspace_bpb"]-row["nonspace_bpb"])
        row.update(whole_book_deltas(np.concatenate(loss_vectors["r21_default"]),
                                     np.concatenate(loss_vectors[arm]),books,WARMUP))
    copy = metrics["copy_only"]
    requirements = json.loads(LOCK.read_text(encoding="utf-8"))["locked_test_requirements"]
    accept = (copy["gain_bpb_vs_r21"] > 0
              and copy["nonspace_gain_bpb_vs_r21"] >= 0
              and copy["whole_book_bootstrap_95pct_bits"][0] > 0
              and metrics["wrong_copy"]["bpb"] > copy["bpb"]
              and copy["books_improved"] >= requirements["minimum_books_improved"])
    return {"metrics":metrics,"test_positions":n,"test_books":len(books),
            "unicode_chars":unicode_chars,"copy_hit_rate":copy_hits/n,
            "per_book":per_book,"independent_test_gate":bool(accept),
            "time_s":time.perf_counter()-started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    theta,lock = load_locked_candidate()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    paths,books = resolve_books(args.corpus_root,BYTES_PER_BOOK,TEST_END,SEED)
    verify_manifest(paths[:52],books[:52],manifest)
    hashes = [hashlib.sha256(b).hexdigest() for b in books]
    if len(set(hashes))!=TEST_END or len({p.name for p in paths})!=TEST_END:
        raise ValueError("book duplicates detected across TEST and prior books")
    if tuple(hashes[132:156]) != R21_TEST_HASHES:
        raise ValueError("earlier sealed R21 TEST corpus fingerprints mismatch")
    prior = train_unigram_prior(books[:TRAIN_BOOKS])
    result = score_locked_test(books[TEST_FIRST:TEST_END],
                               [p.name for p in paths[TEST_FIRST:TEST_END]],prior,theta)
    result.update({
        "architecture":"M3 R24 locked independent TEST of R23 frozen causal-copy readout",
        "candidate_source":lock["training_commit"],
        "training_artifact_sha256":lock["training_artifact_sha256"],
        "sealed_test_indices":[TEST_FIRST,TEST_END],
        "sealed_test_sha256":hashes[TEST_FIRST:TEST_END],
        "corpus":manifest["corpus"],"seed":SEED,"warmup":WARMUP,
        "original_parent_weights_untouched":True,
        "trained_or_selected_on_test":False,
        "promotion":"eligible if independent_test_gate, otherwise rejected",
    })
    output = json.dumps(result,sort_keys=True,indent=2)+"\n"
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(output,encoding="utf-8")


if __name__=="__main__":
    main()
