"""M3 R21: frozen independent TEST of the canonical online R20 default.

First 132 seeded documents are excluded from TEST scoring; only indices
132..155 (24 previously untouched books) are scored. R19 and R20 gates,
original R16 weights, prior and class-specific readout remain immutable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from m3_default.predictor import M3R21Predictor as M3DefaultPredictor
from m3_r14.predictive_probe import resolve_books, train_unigram_prior
from m3_r16.counterfactual_wave_gate import whole_book_deltas
from m3_r17.locked_test import SEED, BYTES_PER_BOOK, WARMUP, TRAIN_BOOKS
from m3_r18.locked_replication import MANIFEST, verify_manifest

SEALED_R18 = 52
R20_DEV_END = 132
NEW_TEST_BOOKS = 24
TOTAL_BOOKS = R20_DEV_END + NEW_TEST_BOOKS
ARMS = ("r19", "r20", "triangle_only", "wrong_address", "wrong_context")


def score_streamed_test(books: list[bytes], names: list[str],
                        prior: np.ndarray) -> dict:
    if len(books) != NEW_TEST_BOOKS or len(names) != NEW_TEST_BOOKS:
        raise ValueError("R21 requires exactly 24 new independent TEST books")
    totals = {mode: {"bits":0.0,"nonspace_bits":0.0,
                     "top1":0,"top5":0} for mode in ARMS}
    loss_vectors = {mode:[] for mode in ARMS}
    per_book = []
    n_positions = n_nonspace = n_chars = 0
    start_time = time.perf_counter()
    for name, book in zip(names, books):
        predictor = M3DefaultPredictor(prior)  # default is frozen R20
        scoring_start = WARMUP
        while scoring_start < len(book) and (book[scoring_start] & 0xC0)==0x80:
            scoring_start += 1
        n_chars += len(book[scoring_start:].decode("utf-8","strict"))
        local = {mode:[] for mode in ARMS}
        local_top1 = {mode:0 for mode in ARMS}
        local_top5 = {mode:0 for mode in ARMS}
        local_nonspace = {mode:[] for mode in ARMS}
        for pos, raw in enumerate(book):
            target = int(raw)
            if pos >= scoring_start:
                candidates = predictor.predict_all()
                for mode in ARMS:
                    lp = candidates[mode]
                    bits = -float(lp[target]) / np.log(2)
                    local[mode].append(bits)
                    local_top1[mode] += int(np.argmax(lp)==target)
                    local_top5[mode] += int(target in np.argpartition(lp,-5)[-5:])
                    if target != 32:
                        local_nonspace[mode].append(bits)
            predictor.observe(target)  # strictly AFTER predicting current target
        n = len(local["r19"])
        if not n or n != len(book)-scoring_start:
            raise ValueError("streaming score/warmup alignment mismatch")
        n_positions += n
        n_nonspace += len(local_nonspace["r19"])
        row = {"filename":name,"sha256":hashlib.sha256(book).hexdigest(),
               "positions":n}
        for mode in ARMS:
            values = np.asarray(local[mode], dtype=np.float64)
            loss_vectors[mode].append(values)
            totals[mode]["bits"] += float(values.sum())
            totals[mode]["nonspace_bits"] += float(np.sum(local_nonspace[mode]))
            totals[mode]["top1"] += local_top1[mode]
            totals[mode]["top5"] += local_top5[mode]
            row[mode+"_bpb"] = float(values.mean())
        row["r20_gain_bpb"] = row["r19_bpb"]-row["r20_bpb"]
        per_book.append(row)
    if not (0 < n_nonspace < n_positions and n_chars > 0):
        raise ValueError("TEST has invalid byte/Unicode counts")
    measured = {}
    for mode, total in totals.items():
        measured[mode] = {
            "positions":n_positions,
            "bpb":total["bits"]/n_positions,
            "bpc":total["bits"]/n_chars,
            "nonspace_bpb":total["nonspace_bits"]/n_nonspace,
            "top1":total["top1"]/n_positions,
            "top5":total["top5"]/n_positions,
        }
    base = measured["r19"]
    for mode in ARMS[1:]:
        measured[mode]["gain_bpb_vs_r19"] = base["bpb"]-measured[mode]["bpb"]
        measured[mode]["nonspace_gain_bpb_vs_r19"] = (
            base["nonspace_bpb"]-measured[mode]["nonspace_bpb"]
        )
    default = measured["r20"]
    default.update(whole_book_deltas(
        np.concatenate(loss_vectors["r19"]),
        np.concatenate(loss_vectors["r20"]), books, WARMUP
    ))
    default["independent_test_gate"] = (
        default["gain_bpb_vs_r19"] > 0 and
        default["nonspace_gain_bpb_vs_r19"] > 0 and
        default["whole_book_bootstrap_95pct_bits"][0] > 0 and
        measured["wrong_address"]["gain_bpb_vs_r19"] <
            default["gain_bpb_vs_r19"]
    )
    return {
        "test":measured, "test_books":NEW_TEST_BOOKS,
        "test_positions":n_positions,"test_unicode_chars":n_chars,
        "test_positions_per_s_all_modes":n_positions/(time.perf_counter()-start_time),
        "per_book":per_book,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    paths,books = resolve_books(args.corpus_root,BYTES_PER_BOOK,TOTAL_BOOKS,SEED)
    verify_manifest(paths[:SEALED_R18],books[:SEALED_R18],manifest)
    hashes = [hashlib.sha256(b).hexdigest() for b in books]
    if len(set(hashes)) != TOTAL_BOOKS or len({p.name for p in paths})!=TOTAL_BOOKS:
        raise ValueError("book overlap or corpus fingerprint collision")
    prior = train_unigram_prior(books[:TRAIN_BOOKS])
    result = score_streamed_test(
        books[R20_DEV_END:],[p.name for p in paths[R20_DEV_END:]],prior
    )
    result.update({
        "architecture":"M3 R21 independent TEST of online R20 default",
        "source_r20_run":"https://github.com/shemshallah/nanlm_experimental/actions/runs/43970087999",
        "r19_space_scale":0.25,"r19_nonspace_scale":1.0,
        "r20_pair_scale":0.5,"r20_triangle_scale":0.5,
        "r12_checkpoint_loaded":False,
        "all_original_weights_frozen":True,
        "previous_132_books_scored_as_test":False,
        "corpus":manifest["corpus"],"seed":SEED,"warmup":WARMUP,
        "bytes_per_book":BYTES_PER_BOOK,
        "new_test_indices":[R20_DEV_END,TOTAL_BOOKS],
        "test_sha256":hashes[R20_DEV_END:],
        "publication":"independent replication; no superiority to matched ngram claimed",
    })
    text=json.dumps(result,sort_keys=True,indent=2)+"\n"
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text,encoding="utf-8")


if __name__=="__main__":
    main()
