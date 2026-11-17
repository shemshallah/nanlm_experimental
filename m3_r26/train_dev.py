"""R26 checkpoint-continuation training and fresh-book evaluation.

Resume the exact R25 theta/m/v/768 steps from the GitHub Actions NPZ. Expand
its three trainable scalars to 14 (seven new context/class residuals plus
four additional class corrections) without modifying the accepted R24 state.
Compare against both R24 and inherited R25, with matched counterfactual
wrong-successor and unrelated-book memory. R25/R24 TEST and R26 TEST sealed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from m3_r14.predictive_probe import resolve_books, train_unigram_prior
from m3_r16.counterfactual_wave_gate import whole_book_deltas
from m3_r17.locked_test import SEED, TRAIN_BOOKS, BYTES_PER_BOOK, WARMUP
from m3_r18.locked_replication import MANIFEST, verify_manifest
from m3_r22.evaluate import R21_TEST_HASHES
from m3_r23.train_dev import scoring_start
from m3_r25.temporal import TemporalSuccessors, readout as r25_readout
from m3_r25.train_dev import FIT, GATE, DEV, SEALED_TEST
from m3_r26.hybrid import (WEIGHTS, ByteContextTree, M3R26Predictor,
                           pack_features, hybrid_readout)

ARMS = ("r24", "r25", "r26", "wrong_successor", "unrelated_history")
LR = 0.003
BATCH = 64
PARENT_STEP = 768
PARENT_THETA = np.array([.0983542567067886, .5696518238334743,
                         .28033085335532953], dtype=np.float64)
PARENT_M = np.array([.021235133587155296, .0011092687134161892,
                     -.0030329350785498104])
PARENT_V = np.array([.0014452031751310188, .0002926373938228902,
                     .0010171247805364427])


class ExpandedAdam:
    """Exact three-state Adam migration, with independent newborn state ages."""

    def __init__(self, parent_path: Path):
        with np.load(parent_path, allow_pickle=False) as previous:
            old_theta = np.asarray(previous["theta"], dtype=np.float64)
            old_m = np.asarray(previous["adam_m"], dtype=np.float64)
            old_v = np.asarray(previous["adam_v"], dtype=np.float64)
            old_step = int(previous["adam_step"])
            prior = np.asarray(previous["prior"], dtype=np.float64)
            fit_indices = tuple(previous["fit_indices"].tolist())
        if (old_theta.shape != (3,) or old_m.shape != (3,) or old_v.shape != (3,)
                or prior.shape != (256,) or fit_indices != FIT or old_step != PARENT_STEP):
            raise ValueError("incorrect R25 optimizer checkpoint shape, step, or fit split")
        for name, actual, expected in (("theta", old_theta, PARENT_THETA),
                                       ("m", old_m, PARENT_M), ("v", old_v, PARENT_V)):
            if not np.allclose(actual, expected, rtol=0, atol=1e-14):
                raise ValueError("R25 inherited " + name + " mismatched verified run")
        self.theta = np.zeros(WEIGHTS, dtype=np.float64)
        self.m = np.zeros(WEIGHTS, dtype=np.float64)
        self.v = np.zeros(WEIGHTS, dtype=np.float64)
        self.steps = np.zeros(WEIGHTS, dtype=np.int64)
        self.theta[:3] = old_theta
        self.m[:3] = old_m
        self.v[:3] = old_v
        self.steps[:3] = old_step
        self.parent_prior = prior

    def update(self, gradient: np.ndarray) -> None:
        if gradient.shape != (WEIGHTS,):
            raise ValueError("expected 14 feedback gradients")
        grad = np.clip(gradient, -2., 2.)
        self.steps += 1
        self.m = .9 * self.m + .1 * grad
        self.v = .999 * self.v + .001 * grad**2
        self.theta -= LR * (self.m / (1 - .9**self.steps)) / (
            np.sqrt(self.v / (1 - .999**self.steps)) + 1e-8)
        np.clip(self.theta, -2, 2, out=self.theta)


def train(books: list[bytes], prior: np.ndarray, opt: ExpandedAdam) -> dict:
    if not np.array_equal(prior, opt.parent_prior):
        raise ValueError("parent optimizer train prior changed")
    grad = np.zeros(WEIGHTS, dtype=np.float64)
    count = queued = 0
    bits = 0.
    started = time.perf_counter()
    for book_index, book in enumerate(books):
        model = M3R26Predictor(prior, opt.theta)
        start = scoring_start(book)
        for pos, target in enumerate(book):
            if pos >= start:
                base, features = model.base_and_hybrid_features()
                logp = hybrid_readout(base, features, opt.theta)
                p = np.exp(logp)
                p[target] -= 1
                # Penalize byte errors outside spaces more heavily; R25 had
                # regressed non-space BPB even while improving overall loss.
                importance = 0.65 if target == 32 else 1.20
                grad += importance * (features @ p) + .0002 * opt.theta
                bits -= float(logp[target]) / np.log(2)
                count += 1
                queued += 1
                if queued == BATCH:
                    opt.update(grad / queued)
                    grad.fill(0)
                    queued = 0
            model.observe(int(target))
        print(f"R26 FIT {book_index+1}/{len(books)} steps={int(opt.steps[0])} "
              f"online_bpb={bits/count:.6f} elapsed_s={time.perf_counter()-started:.1f}",
              file=sys.stderr, flush=True)
    if queued:
        opt.update(grad / queued)
    return {"positions": count, "new_updates": int(opt.steps[0]-PARENT_STEP),
            "training_online_bpb": bits/count,
            "parent_optimizer_state_preserved_at_init": True,
            "theta": opt.theta.tolist(), "adam_m": opt.m.tolist(),
            "adam_v": opt.v.tolist(), "adam_steps": opt.steps.tolist(),
            "elapsed_s": time.perf_counter()-started}


def evaluate(books: list[bytes], names: list[str], prior: np.ndarray,
             theta: np.ndarray) -> dict:
    if len(books) < 2 or len(names) != len(books):
        raise ValueError("at least two matched named books required")
    accum = {arm: {"bits": 0., "nonspace": 0., "top1": 0, "top5": 0} for arm in ARMS}
    losses = {arm: [] for arm in ARMS}
    total = nonspace = chars = 0
    rows = []
    start_clock = time.perf_counter()
    for index, (name, book) in enumerate(zip(names, books)):
        model = M3R26Predictor(prior, theta)
        cross_r25 = TemporalSuccessors()
        cross_tree = ByteContextTree()
        for other in books[(index + 1) % len(books)]:
            cross_r25.observe(int(other))
            cross_tree.observe(int(other))
        start = scoring_start(book)
        chars += len(book[start:].decode("utf-8", "strict"))
        local = {arm: [] for arm in ARMS}
        for pos, target in enumerate(book):
            if pos >= start:
                base, features = model.base_and_hybrid_features()
                wrong = model.hybrid_features_from_base(base, wrong_values=True)
                cross = pack_features(cross_r25.features(base), cross_tree.features(base))
                # R25 comparison is the unchanged 768-step recovered parent,
                # not R26's first three weights after additional training.
                predictions = (base, r25_readout(base, features[:3], PARENT_THETA),
                               hybrid_readout(base, features, theta),
                               hybrid_readout(base, wrong, theta),
                               hybrid_readout(base, cross, theta))
                for arm, logp in zip(ARMS, predictions):
                    loss = -float(logp[target]) / np.log(2)
                    local[arm].append(loss)
                    row = accum[arm]
                    row["bits"] += loss
                    if target != 32:
                        row["nonspace"] += loss
                    row["top1"] += int(np.argmax(logp) == target)
                    row["top5"] += int(target in np.argpartition(logp, -5)[-5:])
                total += 1
                nonspace += int(target != 32)
            model.observe(int(target))
        record = {"filename": name, "sha256": hashlib.sha256(book).hexdigest(),
                  "positions": len(local["r24"])}
        for arm, values in local.items():
            losses[arm].append(np.array(values))
            record[arm+"_bpb"] = float(np.mean(values))
        record["gain_vs_r24"] = record["r24_bpb"] - record["r26_bpb"]
        rows.append(record)
        print(f"R26 EVAL {index+1}/{len(books)} r24={record['r24_bpb']:.6f} "
              f"r25={record['r25_bpb']:.6f} r26={record['r26_bpb']:.6f}",
              file=sys.stderr, flush=True)
    metrics = {}
    for arm, sums in accum.items():
        metrics[arm] = {"bpb": sums["bits"]/total, "bpc": sums["bits"]/chars,
                        "nonspace_bpb": sums["nonspace"]/nonspace,
                        "top1": sums["top1"]/total, "top5": sums["top5"]/total,
                        "positions": total}
    for arm in ARMS[1:]:
        m = metrics[arm]
        m["gain_bpb_vs_r24"] = metrics["r24"]["bpb"]-m["bpb"]
        m["nonspace_gain_vs_r24"] = metrics["r24"]["nonspace_bpb"]-m["nonspace_bpb"]
        m.update(whole_book_deltas(np.concatenate(losses["r24"]),
                                   np.concatenate(losses[arm]), books, WARMUP))
    return {"arms": metrics, "per_book": rows, "books": len(books),
            "positions": total, "characters": chars,
            "elapsed_s": time.perf_counter()-start_clock}


def gate_pass(result: dict, minimum_books: int) -> bool:
    a = result["arms"]
    m = a["r26"]
    return bool(m["bpb"] < min(a["r24"]["bpb"], a["r25"]["bpb"],
                                a["wrong_successor"]["bpb"], a["unrelated_history"]["bpb"])
                and m["nonspace_bpb"] <= a["r24"]["nonspace_bpb"]
                and m["books_improved"] >= minimum_books)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    opt = ExpandedAdam(args.parent_checkpoint)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    paths, books = resolve_books(args.corpus_root, BYTES_PER_BOOK, DEV[1], SEED)
    verify_manifest(paths[:52], books[:52], manifest)
    hashes = [hashlib.sha256(b).hexdigest() for b in books]
    if tuple(hashes[132:156]) != R21_TEST_HASHES or len(set(hashes)) != DEV[1]:
        raise ValueError("book ancestry or disjointness changed")
    prior = train_unigram_prior(books[:TRAIN_BOOKS])
    # A direct migration parity check on actual corpus priors before training.
    migrated = M3R26Predictor(prior, opt.theta)
    for byte in books[FIT[0]][:32]:
        migrated.observe(int(byte))
        np.testing.assert_array_equal(migrated.predict_logp("r25"),
                                      migrated.predict_logp("r26"))
    trained = train(books[FIT[0]:FIT[1]], prior, opt)
    names = [p.name for p in paths]
    gate = evaluate(books[GATE[0]:GATE[1]], names[GATE[0]:GATE[1]], prior, opt.theta)
    pass_gate = gate_pass(gate, minimum_books=3)
    dev = evaluate(books[DEV[0]:DEV[1]], names[DEV[0]:DEV[1]], prior, opt.theta)
    pass_dev = pass_gate and gate_pass(dev, minimum_books=8) and (
        dev["arms"]["r26"]["whole_book_bootstrap_95pct_bits"][0] > 0)
    result = {"architecture": "M3 R26 hierarchical context tree / class-calibrated M3 R25",
              "inherited": "R24 default exact + R25 3-scalar learned path, same canonical state",
              "fit_indices": list(FIT), "gate_indices": list(GATE), "dev_indices": list(DEV),
              "sealed_independent_test": list(SEALED_TEST), "sealed_test_read": False,
              "fit": trained, "gate": gate, "gate_pass": pass_gate,
              "dev": dev, "dev_gate_pass": bool(pass_dev),
              "fit_sha256": hashes[FIT[0]:FIT[1]],
              "gate_sha256": hashes[GATE[0]:GATE[1]],
              "dev_sha256": hashes[DEV[0]:DEV[1]],
              "parent_step": PARENT_STEP, "original_R24_untouched": True,
              "promotion": "none; R24 stays accepted default until locked TEST passes"}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.checkpoint, theta=opt.theta, adam_m=opt.m, adam_v=opt.v,
                 adam_steps=opt.steps, prior=prior, fit_indices=np.array(FIT),
                 fit_sha256=np.array(hashes[FIT[0]:FIT[1]]),
                 sealed_test_indices=np.array(SEALED_TEST))
    print(json.dumps({"fit": trained, "gate": gate["arms"],
                      "gate_pass": pass_gate, "dev": dev["arms"],
                      "dev_gate_pass": bool(pass_dev)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
