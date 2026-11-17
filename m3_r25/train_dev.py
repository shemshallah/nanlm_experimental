"""M3 R25 CPU TRAIN/GATE/DEV: learned time-scale feedback on frozen R24.

FIT 244..259, GATE 260..263, DEV 264..275. Earlier R21/R24 TEST books
are never used for fitting. New independent TEST 276..299 stays unread and
sealed until an R25 candidate is frozen and pre-registered. The accepted R24
model, its signed mirror-wave states and its exact scalar are never updated.
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
from m3_r17.locked_test import BYTES_PER_BOOK, SEED, TRAIN_BOOKS, WARMUP
from m3_r18.locked_replication import MANIFEST, verify_manifest
from m3_r22.evaluate import R21_TEST_HASHES
from m3_r23.train_dev import scoring_start
from m3_r25.temporal import FEATURE_COUNT, M3R25Predictor, TemporalSuccessors, readout

FIT = (244, 260)
GATE = (260, 264)
DEV = (264, 276)
SEALED_TEST = (276, 300)
LR = 0.004
BATCH = 64
ARMS = ("r24", "r25", "wrong_successor", "unrelated_history")


class ScalarAdam:
    def __init__(self, lr: float = LR):
        self.theta = np.zeros(FEATURE_COUNT, dtype=np.float64)
        self.m = np.zeros_like(self.theta)
        self.v = np.zeros_like(self.theta)
        self.step = 0
        self.lr = lr

    def update(self, gradient: np.ndarray) -> None:
        self.step += 1
        grad = np.clip(gradient, -2, 2)
        self.m = 0.9 * self.m + 0.1 * grad
        self.v = 0.999 * self.v + 0.001 * grad**2
        self.theta -= self.lr * (self.m / (1 - 0.9**self.step)) / (
            np.sqrt(self.v / (1 - 0.999**self.step)) + 1e-8)
        # Bound signed feedback to prevent low-support context overconfidence.
        np.clip(self.theta, -2, 2, out=self.theta)


def train(books: list[bytes], prior: np.ndarray) -> tuple[ScalarAdam, dict]:
    opt = ScalarAdam()
    grad = np.zeros(FEATURE_COUNT, dtype=np.float64)
    batch = positions = 0
    bits = 0.0
    started = time.perf_counter()
    for book_index, book in enumerate(books):
        model = M3R25Predictor(prior)
        start = scoring_start(book)
        for i, target in enumerate(book):
            if i >= start:
                base, features = model.base_and_features()
                logp = readout(base, features, opt.theta)
                error = np.exp(logp)
                error[target] -= 1
                grad += features @ error + 0.0001 * opt.theta
                bits += -float(logp[target]) / np.log(2)
                positions += 1
                batch += 1
                if batch == BATCH:
                    opt.update(grad / batch)
                    batch = 0
                    grad.fill(0)
            model.observe(int(target))
        print(f"R25 FIT {book_index + 1}/{len(books)} | updates={opt.step} | "
              f"online_bpb={bits/positions:.6f} | elapsed_s={time.perf_counter()-started:.1f}",
              file=sys.stderr, flush=True)
    if batch:
        opt.update(grad / batch)
    return opt, {"positions": positions, "updates": opt.step,
                 "training_online_bpb": bits / positions,
                 "theta": opt.theta.tolist(), "adam_m": opt.m.tolist(),
                 "adam_v": opt.v.tolist(), "adam_step": opt.step,
                 "elapsed_s": time.perf_counter() - started}


def evaluate(books: list[bytes], names: list[str], prior: np.ndarray,
             theta: np.ndarray) -> dict:
    if len(books) < 2 or len(books) != len(names):
        raise ValueError("require two or more named books for unrelated-history controls")
    scores = {arm: {"bits": 0., "nonspace": 0., "top1": 0, "top5": 0} for arm in ARMS}
    losses = {arm: [] for arm in ARMS}
    count = nonspace = characters = hits = 0
    per_book = []
    started = time.perf_counter()
    for book_index, (name, book) in enumerate(zip(names, books)):
        model = M3R25Predictor(prior)
        unrelated = TemporalSuccessors()
        for other_byte in books[(book_index + 1) % len(books)]:
            unrelated.observe(int(other_byte))
        start = scoring_start(book)
        characters += len(book[start:].decode("utf-8", "strict"))
        local = {arm: [] for arm in ARMS}
        for i, target in enumerate(book):
            if i >= start:
                base, features = model.base_and_features()
                wrong = model.temporal.features(base, wrong_values=True)
                cross = unrelated.features(base)
                if np.any(features):
                    hits += 1
                candidates = (base, readout(base, features, theta),
                              readout(base, wrong, theta), readout(base, cross, theta))
                for arm, logp in zip(ARMS, candidates):
                    loss = -float(logp[target]) / np.log(2)
                    local[arm].append(loss)
                    row = scores[arm]
                    row["bits"] += loss
                    if target != 32:
                        row["nonspace"] += loss
                    row["top1"] += int(np.argmax(logp) == target)
                    row["top5"] += int(target in np.argpartition(logp, -5)[-5:])
                count += 1
                nonspace += int(target != 32)
            model.observe(int(target))  # all history updates AFTER target scoring
        record = {"filename": name, "sha256": hashlib.sha256(book).hexdigest(),
                  "positions": len(local["r24"])}
        for arm, values in local.items():
            losses[arm].append(np.array(values, dtype=np.float64))
            record[arm + "_bpb"] = float(np.mean(values))
        record["gain_bpb"] = record["r24_bpb"] - record["r25_bpb"]
        per_book.append(record)
        print(f"R25 EVAL {book_index + 1}/{len(books)} | "
              f"r24_bpb={record['r24_bpb']:.6f} | r25_bpb={record['r25_bpb']:.6f}",
              file=sys.stderr, flush=True)
    metrics = {}
    for arm, sums in scores.items():
        metrics[arm] = {"bpb": sums["bits"] / count,
                        "bpc": sums["bits"] / characters,
                        "nonspace_bpb": sums["nonspace"] / nonspace,
                        "top1": sums["top1"] / count, "top5": sums["top5"] / count,
                        "positions": count}
    for arm in ARMS[1:]:
        metrics[arm]["gain_bpb_vs_r24"] = metrics["r24"]["bpb"] - metrics[arm]["bpb"]
        metrics[arm]["nonspace_gain_bpb_vs_r24"] = (
            metrics["r24"]["nonspace_bpb"] - metrics[arm]["nonspace_bpb"])
        metrics[arm].update(whole_book_deltas(
            np.concatenate(losses["r24"]), np.concatenate(losses[arm]), books, WARMUP))
    return {"arms": metrics, "per_book": per_book, "positions": count,
            "characters": characters, "books": len(books),
            "retrieval_hit_rate": hits / count,
            "elapsed_s": time.perf_counter() - started}


def gate_pass(result: dict, minimum_books: int) -> bool:
    arm = result["arms"]
    candidate = arm["r25"]
    return bool(candidate["gain_bpb_vs_r24"] > 0
                and candidate["nonspace_gain_bpb_vs_r24"] >= 0
                and candidate["books_improved"] >= minimum_books
                and candidate["bpb"] < arm["wrong_successor"]["bpb"]
                and candidate["bpb"] < arm["unrelated_history"]["bpb"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    paths, books = resolve_books(args.corpus_root, BYTES_PER_BOOK, DEV[1], SEED)
    verify_manifest(paths[:52], books[:52], manifest)
    hashes = [hashlib.sha256(book).hexdigest() for book in books]
    if tuple(hashes[132:156]) != R21_TEST_HASHES:
        raise ValueError("R21 TEST ancestry mismatch")
    if len(set(hashes)) != DEV[1] or len(set(p.name for p in paths)) != DEV[1]:
        raise ValueError("duplicate book across R25/prior lineages")
    prior = train_unigram_prior(books[:TRAIN_BOOKS])
    opt, fitted = train(books[FIT[0]:FIT[1]], prior)
    names = [p.name for p in paths]
    gate = evaluate(books[GATE[0]:GATE[1]], names[GATE[0]:GATE[1]], prior, opt.theta)
    decision = gate_pass(gate, minimum_books=3)
    dev = evaluate(books[DEV[0]:DEV[1]], names[DEV[0]:DEV[1]], prior, opt.theta)
    accepted = decision and gate_pass(dev, minimum_books=8) and (
        dev["arms"]["r25"]["whole_book_bootstrap_95pct_bits"][0] > 0)
    result = {"architecture": "M3 R25 multiscale temporal feedback on accepted R24",
              "parent": "M3 R24 exact canonical default; all earlier coefficients frozen",
              "fit_indices": list(FIT), "gate_indices": list(GATE),
              "dev_indices": list(DEV), "future_independent_test": list(SEALED_TEST),
              "future_test_read": False, "prior_train_indices": [0, TRAIN_BOOKS],
              "prior_original_train_only": True,
              "original_r24_weights_untouched": True,
              "fit": fitted, "gate": gate, "gate_pass": decision,
              "dev": dev, "dev_gate_pass": bool(accepted),
              "candidate_sha256": hashlib.sha256(opt.theta.tobytes()).hexdigest(),
              "fit_sha256": hashes[FIT[0]:FIT[1]],
              "gate_sha256": hashes[GATE[0]:GATE[1]],
              "dev_sha256": hashes[DEV[0]:DEV[1]],
              "seed": SEED, "warmup": WARMUP, "bytes_per_book": BYTES_PER_BOOK,
              "promotion": "candidate only; R24 stays default pending a locked R25 TEST"}
    output = json.dumps(result, sort_keys=True, indent=2) + "\n"
    print(json.dumps({"fit": fitted, "gate_pass": decision, "dev_gate_pass": bool(accepted),
                      "gate": gate["arms"], "dev": dev["arms"]}, sort_keys=True, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.checkpoint, theta=opt.theta, adam_m=opt.m, adam_v=opt.v,
                 adam_step=opt.step, prior=prior,
                 fit_sha256=np.array(hashes[FIT[0]:FIT[1]]),
                 fit_indices=np.array(FIT), gate_indices=np.array(GATE),
                 dev_indices=np.array(DEV), sealed_test_indices=np.array(SEALED_TEST))


if __name__ == "__main__":
    main()
