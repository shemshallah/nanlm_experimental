"""M3 R29 new-book CPU training with exact frozen R26 + Adam state.

Uses NaNROUTER short-story batch 0004 only. TRAIN 300..315, GATE 316..319,
DEV 320..331; 332..355 remain sealed and unread. All previous R24/R26 TEST
books are excluded. Optimization updates only eight newborn lexical scalars;
all 14 R26 readout weights and their independent Adam state ages/moments are
carried forward exactly, not reconstructed or re-trained from TEST.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from m3_default.predictor import M3DefaultPredictor, R26_THETA
from m3_r14.predictive_probe import resolve_books, train_unigram_prior
from m3_r16.counterfactual_wave_gate import whole_book_deltas
from m3_r17.locked_test import SEED, TRAIN_BOOKS, BYTES_PER_BOOK, WARMUP
from m3_r18.locked_replication import MANIFEST, verify_manifest
from m3_r22.evaluate import R21_TEST_HASHES
from m3_r23.train_dev import scoring_start
from m3_r27.locked_test import LOCK, digest
from m3_r29.lexical import (NEW_WEIGHTS, TOTAL_WEIGHTS, M3R29Predictor,
                            WordSuccessorMemory, pack_lexical, lexical_readout)

FIT = (300, 316)
GATE = (316, 320)
DEV = (320, 332)
SEALED_TEST = (332, 356)
ARMS = ("r26", "r29", "wrong_successor", "unrelated_history")
LR = 0.004
BATCH = 64


class FrozenParentAdam:
    """Read-only 14-state R26 Adam + independent eight-scalar Adam updates."""

    def __init__(self, path: Path):
        lock = json.loads(LOCK.read_text(encoding="utf-8"))
        if not path.is_file():
            raise ValueError("complete R26 parent checkpoint is missing")
        if digest(path) != lock["training_checkpoint_sha256"]:
            raise ValueError("R26 checkpoint SHA-256 mismatch")
        with np.load(path, allow_pickle=False) as parent:
            self.parent_theta = np.asarray(parent["theta"], dtype=np.float64).copy()
            self.parent_m = np.asarray(parent["adam_m"], dtype=np.float64).copy()
            self.parent_v = np.asarray(parent["adam_v"], dtype=np.float64).copy()
            self.parent_steps = np.asarray(parent["adam_steps"], dtype=np.int64).copy()
            self.prior = np.asarray(parent["prior"], dtype=np.float64).copy()
            self.parent_fit = tuple(parent["fit_indices"].tolist())
        if (self.parent_theta.shape != (14,) or self.parent_m.shape != (14,)
                or self.parent_v.shape != (14,) or self.parent_steps.shape != (14,)
                or self.prior.shape != (256,) or self.parent_fit != (244, 260)
                or self.parent_steps.tolist() != lock["optimizer_steps"]
                or not np.array_equal(self.parent_theta, R26_THETA)):
            raise ValueError("frozen R26 weights, state ages, or prior were altered")
        self.theta = np.zeros(NEW_WEIGHTS, dtype=np.float64)
        self.m = np.zeros(NEW_WEIGHTS, dtype=np.float64)
        self.v = np.zeros(NEW_WEIGHTS, dtype=np.float64)
        self.step = 0

    def update(self, gradient: np.ndarray) -> None:
        if gradient.shape != (NEW_WEIGHTS,):
            raise ValueError("incorrect R29 gradient shape")
        self.step += 1
        g = np.clip(gradient, -2., 2.)
        self.m = .9 * self.m + .1 * g
        self.v = .999 * self.v + .001 * g ** 2
        self.theta -= LR * (self.m / (1. - .9 ** self.step)) / (
            np.sqrt(self.v / (1. - .999 ** self.step)) + 1e-8)
        np.clip(self.theta, -1.5, 1.5, out=self.theta)

    def save(self, path: Path, hashes: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        theta = np.concatenate((self.parent_theta, self.theta))
        m = np.concatenate((self.parent_m, self.m))
        v = np.concatenate((self.parent_v, self.v))
        steps = np.concatenate((self.parent_steps, np.full(NEW_WEIGHTS, self.step)))
        if theta.shape != (TOTAL_WEIGHTS,):
            raise AssertionError("R29 checkpoint omitted an inherited optimizer group")
        np.savez(path, theta=theta, adam_m=m, adam_v=v, adam_steps=steps,
                 prior=self.prior, parent_checkpoint_sha256=digest_of_parent(),
                 fit_indices=np.array(FIT), fit_sha256=np.array(hashes[FIT[0]:FIT[1]]),
                 gate_indices=np.array(GATE), dev_indices=np.array(DEV),
                 sealed_test_indices=np.array(SEALED_TEST))


def digest_of_parent() -> str:
    return json.loads(LOCK.read_text(encoding="utf-8"))["training_checkpoint_sha256"]


def train(books: list[bytes], prior: np.ndarray, optimizer: FrozenParentAdam) -> dict:
    if not np.array_equal(prior, optimizer.prior):
        raise ValueError("original TRAIN prior changed")
    gradient = np.zeros(NEW_WEIGHTS)
    queued = positions = 0
    bits = 0.
    hits = 0
    start = time.perf_counter()
    for book_index, book in enumerate(books):
        model = M3R29Predictor(prior)
        warmup = scoring_start(book)
        for pos, target in enumerate(book):
            if pos >= warmup:
                base, features = model.base_and_lexical_features()
                if np.any(features):
                    hits += 1
                logp = lexical_readout(base, features, optimizer.theta)
                error = np.exp(logp)
                error[target] -= 1.
                importance = .75 if target == 32 else 1.10
                gradient += importance * (features @ error) + .0001 * optimizer.theta
                bits -= float(logp[target]) / np.log(2)
                queued += 1
                positions += 1
                if queued == BATCH:
                    optimizer.update(gradient / queued)
                    gradient.fill(0)
                    queued = 0
            model.observe(int(target))
        print(f"R29 FIT {book_index+1}/{len(books)} updates={optimizer.step} "
              f"online_bpb={bits/positions:.6f} elapsed={time.perf_counter()-start:.1f}s",
              file=sys.stderr, flush=True)
    if queued:
        optimizer.update(gradient / queued)
    return {"positions": positions, "updates_new_groups": optimizer.step,
            "training_online_bpb": bits / positions, "lexical_hit_rate": hits / positions,
            "theta": optimizer.theta.tolist(), "adam_m": optimizer.m.tolist(),
            "adam_v": optimizer.v.tolist(), "inherited_optimizer_groups": 14,
            "new_optimizer_groups": NEW_WEIGHTS, "inherited_optimizer_unchanged": True,
            "elapsed_s": time.perf_counter() - start}


def evaluate(books: list[bytes], names: list[str], prior: np.ndarray,
             theta: np.ndarray) -> dict:
    if len(books) < 2 or len(names) != len(books):
        raise ValueError("two or more named disjoint books required")
    if theta.shape != (NEW_WEIGHTS,):
        raise ValueError("wrong R29 candidate shape")
    sums = {arm: {"bits": 0., "nonspace": 0., "top1": 0, "top5": 0} for arm in ARMS}
    losses = {arm: [] for arm in ARMS}
    total = nonspace = unicode_chars = hits = 0
    per_book = []
    start_clock = time.perf_counter()
    for index, (name, book) in enumerate(zip(names, books)):
        model = M3R29Predictor(prior)
        unrelated = WordSuccessorMemory()
        for byte in books[(index + 1) % len(books)]:
            unrelated.observe(int(byte))
        warmup = scoring_start(book)
        unicode_chars += len(book[warmup:].decode("utf-8", "strict"))
        local = {arm: [] for arm in ARMS}
        for pos, target in enumerate(book):
            if pos >= warmup:
                base, features = model.base_and_lexical_features()
                wrong = pack_lexical(model.lexical.features(base, wrong_values=True))
                cross = pack_lexical(unrelated.features(base, query=model.lexical))
                if np.any(features):
                    hits += 1
                arms = (base, lexical_readout(base, features, theta),
                        lexical_readout(base, wrong, theta),
                        lexical_readout(base, cross, theta))
                for arm, logp in zip(ARMS, arms):
                    loss = -float(logp[target]) / np.log(2)
                    local[arm].append(loss)
                    sums[arm]["bits"] += loss
                    if target != 32:
                        sums[arm]["nonspace"] += loss
                    sums[arm]["top1"] += int(np.argmax(logp) == target)
                    sums[arm]["top5"] += int(target in np.argpartition(logp, -5)[-5:])
                total += 1
                nonspace += int(target != 32)
            model.observe(int(target))
        row = {"filename": name, "sha256": hashlib.sha256(book).hexdigest(),
               "positions": len(local["r26"])}
        for arm, values in local.items():
            losses[arm].append(np.asarray(values))
            row[arm + "_bpb"] = float(np.mean(values))
        row["gain_vs_r26"] = row["r26_bpb"] - row["r29_bpb"]
        per_book.append(row)
        print(f"R29 EVAL {index+1}/{len(books)} r26={row['r26_bpb']:.6f} "
              f"r29={row['r29_bpb']:.6f}", file=sys.stderr, flush=True)
    arms = {}
    for name, v in sums.items():
        arms[name] = {"bpb": v["bits"] / total,
                      "bpc": v["bits"] / unicode_chars,
                      "nonspace_bpb": v["nonspace"] / nonspace,
                      "top1": v["top1"] / total, "top5": v["top5"] / total,
                      "positions": total}
    for name in ARMS[1:]:
        arms[name]["gain_bpb_vs_r26"] = arms["r26"]["bpb"] - arms[name]["bpb"]
        arms[name].update(whole_book_deltas(np.concatenate(losses["r26"]),
                                           np.concatenate(losses[name]),books,WARMUP))
    return {"arms": arms, "books": len(books), "positions": total,
            "characters": unicode_chars, "lexical_hit_rate": hits / total,
            "per_book": per_book, "elapsed_s": time.perf_counter() - start_clock}


def gate_pass(result: dict, minimum_improved: int) -> bool:
    arms = result["arms"]
    base, candidate = arms["r26"], arms["r29"]
    return bool(candidate["bpb"] < min(base["bpb"],
              arms["wrong_successor"]["bpb"], arms["unrelated_history"]["bpb"])
        and candidate["nonspace_bpb"] <= base["nonspace_bpb"]
        and candidate["top1"] >= base["top1"]
        and candidate["top5"] >= base["top5"]
        and candidate["books_improved"] >= minimum_improved)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus-root", type=Path, required=True)
    p.add_argument("--parent-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--checkpoint", type=Path)
    args = p.parse_args()
    optimizer = FrozenParentAdam(args.parent_checkpoint)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    paths, books = resolve_books(args.corpus_root, BYTES_PER_BOOK, DEV[1], SEED)
    verify_manifest(paths[:52], books[:52], manifest)
    hashes = [hashlib.sha256(book).hexdigest() for book in books]
    if (len(set(hashes)) != DEV[1] or len(set(p.name for p in paths)) != DEV[1]
            or tuple(hashes[132:156]) != R21_TEST_HASHES):
        raise ValueError("NaNROUTER ancestry, book uniqueness or split changed")
    prior = train_unigram_prior(books[:TRAIN_BOOKS])
    np.testing.assert_array_equal(prior, optimizer.prior)
    canonical = M3DefaultPredictor(prior)
    new = M3R29Predictor(prior)
    for byte in books[FIT[0]][:32]:
        canonical.observe(int(byte))
        new.observe(int(byte))
        np.testing.assert_array_equal(new.predict_logp(), canonical.predict_logp())
    fitted = train(books[FIT[0]:FIT[1]], prior, optimizer)
    names = [p.name for p in paths]
    gate = evaluate(books[GATE[0]:GATE[1]], names[GATE[0]:GATE[1]],prior,optimizer.theta)
    accepted_gate = gate_pass(gate, minimum_improved=3)
    dev = evaluate(books[DEV[0]:DEV[1]], names[DEV[0]:DEV[1]],prior,optimizer.theta)
    accepted_dev = accepted_gate and gate_pass(dev, minimum_improved=9) and (
        dev["arms"]["r29"]["whole_book_bootstrap_95pct_bits"][0] > 0)
    report = {"architecture": "M3 R29 lexical successor feedback on frozen R26",
              "parent_checkpoint_sha256": digest_of_parent(),
              "parent_R26_weights_or_optimizer_modified": False,
              "fit_indices": list(FIT), "gate_indices": list(GATE),
              "dev_indices": list(DEV), "sealed_test_indices": list(SEALED_TEST),
              "sealed_test_read": False, "fit": fitted,
              "fit_sha256": hashes[FIT[0]:FIT[1]],
              "gate_sha256": hashes[GATE[0]:GATE[1]],
              "dev_sha256": hashes[DEV[0]:DEV[1]],
              "gate": gate, "gate_pass": accepted_gate,
              "dev": dev, "dev_gate_pass": bool(accepted_dev),
              "promotion": "none; R26 remains canonical pending an independently sealed TEST"}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, sort_keys=True, indent=2)+"\n",encoding="utf-8")
    if args.checkpoint:
        optimizer.save(args.checkpoint, hashes)
    print(json.dumps({"fit": fitted, "gate_pass": accepted_gate,
                      "gate": gate["arms"], "dev_gate_pass": bool(accepted_dev),
                      "dev": dev["arms"]},sort_keys=True,indent=2))


if __name__ == "__main__":
    main()
