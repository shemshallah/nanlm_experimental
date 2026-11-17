"""One-shot independent NaNROUTER TEST for the previously frozen R26 readout.

This script loads the exact trained R26 NPZ and DEV report from the successful
GitHub Actions artifact and validates their SHA-256 against the precommitted
lock. It DOES NOT optimize, select, recalibrate or update model coefficients.
It scores only the previously untouched book indices 276..299, once each,
with causal updates strictly after target-byte prediction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from m3_r14.predictive_probe import resolve_books, train_unigram_prior
from m3_r17.locked_test import SEED, TRAIN_BOOKS, BYTES_PER_BOOK, WARMUP
from m3_r18.locked_replication import MANIFEST, verify_manifest
from m3_r22.evaluate import R21_TEST_HASHES
from m3_r26.train_dev import evaluate

LOCK = Path(__file__).with_name("r26_frozen_candidate.json")
TEST_FIRST, TEST_END = 276, 300


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as inp:
        for block in iter(lambda: inp.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def load_frozen(checkpoint: Path, report: Path) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["sealed_test_indices"] != [TEST_FIRST, TEST_END]:
        raise ValueError("sealed TEST split changed")
    if digest(checkpoint) != lock["training_checkpoint_sha256"]:
        raise ValueError("checkpoint digest no longer matches frozen R26")
    if digest(report) != lock["training_dev_report_sha256"]:
        raise ValueError("train/gate/DEV report digest mismatch")
    results = json.loads(report.read_text(encoding="utf-8"))
    if not results["gate_pass"] or not results["dev_gate_pass"]:
        raise ValueError("candidate did not pass the precommitted development gates")
    for original, key in (("training_indices", "fit_indices"),
                          ("gate_indices", "gate_indices"),
                          ("dev_indices", "dev_indices")):
        if lock[original] != results[key]:
            raise ValueError("R26 corpus split changed after training")
    with np.load(checkpoint, allow_pickle=False) as npz:
        theta = np.asarray(npz["theta"], dtype=np.float64).copy()
        steps = np.asarray(npz["adam_steps"], dtype=np.int64).copy()
        prior = np.asarray(npz["prior"], dtype=np.float64).copy()
        fit_hashes = npz["fit_sha256"].tolist()
        sealed_indices = npz["sealed_test_indices"].tolist()
    if (theta.shape != (14,) or not np.isfinite(theta).all()
            or not np.array_equal(theta, np.asarray(lock["theta"], dtype=np.float64))
            or steps.tolist() != lock["optimizer_steps"]
            or prior.shape != (256,) or np.any(prior <= 0)
            or fit_hashes != results["fit_sha256"]
            or sealed_indices != [TEST_FIRST, TEST_END]):
        raise ValueError("trained R26 checkpoint is not exactly the frozen candidate")
    return theta, prior, lock, results


def pass_locked_test(measured: dict, lock: dict) -> bool:
    arms = measured["arms"]
    baseline, parent, candidate = arms["r24"], arms["r25"], arms["r26"]
    req = lock["test_requirements"]
    return bool(measured["books"] == TEST_END - TEST_FIRST
                and candidate["bpb"] < min(baseline["bpb"], parent["bpb"])
                and candidate["nonspace_bpb"] <= baseline["nonspace_bpb"]
                and candidate["whole_book_bootstrap_95pct_bits"][0] > 0
                and candidate["books_improved"] >= req["minimum_improved_books"]
                and candidate["bpb"] < arms["wrong_successor"]["bpb"]
                and candidate["bpb"] < arms["unrelated_history"]["bpb"]
                and candidate["top1"] >= baseline["top1"]
                and candidate["top5"] >= baseline["top5"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    theta, locked_prior, lock, train_report = load_frozen(args.checkpoint, args.train_report)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    paths, books = resolve_books(args.corpus_root, BYTES_PER_BOOK, TEST_END, SEED)
    verify_manifest(paths[:52], books[:52], manifest)
    hashes = [hashlib.sha256(b).hexdigest() for b in books]
    if (len(set(hashes)) != TEST_END or len(set(p.name for p in paths)) != TEST_END
            or tuple(hashes[132:156]) != R21_TEST_HASHES):
        raise ValueError("corpus changed, duplicated, or prior sealed TEST was modified")
    for start, end, key in ((244, 260, "fit_sha256"),
                            (260, 264, "gate_sha256"),
                            (264, 276, "dev_sha256")):
        if hashes[start:end] != train_report[key]:
            raise ValueError("predeclared training/DEV identity changed")
    prior = train_unigram_prior(books[:TRAIN_BOOKS])
    np.testing.assert_array_equal(prior, locked_prior)
    started = time.perf_counter()
    measured = evaluate(books[TEST_FIRST:TEST_END],
                        [p.name for p in paths[TEST_FIRST:TEST_END]],prior,theta)
    accepted = pass_locked_test(measured, lock)
    result = {"architecture": "M3 R27 one-shot locked TEST of frozen R26",
              "frozen_checkpoint_sha256": lock["training_checkpoint_sha256"],
              "frozen_trained_commit": lock["trained_commit"],
              "frozen_training_run": lock["training_run"],
              "locked_test_indices": [TEST_FIRST,TEST_END],
              "locked_test_sha256": hashes[TEST_FIRST:TEST_END],
              "test": measured, "independent_test_gate": bool(accepted),
              "trained_or_selected_on_test": False,
              "model_weights_or_optimizer_updated": False,
              "original_R24_default_untouched": True,
              "elapsed_s": time.perf_counter()-started,
              "promotion": "eligible only if independent_test_gate; no automatic promotion"}
    output = json.dumps(result, sort_keys=True, indent=2)+"\n"
    print(json.dumps({"arms": measured["arms"],
                      "books": measured["books"],
                      "independent_test_gate": bool(accepted)},indent=2,sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
