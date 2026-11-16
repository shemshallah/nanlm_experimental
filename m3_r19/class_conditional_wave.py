"""M3 R19: TRAIN-calibrated class-conditional spatial wave, fresh-book DEV.

Frozen R16/R17 oscillator and spatial coefficients. No R17/R18 TEST observations
participate in fitting, gate choice, prior construction, or DEV evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from m3_r14.predictive_probe import resolve_books, train_unigram_prior, metrics
from m3_r15.learned_wave_readout import collect
from m3_r16.counterfactual_wave_gate import whole_book_deltas
from m3_r17.locked_test import (
    SEED, BYTES_PER_BOOK, WARMUP, TRAIN_BOOKS,
    R16_SPATIAL_RESIDUAL_WEIGHTS, frozen_predictions,
)
from m3_r18.locked_replication import MANIFEST, PRIOR_PREFIX, NEW_TEST_BOOKS, verify_manifest

PRIOR_SEALED = PRIOR_PREFIX + NEW_TEST_BOOKS  # 52 frozen earlier documents
CALIBRATION_BOOKS = 16
GATE_BOOKS = 4
DEV_BOOKS = 12
TOTAL_BOOKS = PRIOR_SEALED + CALIBRATION_BOOKS + GATE_BOOKS + DEV_BOOKS
SCALES = (0.0, 0.25, 0.5, 1.0)
SPACE = 32
WRONG_ADDRESS_SHIFT = 37
ARMS = ("oscillator", "full_spatial", "space_only", "train_selected", "wrong_address")


def residual_scores(data: dict) -> np.ndarray:
    """Causal geometric logit differences, evaluated on all 256 candidate bytes."""
    delta = data["features"]["spatial_wave"] - data["features"]["uncoupled"]
    return np.einsum("nbk,k->nb", delta, R16_SPATIAL_RESIDUAL_WEIGHTS,
                     optimize=True)


def scaled_predictions(carrier: np.ndarray, scores: np.ndarray,
                       space_scale: float, nonspace_scale: float) -> np.ndarray:
    """Two scalar gates; zero/zero is exactly the inherited oscillator."""
    if space_scale == nonspace_scale == 0:
        return carrier
    channel_scales = np.full(256, nonspace_scale, dtype=np.float64)
    channel_scales[SPACE] = space_scale
    logits = carrier + scores * channel_scales[None, :]
    maximum = logits.max(axis=1, keepdims=True)
    return logits - maximum - np.log(np.exp(logits - maximum).sum(axis=1, keepdims=True))


def loss_sums(logp: np.ndarray, data: dict) -> tuple[float, float, int, int]:
    targets = data["targets"]
    mask = data["nonspace"]
    bits = -logp[np.arange(len(targets)), targets] / np.log(2)
    return float(bits.sum()), float(bits[mask].sum()), len(targets), int(mask.sum())


def grid_scores(books: list[bytes], prior: np.ndarray) -> list[dict]:
    """Aggregate TRAIN books with byte weighting; no DEV or TEST access."""
    totals = {(s, n): [0.0, 0.0] for s in SCALES for n in SCALES}
    baseline = [0.0, 0.0]
    counts = [0, 0]
    for book in books:
        data = collect([book], prior, WARMUP)
        carrier, _ = frozen_predictions(data)
        scores = residual_scores(data)
        b = loss_sums(carrier, data)
        baseline[0] += b[0]
        baseline[1] += b[1]
        counts[0] += b[2]
        counts[1] += b[3]
        for key, accum in totals.items():
            predicted = scaled_predictions(carrier, scores, *key)
            value = loss_sums(predicted, data)
            accum[0] += value[0]
            accum[1] += value[1]
    result = []
    for (s, n), total in totals.items():
        result.append({
            "space_scale": s, "nonspace_scale": n,
            "bpb": total[0] / counts[0],
            "nonspace_bpb": total[1] / counts[1],
            "gain_vs_oscillator": (baseline[0] - total[0]) / counts[0],
            "nonspace_gain_vs_oscillator": (baseline[1] - total[1]) / counts[1],
        })
    return result


def choose_gate(fit_rows: list[dict], gate_rows: list[dict]) -> dict:
    """Two-stage TRAIN-only choice with non-space no-regression constraint."""
    if len(fit_rows) != len(SCALES) ** 2 or len(gate_rows) != len(fit_rows):
        raise ValueError("incomplete class-conditional scale grid")
    ident = lambda row: (row["space_scale"], row["nonspace_scale"])
    fallback = (0.0, 0.0)
    # The first 16 TRAIN books form a shortlist. A distinct four-book TRAIN
    # gate chooses among them; the zero-scale fallback is always admissible.
    eligible_fit = [
        row for row in fit_rows
        if row["gain_vs_oscillator"] > 0 and row["nonspace_gain_vs_oscillator"] >= -1e-12
    ]
    shortlist = {ident(x) for x in sorted(
        eligible_fit,
        key=lambda r: (-r["gain_vs_oscillator"],
                       r["space_scale"] + r["nonspace_scale"])
    )[:4]}
    shortlist.add(fallback)
    finalists = [
        row for row in gate_rows if ident(row) in shortlist and
        row["nonspace_gain_vs_oscillator"] >= -1e-12 and
        row["gain_vs_oscillator"] >= -1e-12
    ]
    if not finalists:
        raise ValueError("zero-scale fallback unexpectedly absent")
    choice = min(finalists, key=lambda r: (
        r["bpb"], r["space_scale"] + r["nonspace_scale"]
    ))
    return {
        "space_scale": float(choice["space_scale"]),
        "nonspace_scale": float(choice["nonspace_scale"]),
        "fit_shortlist": [list(x) for x in sorted(shortlist)],
        "train_gate_bpb": float(choice["bpb"]),
        "train_gate_nonspace_bpb": float(choice["nonspace_bpb"]),
    }


def dev_predictions(data: dict, scales: tuple[float, float]) -> dict:
    carrier, full = frozen_predictions(data)
    scores = residual_scores(data)
    wrong = np.roll(scores, WRONG_ADDRESS_SHIFT, axis=1)
    return {
        "oscillator": carrier,
        "full_spatial": full,
        "space_only": scaled_predictions(carrier, scores, 1.0, 0.0),
        "train_selected": scaled_predictions(carrier, scores, *scales),
        "wrong_address": scaled_predictions(carrier, wrong, *scales),
    }


def evaluate_dev(dev_books: list[bytes], prior: np.ndarray,
                 scales: tuple[float, float], names: list[str]) -> dict:
    if len(dev_books) != DEV_BOOKS or len(names) != DEV_BOOKS:
        raise ValueError("R19 requires exactly twelve fresh DEV books")
    acc = {arm: {"bits": 0.0, "nonspace_bits": 0.0,
                 "top1_hits": 0.0, "top5_hits": 0.0} for arm in ARMS}
    n_total = n_nonspace = chars_total = 0
    elapsed = 0.0
    losses = {arm: [] for arm in ARMS}
    per_book = []
    for name, book in zip(names, dev_books):
        data = collect([book], prior, WARMUP)
        elapsed += data["elapsed_s"]
        predictions = dev_predictions(data, scales)
        n = len(data["targets"])
        n_total += n
        n_nonspace += int(data["nonspace"].sum())
        chars_total += data["unicode_chars"]
        row = {"filename": name, "positions": n,
               "sha256": hashlib.sha256(book).hexdigest()}
        for arm in ARMS:
            logp = predictions[arm]
            m = metrics(logp, data)
            bits = -logp[np.arange(n), data["targets"]] / np.log(2)
            losses[arm].append(bits)
            a = acc[arm]
            a["bits"] += float(bits.sum())
            a["nonspace_bits"] += float(bits[data["nonspace"]].sum())
            a["top1_hits"] += m["top1"] * n
            a["top5_hits"] += m["top5"] * n
            row[arm + "_bpb"] = m["bpb"]
        row["selected_gain_vs_oscillator"] = row["oscillator_bpb"] - row["train_selected_bpb"]
        per_book.append(row)
    if n_total <= 0 or n_nonspace <= 0 or n_nonspace >= n_total:
        raise ValueError("invalid DEV byte class counts")
    totals = {}
    for arm, a in acc.items():
        totals[arm] = {
            "positions": n_total, "bpb": a["bits"] / n_total,
            "bpc": a["bits"] / chars_total,
            "nonspace_bpb": a["nonspace_bits"] / n_nonspace,
            "top1": a["top1_hits"] / n_total,
            "top5": a["top5_hits"] / n_total,
        }
    carrier = totals["oscillator"]
    for arm in ARMS[1:]:
        totals[arm]["gain_bpb_vs_oscillator"] = carrier["bpb"] - totals[arm]["bpb"]
        totals[arm]["nonspace_gain_bpb_vs_oscillator"] = (
            carrier["nonspace_bpb"] - totals[arm]["nonspace_bpb"]
        )
    selected = totals["train_selected"]
    selected.update(whole_book_deltas(
        np.concatenate(losses["oscillator"]),
        np.concatenate(losses["train_selected"]), dev_books, WARMUP
    ))
    selected["dev_gate_pass"] = (
        selected["gain_bpb_vs_oscillator"] > 0 and
        selected["nonspace_gain_bpb_vs_oscillator"] >= 0 and
        selected["whole_book_bootstrap_95pct_bits"][0] > 0
    )
    return {"dev": totals, "per_dev_book": per_book,
            "dev_books": DEV_BOOKS, "dev_positions": n_total,
            "dev_positions_per_s_all_arms": n_total / elapsed,
            "promotion": "none; separate untouched TEST required"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if (manifest["seed"], manifest["bytes_per_book"], manifest["warmup"]) != (
        SEED, BYTES_PER_BOOK, WARMUP
    ):
        raise ValueError("R18 inherited split configuration altered")
    paths, books = resolve_books(args.corpus_root, BYTES_PER_BOOK, TOTAL_BOOKS, SEED)
    verify_manifest(paths[:PRIOR_SEALED], books[:PRIOR_SEALED], manifest)
    checksums = [hashlib.sha256(b).hexdigest() for b in books]
    if len(set(checksums)) != TOTAL_BOOKS or len({p.name for p in paths}) != TOTAL_BOOKS:
        raise ValueError("document overlap among inherited and R19 splits")
    original_train = books[:TRAIN_BOOKS]
    fit = books[PRIOR_SEALED:PRIOR_SEALED + CALIBRATION_BOOKS]
    gate = books[PRIOR_SEALED + CALIBRATION_BOOKS:PRIOR_SEALED + CALIBRATION_BOOKS + GATE_BOOKS]
    dev = books[PRIOR_SEALED + CALIBRATION_BOOKS + GATE_BOOKS:]
    dev_names = [p.name for p in paths[-DEV_BOOKS:]]
    prior = train_unigram_prior(original_train)
    started = time.perf_counter()
    fit_rows = grid_scores(fit, prior)
    gate_rows = grid_scores(gate, prior)
    choice = choose_gate(fit_rows, gate_rows)
    result = evaluate_dev(dev, prior,
                          (choice["space_scale"], choice["nonspace_scale"]), dev_names)
    result.update({
        "architecture": "M3 R19 class-conditional spatial correction",
        "coefficient_source": "R16 TRAIN; frozen through R17 and R18",
        "r12_checkpoint_loaded": False, "original_weights_changed": False,
        "original_train_prior_changed": False, "r17_r18_test_fitting": False,
        "corpus": manifest["corpus"], "seed": SEED, "warmup": WARMUP,
        "bytes_per_book": BYTES_PER_BOOK, "sealed_original_books": PRIOR_SEALED,
        "new_train_fit_books": CALIBRATION_BOOKS,
        "new_train_gate_books": GATE_BOOKS,
        "new_train_filenames": [p.name for p in paths[PRIOR_SEALED:-DEV_BOOKS]],
        "dev_filenames": dev_names, "dev_sha256": checksums[-DEV_BOOKS:],
        "grid": list(SCALES), "fit_results": fit_rows, "gate_results": gate_rows,
        "chosen_train_only_gate": choice,
        "wrong_address_shift": WRONG_ADDRESS_SHIFT,
        "total_elapsed_seconds": time.perf_counter() - started,
    })
    output = json.dumps(result, sort_keys=True, indent=2) + "\n"
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
