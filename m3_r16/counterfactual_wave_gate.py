"""M3 R16: preserve R15's temporal carrier, train a gated geometric residual.

The oscillator readout is the inherited reference. Spatial/reciprocal wave
specialists learn only *differences* from that carrier. A separate pair of
TRAIN books selects their scales; final DEV books remain untouched until
every parameter and geometry scale is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from m3_r14.predictive_probe import resolve_books, train_unigram_prior, metrics
from m3_r15.learned_wave_readout import collect, fit_readout, normalized_logits


GEOMETRY_ARMS = ("spatial_wave", "reciprocal_wave")
GEOMETRY_PENALTY = 0.4
SCALE_GRID = (0.0, 0.25, 0.5, 1.0)


def residual_features(data: dict, arm: str) -> np.ndarray:
    """True graph-only contrast after subtracting the oscillator carrier."""
    return data["features"][arm] - data["features"]["uncoupled"]


def fit_pair(data: dict, arm: str, oscillator: dict | None = None) -> dict:
    if oscillator is None:
        oscillator = fit_readout(data, "uncoupled")
    carrier = normalized_logits(
        data["base_logp"], data["features"]["uncoupled"], oscillator["weights"]
    )
    delta = residual_features(data, arm)
    specialist_data = {
        "base_logp": carrier,
        "targets": data["targets"],
        "features": {"residual": delta},
    }
    specialist = fit_readout(specialist_data, "residual", penalty=GEOMETRY_PENALTY)
    return {"oscillator": oscillator, "specialist": specialist}


def predict_pair(data: dict, arm: str, model: dict, scale: float) -> np.ndarray:
    carrier = normalized_logits(
        data["base_logp"], data["features"]["uncoupled"],
        model["oscillator"]["weights"],
    )
    if scale == 0:
        return carrier  # exact function-preserving fallback
    return normalized_logits(
        carrier, residual_features(data, arm),
        scale * model["specialist"]["weights"],
    )


def whole_book_deltas(reference: np.ndarray, candidate: np.ndarray,
                      books: list[bytes], warmup: int) -> dict:
    """Per-book loss differences with a descriptive book-bootstrap interval."""
    lengths = []
    for book in books:
        i = warmup
        while i < len(book) and book[i] & 0xC0 == 0x80:
            i += 1
        lengths.append(len(book) - i)
    if sum(lengths) != len(reference) or len(reference) != len(candidate):
        raise ValueError("book boundaries are misaligned with scored positions")
    deltas = []
    start = 0
    for n in lengths:
        deltas.append(float(np.mean(reference[start:start+n] - candidate[start:start+n])))
        start += n
    rng = np.random.default_rng(20261116)
    indices = rng.integers(0, len(books), size=(4096, len(books)))
    weighted = np.asarray(deltas)[indices]
    weights = np.asarray(lengths)[indices]
    samples = (weighted * weights).sum(axis=1) / weights.sum(axis=1)
    return {
        "per_book_gain_bits": deltas,
        "books_improved": sum(d > 0 for d in deltas),
        "whole_book_bootstrap_95pct_bits": [float(v) for v in np.percentile(samples, [2.5, 97.5])],
        "bootstrap_is_descriptive": True,
    }


def evaluate(train_books: list[bytes], dev_books: list[bytes], warmup: int = 1024) -> dict:
    if len(train_books) < 4 or len(dev_books) < 2:
        raise ValueError("need >=4 TRAIN books and >=2 DEV books")
    if min(map(len, train_books + dev_books)) <= warmup:
        raise ValueError("all books must exceed warmup")

    # TRAIN-only inner book holdout: no gate sample is used to fit weights or
    # prior during operator/scale selection.
    inner_fit = train_books[:-2]
    inner_gate = train_books[-2:]
    inner_prior = train_unigram_prior(inner_fit)
    fit_data = collect(inner_fit, inner_prior, warmup)
    gate_data = collect(inner_gate, inner_prior, warmup)
    gate_results = {}
    scales = {}
    inner_oscillator = fit_readout(fit_data, "uncoupled")
    for arm in GEOMETRY_ARMS:
        model = fit_pair(fit_data, arm, inner_oscillator)
        losses = []
        for scale in SCALE_GRID:
            p = predict_pair(gate_data, arm, model, scale)
            losses.append({"scale": scale, "bpb": metrics(p, gate_data)["bpb"]})
        choice = min(losses, key=lambda row: (row["bpb"], row["scale"]))
        scales[arm] = float(choice["scale"])
        gate_results[arm] = losses

    prior = train_unigram_prior(train_books)
    train_data = collect(train_books, prior, warmup)
    oscillator = fit_readout(train_data, "uncoupled")
    full_models = {arm: fit_pair(train_data, arm, oscillator) for arm in GEOMETRY_ARMS}
    dev = collect(dev_books, prior, warmup)

    reference = normalized_logits(
        dev["base_logp"], dev["features"]["uncoupled"], oscillator["weights"]
    )
    results = {
        "memory_only": metrics(dev["base_logp"], dev),
        "oscillator_carrier": metrics(reference, dev),
    }
    rows = np.arange(len(dev["targets"]))
    original_bits = -reference[rows, dev["targets"]] / np.log(2)
    for arm in GEOMETRY_ARMS:
        prediction = predict_pair(dev, arm, full_models[arm], scales[arm])
        outcome = metrics(prediction, dev)
        candidate_bits = -prediction[rows, dev["targets"]] / np.log(2)
        outcome["delta_bpb_vs_oscillator"] = results["oscillator_carrier"]["bpb"] - outcome["bpb"]
        outcome["selected_scale_train_only"] = scales[arm]
        outcome["oscillator_weights"] = [float(v) for v in full_models[arm]["oscillator"]["weights"]]
        outcome["geometry_residual_weights"] = [float(v) for v in full_models[arm]["specialist"]["weights"]]
        outcome.update(whole_book_deltas(original_bits, candidate_bits, dev_books, warmup))
        results["oscillator_plus_" + arm] = outcome

    return {
        "architecture": "M3 R16 counterfactual gated geometric specialist",
        "checkpoint_promotion": "none; R12 readout/checkpoint not recovered",
        "train_only_gate": "inner fit books / 2 distinct inner gate books; all TEST untouched",
        "gate_scale_grid": list(SCALE_GRID),
        "gate_losses": gate_results,
        "frozen_scales": scales,
        "geometry_penalty": GEOMETRY_PENALTY,
        "train_books": len(train_books),
        "dev_books": len(dev_books),
        "dev_positions": len(dev["targets"]),
        "dev_positions_per_second_all_arms": len(dev["targets"]) / dev["elapsed_s"],
        "dev": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--train-books", type=int, default=8)
    parser.add_argument("--dev-books", type=int, default=6)
    parser.add_argument("--bytes-per-book", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=160016)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    chosen, books = resolve_books(args.corpus_root, args.bytes_per_book,
                                  args.train_books + args.dev_books, args.seed)
    train, dev = books[:args.train_books], books[args.train_books:]
    result = evaluate(train, dev, args.warmup)
    result.update({
        "corpus": "NaNROUTER public-domain short stories batch 0004",
        "seed": args.seed,
        "bytes_per_book": args.bytes_per_book,
        "warmup": args.warmup,
        "train_filenames": [p.name for p in chosen[:args.train_books]],
        "dev_filenames": [p.name for p in chosen[args.train_books:]],
        "source_sha256": [hashlib.sha256(b).hexdigest() for b in books],
    })
    output = json.dumps(result, sort_keys=True, indent=2) + "\n"
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
