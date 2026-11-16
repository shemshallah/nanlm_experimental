"""M3 R20: gated directed mirror-pair and triangle successor memory.

R19 is the exact function-preserving fallback. R16/R17/R18 TEST and the
reserved R19 TEST are excluded from training and DEV scoring.
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
from m3_r17.locked_test import SEED, BYTES_PER_BOOK, WARMUP, TRAIN_BOOKS, frozen_predictions
from m3_r18.locked_replication import MANIFEST, verify_manifest
from m3_r19.class_conditional_wave import scaled_predictions, residual_scores
from m3_r20.signed_mirror_memory import causal_successor_deltas, MIRROR_PAIRS

SEALED_R18_END = 52
R19_TRAIN_DEV_END = 84
R19_RESERVED_TEST_END = 100
FIT_COUNT = 16
GATE_COUNT = 4
DEV_COUNT = 12
TOTAL_BOOKS = R19_RESERVED_TEST_END + FIT_COUNT + GATE_COUNT + DEV_COUNT
R19_SPACE_SCALE = 0.25
R19_NONSPACE_SCALE = 1.0
GRIDS = (0.0, .25, .5, 1.0)
SHIFT = 37
ARMS = ("r19_frozen", "pair_only", "triangle_only", "train_selected",
        "wrong_address", "wrong_context")


def frozen_r19(data: dict) -> np.ndarray:
    carrier, _ = frozen_predictions(data)
    return scaled_predictions(carrier, residual_scores(data),
                              R19_SPACE_SCALE, R19_NONSPACE_SCALE)


def combined_logp(base: np.ndarray, pair: np.ndarray, triangle: np.ndarray,
                  pair_scale: float, tri_scale: float) -> np.ndarray:
    if pair_scale == tri_scale == 0:
        return base
    logits = base + pair_scale * pair + tri_scale * triangle
    maximum = logits.max(axis=1, keepdims=True)
    return logits - maximum - np.log(np.exp(logits - maximum).sum(axis=1, keepdims=True))


def collect_candidate(book: bytes, prior: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, dict]:
    data = collect([book], prior, WARMUP)
    base = frozen_r19(data)
    pair, triangle, invariant = causal_successor_deltas(
        book, prior, data["base_logp"], WARMUP
    )
    if not invariant["pair_nonnegative"] or invariant["edge_mass"] != invariant["retained_edges"]:
        raise ValueError("mirror-pair mass conservation failed")
    if invariant["triangle_mass"] != invariant["retained_triangles"]:
        raise ValueError("ordered triangle mass conservation failed")
    return data, base, pair, triangle, invariant


def score_grid(books: list[bytes], prior: np.ndarray) -> list[dict]:
    total = {(a, b): [0.0, 0.0] for a in GRIDS for b in GRIDS}
    reference = [0.0, 0.0]
    count = nonspace = 0
    for book in books:
        data, base, pair, tri, _ = collect_candidate(book, prior)
        target = data["targets"]
        nonspace_mask = data["nonspace"]
        rows = np.arange(len(target))
        base_bits = -base[rows, target] / np.log(2)
        reference[0] += float(base_bits.sum())
        reference[1] += float(base_bits[nonspace_mask].sum())
        count += len(target)
        nonspace += int(nonspace_mask.sum())
        for scales, agg in total.items():
            lp = combined_logp(base, pair, tri, *scales)
            bits = -lp[rows, target] / np.log(2)
            agg[0] += float(bits.sum())
            agg[1] += float(bits[nonspace_mask].sum())
    results = []
    for (a, b), agg in total.items():
        results.append({
            "pair_scale": a, "triangle_scale": b,
            "bpb": agg[0] / count,
            "nonspace_bpb": agg[1] / nonspace,
            "gain_bpb": (reference[0] - agg[0]) / count,
            "nonspace_gain_bpb": (reference[1] - agg[1]) / nonspace,
        })
    return results


def choose_scales(fit: list[dict], gate: list[dict]) -> dict:
    if len(fit) != 16 or len(gate) != 16:
        raise ValueError("missing two-level scale grid")
    key = lambda r: (r["pair_scale"], r["triangle_scale"])
    eligible = [r for r in fit if r["gain_bpb"] > 0 and r["nonspace_gain_bpb"] >= -1e-12]
    shortlist = {key(r) for r in sorted(
        eligible, key=lambda r: (-r["gain_bpb"], r["pair_scale"]+r["triangle_scale"])
    )[:4]}
    shortlist.add((0.0, 0.0))
    choices = [r for r in gate if key(r) in shortlist and
               r["gain_bpb"] >= -1e-12 and r["nonspace_gain_bpb"] >= -1e-12]
    winner = min(choices, key=lambda r: (
        r["bpb"], r["pair_scale"] + r["triangle_scale"]
    ))
    return {"pair_scale": winner["pair_scale"], "triangle_scale": winner["triangle_scale"],
            "gate_bpb": winner["bpb"], "gate_nonspace_bpb": winner["nonspace_bpb"],
            "train_only_shortlist": [list(k) for k in sorted(shortlist)]}


def dev_predictions(base: np.ndarray, pair: np.ndarray, tri: np.ndarray,
                    selected: tuple[float, float]) -> dict[str, np.ndarray]:
    return {
        "r19_frozen": base,
        "pair_only": combined_logp(base, pair, tri, 1.0, 0.0),
        "triangle_only": combined_logp(base, pair, tri, 0.0, 1.0),
        "train_selected": combined_logp(base, pair, tri, *selected),
        "wrong_address": combined_logp(base, np.roll(pair, SHIFT, axis=1),
                                       np.roll(tri, SHIFT, axis=1), *selected),
        "wrong_context": combined_logp(base, pair, np.roll(tri, SHIFT, axis=1),
                                       *selected),
    }


def evaluate_dev(books: list[bytes], names: list[str], prior: np.ndarray,
                 selected: tuple[float, float]) -> dict:
    if len(books) != DEV_COUNT or len(names) != DEV_COUNT:
        raise ValueError("twelve fresh DEV books required")
    totals = {arm: {"bits":0.0, "nonspace_bits":0.0,
                     "top1_hits":0.0, "top5_hits":0.0} for arm in ARMS}
    losses = {arm: [] for arm in ARMS}
    per_book = []
    position_count = nonspace_count = char_count = 0
    elapsed = 0.0
    for name, book in zip(names, books):
        data, base, pair, tri, invariants = collect_candidate(book, prior)
        predictions = dev_predictions(base, pair, tri, selected)
        n = len(data["targets"])
        position_count += n
        nonspace_count += int(data["nonspace"].sum())
        char_count += data["unicode_chars"]
        elapsed += data["elapsed_s"]
        row = {"filename": name, "sha256": hashlib.sha256(book).hexdigest(),
               "positions": n, "mirror_pair_invariants": invariants}
        for arm, lp in predictions.items():
            m = metrics(lp, data)
            bits = -lp[np.arange(n), data["targets"]] / np.log(2)
            losses[arm].append(bits)
            t = totals[arm]
            t["bits"] += float(bits.sum())
            t["nonspace_bits"] += float(bits[data["nonspace"]].sum())
            t["top1_hits"] += m["top1"] * n
            t["top5_hits"] += m["top5"] * n
            row[arm + "_bpb"] = m["bpb"]
        row["selected_gain_bpb"] = row["r19_frozen_bpb"] - row["train_selected_bpb"]
        per_book.append(row)
    if not (0 < nonspace_count < position_count and char_count > 0):
        raise ValueError("invalid DEV scoring counts")
    results = {}
    for arm, t in totals.items():
        results[arm] = {"positions": position_count,
                        "bpb": t["bits"] / position_count,
                        "bpc": t["bits"] / char_count,
                        "nonspace_bpb": t["nonspace_bits"] / nonspace_count,
                        "top1": t["top1_hits"] / position_count,
                        "top5": t["top5_hits"] / position_count}
    baseline = results["r19_frozen"]
    for arm in ARMS[1:]:
        results[arm]["gain_bpb_vs_r19"] = baseline["bpb"] - results[arm]["bpb"]
        results[arm]["nonspace_gain_bpb_vs_r19"] = (
            baseline["nonspace_bpb"] - results[arm]["nonspace_bpb"]
        )
    result = results["train_selected"]
    result.update(whole_book_deltas(
        np.concatenate(losses["r19_frozen"]), np.concatenate(losses["train_selected"]),
        books, WARMUP
    ))
    result["dev_gate_pass"] = (
        result["gain_bpb_vs_r19"] > 0 and
        result["nonspace_gain_bpb_vs_r19"] >= 0 and
        result["whole_book_bootstrap_95pct_bits"][0] > 0
    )
    return {"dev": results, "per_book": per_book,
            "dev_positions": position_count, "dev_books": DEV_COUNT,
            "dev_positions_per_s_frozen_wave_and_successor_memory": position_count / elapsed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    paths, books = resolve_books(args.corpus_root, BYTES_PER_BOOK, TOTAL_BOOKS, SEED)
    verify_manifest(paths[:SEALED_R18_END], books[:SEALED_R18_END], manifest)
    names = [p.name for p in paths]
    hashes = [hashlib.sha256(b).hexdigest() for b in books]
    if len(set(names)) != TOTAL_BOOKS or len(set(hashes)) != TOTAL_BOOKS:
        raise ValueError("TRAIN/DEV/reserved TEST duplicate documents")
    prior = train_unigram_prior(books[:TRAIN_BOOKS])
    fit_start = R19_RESERVED_TEST_END
    gate_start = fit_start + FIT_COUNT
    dev_start = gate_start + GATE_COUNT
    started = time.perf_counter()
    fit_rows = score_grid(books[fit_start:gate_start], prior)
    gate_rows = score_grid(books[gate_start:dev_start], prior)
    selected = choose_scales(fit_rows, gate_rows)
    result = evaluate_dev(
        books[dev_start:], names[dev_start:], prior,
        (selected["pair_scale"], selected["triangle_scale"])
    )
    result.update({
        "architecture": "M3 R20 signed C(256,2) pair plus sparse ordered triangle",
        "mirror_addresses": MIRROR_PAIRS, "heads": 256, "signed_rails": 2,
        "ordered_triangle_memory": "sparse 8192-transition causal window",
        "original_r19_predictions_preserved_when_scales_zero": True,
        "r12_checkpoint_loaded": False, "promotion": "none; new independent TEST required",
        "r19_frozen_space_scale": R19_SPACE_SCALE,
        "r19_frozen_nonspace_scale": R19_NONSPACE_SCALE,
        "sealed_documents": SEALED_R18_END,
        "r19_reserved_test_indices": [R19_TRAIN_DEV_END, R19_RESERVED_TEST_END],
        "new_fit_indices": [fit_start, gate_start], "new_gate_indices": [gate_start, dev_start],
        "new_dev_indices": [dev_start, TOTAL_BOOKS],
        "new_dev_sha256": hashes[dev_start:],
        "grid": list(GRIDS), "fit_grid": fit_rows, "gate_grid": gate_rows,
        "selected_train_only": selected, "wrong_address_shift": SHIFT,
        "corpus": manifest["corpus"], "seed": SEED, "warmup": WARMUP,
        "bytes_per_book": BYTES_PER_BOOK,
        "elapsed_s": time.perf_counter() - started,
    })
    output = json.dumps(result, sort_keys=True, indent=2) + "\n"
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
