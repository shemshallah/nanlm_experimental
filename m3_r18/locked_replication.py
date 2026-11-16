"""M3 R18: pre-fingerprinted 32-book replication of R17 frozen spatial wave.

No fit, optimizer, scale selection, model promotion, or TEST-conditioned fallback.
The original eight TRAIN documents alone define the unchanged unigram prior.
Each new book is scored independently, one at a time, to bound peak RAM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from m3_r14.predictive_probe import metrics, resolve_books, train_unigram_prior
from m3_r15.learned_wave_readout import collect
from m3_r16.counterfactual_wave_gate import whole_book_deltas
from m3_r17.locked_test import (
    BYTES_PER_BOOK, DEV_BOOKS, SEED, TEST_BOOKS, TRAIN_BOOKS, WARMUP,
    R16_OSCILLATOR_WEIGHTS, R16_SPATIAL_RESIDUAL_WEIGHTS,
    frozen_predictions, verify_r16_prefix,
)

MANIFEST = Path(__file__).with_name("frozen_split.json")
NEW_TEST_BOOKS = 32
PRIOR_PREFIX = TRAIN_BOOKS + DEV_BOOKS + TEST_BOOKS


def verify_manifest(selected: list[Path], books: list[bytes], manifest: dict) -> list[str]:
    """Enforce exact predeclared documents and unchanged R16/R17 ancestry."""
    if (manifest.get("seed"), manifest.get("bytes_per_book"), manifest.get("warmup")) != (
        SEED, BYTES_PER_BOOK, WARMUP
    ):
        raise ValueError("precommitted corpus selection parameters changed")
    if len(selected) != PRIOR_PREFIX + NEW_TEST_BOOKS or len(books) != len(selected):
        raise ValueError("expected original 20 documents and exactly 32 untouched TEST books")
    names = [p.name for p in selected]
    if len(set(names)) != len(names):
        raise ValueError("duplicate document filename across splits")
    # Verifies original R16 TRAIN and DEV document fingerprints and all overlaps.
    hashes = verify_r16_prefix(books[:PRIOR_PREFIX])
    expected_old_test = manifest.get("r17_test_sha256")
    expected_new_test = manifest.get("r18_test_sha256")
    if not isinstance(expected_old_test, list) or len(expected_old_test) != TEST_BOOKS:
        raise ValueError("R17 TEST fingerprint manifest missing or malformed")
    if tuple(hashes[TRAIN_BOOKS + DEV_BOOKS:]) != tuple(expected_old_test):
        raise ValueError("previous R17 TEST identities changed; evaluation refused")
    if not isinstance(expected_new_test, list) or len(expected_new_test) != NEW_TEST_BOOKS:
        raise ValueError("R18 precommitted TEST fingerprints missing or malformed")
    new_hashes = [hashlib.sha256(b).hexdigest() for b in books[PRIOR_PREFIX:]]
    if tuple(new_hashes) != tuple(expected_new_test):
        raise ValueError("R18 new TEST identities changed; evaluation refused")
    if len(set(hashes + new_hashes)) != len(hashes) + len(new_hashes):
        raise ValueError("overlap in TRAIN, DEV, R17 TEST or R18 TEST")
    return hashes + new_hashes


def read_frozen_corpus(root: Path, manifest: dict) -> tuple[list[Path], list[bytes], list[str]]:
    # R16/R17 identities are checked before accessing the new R18 document bytes.
    old_paths, old_books = resolve_books(root, BYTES_PER_BOOK, PRIOR_PREFIX, SEED)
    verify_r16_prefix(old_books)
    old_test_hashes = [hashlib.sha256(b).hexdigest() for b in old_books[TRAIN_BOOKS + DEV_BOOKS:]]
    if old_test_hashes != manifest.get("r17_test_sha256"):
        raise ValueError("R17 TEST prefix changed")
    paths, books = resolve_books(root, BYTES_PER_BOOK, PRIOR_PREFIX + NEW_TEST_BOOKS, SEED)
    if paths[:PRIOR_PREFIX] != old_paths or books[:PRIOR_PREFIX] != old_books:
        raise ValueError("corpus order or original prefix changed during selection")
    hashes = verify_manifest(paths, books, manifest)
    return paths, books, hashes


def summarize_arm(logps: list[np.ndarray], data: dict) -> tuple[list[dict], list[np.ndarray]]:
    """Book-scoped predictions and their scoring vectors; no persistent features."""
    return [metrics(lp, data) for lp in logps], [
        -lp[np.arange(len(data["targets"])), data["targets"]] / np.log(2)
        for lp in logps
    ]


def evaluate_streamed(train_books: list[bytes], test_books: list[bytes],
                      names: list[str] | None = None,
                      fingerprints: list[str] | None = None,
                      warmup: int = WARMUP) -> dict:
    if len(train_books) != TRAIN_BOOKS or len(test_books) != NEW_TEST_BOOKS:
        raise ValueError("R18 needs original 8 TRAIN and exactly 32 new TEST documents")
    if min(map(len, train_books + test_books)) <= warmup:
        raise ValueError("document too short for locked warmup")
    names = names if names is not None else [f"test-{i:02d}" for i in range(len(test_books))]
    fingerprints = fingerprints if fingerprints is not None else [
        hashlib.sha256(b).hexdigest() for b in test_books
    ]
    if len(names) != NEW_TEST_BOOKS or len(fingerprints) != NEW_TEST_BOOKS:
        raise ValueError("inconsistent TEST provenance lengths")

    prior = train_unigram_prior(train_books)
    labels = ("memory_only", "oscillator_carrier", "oscillator_plus_spatial_wave")
    totals = {label: {"bits": 0.0, "nonspace_bits": 0.0,
                      "correct_1": 0.0, "correct_5": 0.0} for label in labels}
    before_losses, after_losses, per_book = [], [], []
    positions = nonspace_positions = characters = 0
    total_elapsed = 0.0

    for index, book in enumerate(test_books):
        data = collect([book], prior, warmup)
        oscillator, spatial = frozen_predictions(data)
        logps = [data["base_logp"], oscillator, spatial]
        arm_metrics, arm_bits = summarize_arm(logps, data)
        targets = data["targets"]
        n = len(targets)
        mask = data["nonspace"]
        n_nonspace = int(mask.sum())
        if not n or not n_nonspace or n_nonspace == n:
            raise ValueError("TEST book has no scores or lacks space/nonspace attribution")
        total_elapsed += data["elapsed_s"]
        positions += n
        nonspace_positions += n_nonspace
        characters += data["unicode_chars"]
        for label, outcome, bits in zip(labels, arm_metrics, arm_bits):
            acc = totals[label]
            acc["bits"] += float(bits.sum())
            acc["nonspace_bits"] += float(bits[mask].sum())
            acc["correct_1"] += outcome["top1"] * n
            acc["correct_5"] += outcome["top5"] * n
        before_losses.append(arm_bits[1])
        after_losses.append(arm_bits[2])
        per_book.append({
            "filename": names[index], "sha256": fingerprints[index], "positions": n,
            "oscillator_bpb": arm_metrics[1]["bpb"],
            "spatial_bpb": arm_metrics[2]["bpb"],
            "spatial_gain_bpb": arm_metrics[1]["bpb"] - arm_metrics[2]["bpb"],
            "nonspace_gain_bpb": float((arm_bits[1][mask] - arm_bits[2][mask]).mean()),
            "space_gain_bpb": float((arm_bits[1][~mask] - arm_bits[2][~mask]).mean()),
            "oscillator_top1": arm_metrics[1]["top1"],
            "spatial_top1": arm_metrics[2]["top1"],
        })

    results = {}
    for label in labels:
        a = totals[label]
        results[label] = {
            "positions": positions,
            "bpb": a["bits"] / positions,
            "bpc": a["bits"] / characters,
            "nonspace_bpb": a["nonspace_bits"] / nonspace_positions,
            "top1": a["correct_1"] / positions,
            "top5": a["correct_5"] / positions,
        }
    carrier = results["oscillator_carrier"]
    spatial = results["oscillator_plus_spatial_wave"]
    spatial["delta_bpb_vs_oscillator"] = carrier["bpb"] - spatial["bpb"]
    spatial["delta_bpb_vs_memory"] = results["memory_only"]["bpb"] - spatial["bpb"]
    spatial["nonspace_gain_bpb"] = carrier["nonspace_bpb"] - spatial["nonspace_bpb"]
    n_space = positions - nonspace_positions
    spatial["space_gain_bpb"] = (
        (totals["oscillator_carrier"]["bits"] - totals["oscillator_carrier"]["nonspace_bits"])
        - (totals["oscillator_plus_spatial_wave"]["bits"] - totals["oscillator_plus_spatial_wave"]["nonspace_bits"])
    ) / n_space
    spatial.update(whole_book_deltas(np.concatenate(before_losses),
                                      np.concatenate(after_losses), test_books, warmup))
    return {
        "architecture": "M3 R18: frozen R16 spatial wave, 32 new book TEST replication",
        "r12_checkpoint_loaded": False,
        "weights_refit": False,
        "scale_reselected": False,
        "train_books": TRAIN_BOOKS,
        "new_test_books": NEW_TEST_BOOKS,
        "new_test_positions": positions,
        "new_test_characters": characters,
        "new_test_positions_per_second_all_arms": positions / total_elapsed,
        "test": results,
        "per_book": per_book,
        "promotion": "none; effect must survive broader domains and capacity-matched controls",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    started = time.perf_counter()
    paths, books, hashes = read_frozen_corpus(args.corpus_root, manifest)
    result = evaluate_streamed(
        books[:TRAIN_BOOKS], books[PRIOR_PREFIX:],
        [p.name for p in paths[PRIOR_PREFIX:]], hashes[PRIOR_PREFIX:],
    )
    result.update({
        "corpus": manifest["corpus"], "seed": SEED,
        "bytes_per_book": BYTES_PER_BOOK, "warmup": WARMUP,
        "original_train_sha256": hashes[:TRAIN_BOOKS],
        "original_dev_sha256": hashes[TRAIN_BOOKS:TRAIN_BOOKS + DEV_BOOKS],
        "r17_test_sha256": hashes[TRAIN_BOOKS + DEV_BOOKS:PRIOR_PREFIX],
        "r18_test_sha256": hashes[PRIOR_PREFIX:],
        "frozen_oscillator_weights": R16_OSCILLATOR_WEIGHTS.tolist(),
        "frozen_spatial_residual_weights": R16_SPATIAL_RESIDUAL_WEIGHTS.tolist(),
        "source_r17_run": "https://github.com/shemshallah/nanlm_experimental/actions/runs/43915921098",
        "total_elapsed_seconds": time.perf_counter() - started,
    })
    output = json.dumps(result, sort_keys=True, indent=2) + "\n"
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
