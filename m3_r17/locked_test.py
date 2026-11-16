"""M3 R17: one-shot, book-disjoint TEST of the frozen R16 spatial wave.

The R16 DEVELOPMENT result selected the spatial arm. This module contains
its exact TRAIN-fitted weights and TRAIN-selected gate. Neither TEST labels
nor DEV labels can update coefficients, priors, or scales here.

R17 is an attribution/replication experiment, not a recovered R12 checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from m3_r14.predictive_probe import metrics, resolve_books, train_unigram_prior
from m3_r15.learned_wave_readout import collect, normalized_logits
from m3_r16.counterfactual_wave_gate import residual_features, whole_book_deltas


SEED = 160016
TRAIN_BOOKS = 8
DEV_BOOKS = 6
TEST_BOOKS = 6
BYTES_PER_BOOK = 4096
WARMUP = 1024
ARM = "spatial_wave"  # chosen before TEST from R16 DEV
GATE = 1.0  # R16 TRAIN-only selection

# First 14 document SHA-256 values reported by R16 successful CI #43806939769.
# Verify these BEFORE reading/scoring any previously untouched TEST labels.
R16_TRAIN_DEV_SHA256 = (
    "263b9f90c335da184ac36ee83672b89918a3dbf432d38f24ddb24c50cba9e285",
    "9688b67a2e0d5d34da96fbdf5b152a82960c010a38d72d6e61a280f73af02711",
    "21b538576af234c663d1988bb53ab7c44938c259e8763fa397059a25ac195014",
    "93247f29ada68071d326136fc7e806a2642db4259b85d76967f564e2a47630e7",
    "484821767e72da885ce88e53177a9e75c09fd4ebe47fa2e86b3e7de979ae5d98",
    "81a4098e8b1aefa01f025c4ccba1513a789199d64223f4383c13fec3229c5afb",
    "01161c55cd91c549147cca95ec7d0e1edc786b252044885ba3db09cff2c95b4d",
    "4c126376ac2fa1c05230fc872a3c8a2d00b309628300854602cc4c9f93b392dc",
    "13be59b2029b9c50a4082faca2f767d2d0589f5539f5aae64673833f039be843",
    "15a0d1485c3f6910fffbba55584c9f35d3204ad0ea6a472b1a1f625ba44202af",
    "6debb20ff23956d7fdd0276890ed1c79b6eda2569095fbab2e4e17cf7f32ee63",
    "696e729fe242dfb2a48b56eeadf4374ce94490b1af46b412297368d7bc8764e0",
    "ea676547f1a2249c69119fb4f140a22efdd9c3b9f0f6b9a6fb3b072c3e103c49",
    "d1b581aedeb162be33620ae8072806ed835664e8eedb6ba0281c78a97635b04d",
)

# Frozen TRAIN-derived readouts emitted by R16 CI; no re-fitting in R17.
R16_OSCILLATOR_WEIGHTS = np.array((
    -0.008261844901553878, -0.015209865629857202,
     0.038325515357472924, 0.017215445836043123,
    -0.0024008059305847228,
), dtype=np.float64)
R16_SPATIAL_RESIDUAL_WEIGHTS = np.array((
    -0.0024974394624970894, -0.004024377422434821,
    -0.02456422496602529, 0.02502132817052411,
    -0.00701712038563265,
), dtype=np.float64)


def verify_r16_prefix(books: list[bytes]) -> list[str]:
    if len(books) != TRAIN_BOOKS + DEV_BOOKS + TEST_BOOKS:
        raise ValueError("unexpected book count: R16 TRAIN/DEV plus six fresh TEST required")
    fingerprints = [hashlib.sha256(book).hexdigest() for book in books]
    if tuple(fingerprints[:TRAIN_BOOKS + DEV_BOOKS]) != R16_TRAIN_DEV_SHA256:
        raise ValueError("R16 TRAIN/DEV document identity mismatch; TEST evaluation refused")
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("overlapping TRAIN/DEV/TEST documents")
    return fingerprints


def frozen_predictions(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Apply R16's frozen oscillator and selected wave, with no fit calls."""
    oscillator = normalized_logits(
        data["base_logp"], data["features"]["uncoupled"], R16_OSCILLATOR_WEIGHTS
    )
    spatial = normalized_logits(
        oscillator, residual_features(data, ARM), GATE * R16_SPATIAL_RESIDUAL_WEIGHTS
    )
    return oscillator, spatial


def evaluate_locked(train_books: list[bytes], test_books: list[bytes]) -> dict:
    if len(train_books) != TRAIN_BOOKS or len(test_books) != TEST_BOOKS:
        raise ValueError("frozen R16 requires exactly eight TRAIN and six TEST books")
    if min(map(len, train_books + test_books)) <= WARMUP:
        raise ValueError("all documents must exceed the locked warmup")
    # The only data-dependent prior uses original R16 TRAIN documents.
    prior = train_unigram_prior(train_books)
    test = collect(test_books, prior, WARMUP)
    carrier, spatial = frozen_predictions(test)
    m0 = metrics(test["base_logp"], test)
    m1 = metrics(carrier, test)
    m2 = metrics(spatial, test)
    indices = np.arange(len(test["targets"]))
    before = -carrier[indices, test["targets"]] / np.log(2)
    after = -spatial[indices, test["targets"]] / np.log(2)
    m2["delta_bpb_vs_oscillator"] = m1["bpb"] - m2["bpb"]
    m2["delta_bpb_vs_memory"] = m0["bpb"] - m2["bpb"]
    m2.update(whole_book_deltas(before, after, test_books, WARMUP))
    # No unrecorded reweighting, test-target fitting, or fallback selection.
    return {
        "architecture": "M3 R17 frozen R16 spatial-wave TEST replication",
        "source_run": "https://github.com/shemshallah/nanlm_experimental/actions/runs/43806939769",
        "r12_checkpoint_loaded": False,
        "model_selection": "spatial arm from R16 DEV; weights+scale frozen before TEST",
        "train_books": len(train_books),
        "dev_books_used_for_fitting": 0,
        "test_books": len(test_books),
        "test_positions": len(test["targets"]),
        "test_positions_per_second_all_arms": len(test["targets"]) / test["elapsed_s"],
        "locked_scale": GATE,
        "locked_oscillator_weights": R16_OSCILLATOR_WEIGHTS.tolist(),
        "locked_spatial_residual_weights": R16_SPATIAL_RESIDUAL_WEIGHTS.tolist(),
        "test": {"memory_only": m0, "oscillator_carrier": m1, "oscillator_plus_spatial_wave": m2},
        "promotion": "requires separate judgment; TEST not used to tune",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    chosen, books = resolve_books(
        args.corpus_root, BYTES_PER_BOOK, TRAIN_BOOKS + DEV_BOOKS + TEST_BOOKS, SEED
    )
    hashes = verify_r16_prefix(books)
    result = evaluate_locked(books[:TRAIN_BOOKS], books[TRAIN_BOOKS + DEV_BOOKS:])
    result.update({
        "corpus": "NaNROUTER short stories batch 0004",
        "seed": SEED,
        "bytes_per_book": BYTES_PER_BOOK,
        "warmup": WARMUP,
        "source_sha256": hashes,
        "train_filenames": [p.name for p in chosen[:TRAIN_BOOKS]],
        "dev_filenames_fingerprint_only": [p.name for p in chosen[TRAIN_BOOKS:TRAIN_BOOKS + DEV_BOOKS]],
        "test_filenames": [p.name for p in chosen[TRAIN_BOOKS + DEV_BOOKS:]],
    })
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
