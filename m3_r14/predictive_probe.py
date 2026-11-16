"""Small, causal M3 mirror-wave predictive attribution experiment.

This is an independent R14 probe, NOT an R12 decoder/checkpoint continuation.
No target byte may enter the memory or wave before the prediction is scored.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np

from m3_r13.mirror_wave import M3MirrorWave, WaveConfig


ARM_CONFIGS = {
    "uncoupled": WaveConfig(dt=0.25, pair_speed=0, head_speed=0, damping=0.08, bridge=0, onsite_frequency=1),
    "local_wave": WaveConfig(dt=0.25, pair_speed=1, head_speed=1, damping=0.08, bridge=0, onsite_frequency=1),
    "reciprocal_wave": WaveConfig(dt=0.25, pair_speed=1, head_speed=1, damping=0.08, bridge=0.2, onsite_frequency=1),
}
GAIN_GRID = (-0.8, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 0.8)


class ExactWindowMemory:
    """Explicit 128-transition near + 2048-transition older byte counts."""

    def __init__(self, train_prior: np.ndarray):
        if train_prior.shape != (256,) or np.any(train_prior <= 0):
            raise ValueError("prior must be a positive 256-entry distribution")
        self.prior = train_prior / train_prior.sum()
        self.near: deque[tuple[int, int]] = deque()
        self.old: deque[tuple[int, int]] = deque()
        self.near_count = np.zeros((256, 256), dtype=np.int32)
        self.old_count = np.zeros_like(self.near_count)

    def predict(self, previous: int) -> np.ndarray:
        row = self.near_count[previous] + 0.2 * self.old_count[previous]
        return (row + 8.0 * self.prior) / (row.sum() + 8.0)

    def observe(self, previous: int, target: int) -> None:
        self.near.append((previous, target))
        self.near_count[previous, target] += 1
        if len(self.near) > 128:
            a, b = self.near.popleft()
            self.near_count[a, b] -= 1
            self.old.append((a, b))
            self.old_count[a, b] += 1
            if len(self.old) > 2048:
                a, b = self.old.popleft()
                self.old_count[a, b] -= 1


def wave_score(wave: M3MirrorWave, previous: int) -> np.ndarray:
    """Direction-aware, target-free 256-byte score from actual wave state."""
    forward_displacement = wave.pair_u[0, previous] + wave.pair_u[1, previous]
    forward_momentum = wave.pair_v[0, previous] + wave.pair_v[1, previous]
    source = forward_displacement + wave.config.dt * forward_momentum
    candidate_head = 0.25 * (wave.head_u[0] + wave.head_u[1])
    raw = source + candidate_head
    raw = raw.copy()
    raw[previous] += wave.head_u[0, previous]  # diagonal fixed-head transition
    raw -= raw.mean()
    # Scale only from current causal features; cap very sparse spikes.
    rms = np.sqrt(float(np.mean(raw * raw)))
    return np.clip(raw / max(rms, 0.1), -8.0, 8.0)


def train_unigram_prior(books: list[bytes]) -> np.ndarray:
    counts = np.full(256, 0.5, dtype=np.float64)
    for b in books:
        counts += np.bincount(np.frombuffer(b, dtype=np.uint8), minlength=256)
    return counts / counts.sum()


def collect_positions(
    books: list[bytes], prior: np.ndarray, warmup: int, arms: dict[str, WaveConfig]
) -> dict:
    all_logp = []
    all_targets = []
    all_nonspace = []
    all_scores = {arm: [] for arm in arms}
    bytes_evaluated = 0
    characters_evaluated = 0
    began = time.perf_counter()

    for book in books:
        memory = ExactWindowMemory(prior)
        waves = {arm: M3MirrorWave(config) for arm, config in arms.items()}
        # Score complete Unicode characters in BPC. The optional warmup cut
        # must not begin with an interior UTF-8 continuation byte.
        scoring_start = warmup
        while scoring_start < len(book) and book[scoring_start] & 0xC0 == 0x80:
            scoring_start += 1
        for pos in range(1, len(book)):
            previous, target = book[pos - 1], book[pos]
            if pos >= scoring_start:
                p = memory.predict(previous)
                all_logp.append(np.log(p))
                all_targets.append(target)
                all_nonspace.append(target != 0x20)
                for arm, w in waves.items():
                    all_scores[arm].append(wave_score(w, previous))
            # Causal update: target cannot affect its own score.
            memory.observe(previous, target)
            for wave in waves.values():
                wave.advance_observation(previous, target, impulse=0.2)
        scored = book[scoring_start:]
        bytes_evaluated += len(scored)
        characters_evaluated += len(scored.decode("utf-8", errors="strict"))

    return {
        "base_logp": np.asarray(all_logp, dtype=np.float64),
        "targets": np.asarray(all_targets, dtype=np.int64),
        "nonspace": np.asarray(all_nonspace, dtype=bool),
        "scores": {k: np.asarray(v, dtype=np.float64) for k, v in all_scores.items()},
        "bytes": bytes_evaluated,
        "unicode_chars": characters_evaluated,
        "elapsed_s": time.perf_counter() - began,
    }


def apply_gain(base_logp: np.ndarray, scores: np.ndarray | None, gain: float) -> np.ndarray:
    logits = base_logp if scores is None else base_logp + gain * scores
    maximum = logits.max(axis=1, keepdims=True)
    log_norm = maximum + np.log(np.exp(logits - maximum).sum(axis=1, keepdims=True))
    return logits - log_norm


def metrics(logp: np.ndarray, data: dict) -> dict:
    targets = data["targets"]
    rows = np.arange(len(targets))
    loss_bits = -logp[rows, targets] / np.log(2)
    top1 = np.argmax(logp, axis=1)
    top5 = np.argpartition(logp, -5, axis=1)[:, -5:]
    nonspace = data["nonspace"]
    return {
        "positions": len(targets),
        "bpb": float(loss_bits.mean()),
        "bpc": float(loss_bits.sum() / data["unicode_chars"]),
        "nonspace_bpb": float(loss_bits[nonspace].mean()) if nonspace.any() else None,
        "top1": float(np.mean(top1 == targets)),
        "top5": float(np.mean(np.any(top5 == targets[:, None], axis=1))),
    }


def select_gain(train: dict, arm: str) -> tuple[float, list[dict]]:
    trials = []
    for gain in GAIN_GRID:
        observed = metrics(apply_gain(train["base_logp"], train["scores"][arm], gain), train)
        trials.append({"gain": gain, "train_bpb": observed["bpb"]})
    # Selection uses TRAIN positions only; never DEV or TEST labels.
    best = min(trials, key=lambda item: (item["train_bpb"], abs(item["gain"])))
    return float(best["gain"]), trials


def evaluate_books(train_books: list[bytes], dev_books: list[bytes], warmup: int = 256) -> dict:
    if min(map(len, train_books + dev_books)) <= warmup:
        raise ValueError("books must exceed warmup length")
    prior = train_unigram_prior(train_books)
    train = collect_positions(train_books, prior, warmup, ARM_CONFIGS)
    frozen_gains = {arm: select_gain(train, arm)[0] for arm in ARM_CONFIGS}
    dev = collect_positions(dev_books, prior, warmup, ARM_CONFIGS)

    memory = metrics(dev["base_logp"], dev)
    outcomes = {"memory_only": {**memory, "gain": 0.0}}
    for arm, gain in frozen_gains.items():
        outcome = metrics(apply_gain(dev["base_logp"], dev["scores"][arm], gain), dev)
        outcomes[arm] = {**outcome, "gain": gain, "delta_bpb_vs_memory": memory["bpb"] - outcome["bpb"]}

    return {
        "architecture": "M3 R14 causal wave predictive attribution probe",
        "not_comparable_to_r12_checkpoint": True,
        "split": "disjoint books; gains optimized on TRAIN only",
        "prior": "TRAIN-derived byte unigram",
        "memory": "128 near + 2048 older transitions; additive train prior",
        "wave_impulse": 0.2,
        "gain_grid": list(GAIN_GRID),
        "frozen_gain": frozen_gains,
        "train_positions": int(len(train["targets"])),
        "dev_positions": int(len(dev["targets"])),
        "train_elapsed_s": train["elapsed_s"],
        "dev_elapsed_s": dev["elapsed_s"],
        "dev_positions_per_s_including_all_arms": len(dev["targets"]) / dev["elapsed_s"],
        "dev": outcomes,
        "promotion": "no; small pilot and missing R12 checkpoint",
    }


def resolve_books(root: Path, bytes_per_book: int, count: int, seed: int) -> tuple[list[Path], list[bytes]]:
    files = sorted(p for p in root.rglob("*.txt") if p.is_file() and p.stat().st_size >= bytes_per_book)
    random.Random(seed).shuffle(files)
    if len(files) < count:
        raise ValueError(f"need {count} sufficiently long .txt books, found {len(files)}")
    chosen = files[:count]
    payloads = []
    for p in chosen:
        data = p.read_bytes()[:bytes_per_book]
        # Drop at most the incomplete terminal UTF-8 character, preserving
        # exact UTF-8 decoding and meaningful Unicode-character accounting.
        while data:
            try:
                data.decode("utf-8", errors="strict")
                break
            except UnicodeDecodeError as exc:
                if exc.start < len(data) - 4:
                    raise ValueError(f"invalid UTF-8 inside book {p.name}") from exc
                data = data[:exc.start]
        payloads.append(data)
    return chosen, payloads


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--train-books", type=int, default=4)
    parser.add_argument("--dev-books", type=int, default=4)
    parser.add_argument("--bytes-per-book", type=int, default=768)
    parser.add_argument("--warmup", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.bytes_per_book <= args.warmup or args.train_books < 1 or args.dev_books < 1:
        parser.error("each book must exceed warmup and each split must have a book")
    chosen, books = resolve_books(
        args.corpus_root, args.bytes_per_book, args.train_books + args.dev_books, args.seed
    )
    result = evaluate_books(books[:args.train_books], books[args.train_books:], args.warmup)
    result["source"] = "NaNROUTER public-domain short-story books when run on their release"
    result["seed"] = args.seed
    result["bytes_per_book"] = args.bytes_per_book
    result["warmup"] = args.warmup
    result["train_books"] = [p.name for p in chosen[:args.train_books]]
    result["dev_books"] = [p.name for p in chosen[args.train_books:]]
    result["sha256"] = [hashlib.sha256(b).hexdigest() for b in books]
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    cli()
