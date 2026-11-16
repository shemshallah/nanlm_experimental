"""R15: train-only convex multi-channel readout for genuine M3 waves.

Independent candidate, not a restoration of the historical M3 R12 decoder.
Every channel is read before its target byte is observed; all readout weights
are fitted on TRAIN books and then frozen for book-disjoint DEV evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from m3_r13.mirror_wave import M3MirrorWave, WaveConfig
from m3_r14.predictive_probe import ExactWindowMemory, train_unigram_prior, metrics


STRIDE = 4  # one wave-time integration tick per four observed byte transitions
L2_PENALTY = 0.2
ARM_CONFIGS = {
    "uncoupled": WaveConfig(dt=.25, pair_speed=0, head_speed=0, damping=.004,
                            bridge=0, onsite_frequency=.2),
    "spatial_wave": WaveConfig(dt=.25, pair_speed=1, head_speed=1, damping=.004,
                               bridge=0, onsite_frequency=.2),
    "reciprocal_wave": WaveConfig(dt=.25, pair_speed=1, head_speed=1, damping=.004,
                                  bridge=.2, onsite_frequency=.2),
}
FEATURE_NAMES = (
    "directional_displacement", "directional_momentum",
    "forward_wave_energy", "forward_minus_reverse_energy",
    "reciprocal_head_state",
)


def _scale_and_center(raw: np.ndarray) -> np.ndarray:
    # Per-position target-free feature scale, common to all candidate bytes.
    centered = raw - raw.mean(axis=-1, keepdims=True)
    rms = np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True))
    return np.clip(centered / np.maximum(rms, 0.1), -8, 8)


def wave_features(w: M3MirrorWave, previous: int) -> np.ndarray:
    """Return five phase/energy/head channels x 256 possible next bytes."""
    up = w.pair_u[0, previous]
    um = w.pair_u[1, previous]
    vp = w.pair_v[0, previous]
    vm = w.pair_v[1, previous]
    fwd_u, rev_u = up + um, up - um
    fwd_v, rev_v = vp + vm, vp - vm
    omega = max(w.config.onsite_frequency, 0.2)
    power_fwd = fwd_u * fwd_u + (fwd_v / omega) ** 2
    power_rev = rev_u * rev_u + (rev_v / omega) ** 2
    head = w.head_u[0] + w.head_v[0] * w.config.dt
    return _scale_and_center(np.stack((
        fwd_u, fwd_v, power_fwd, power_fwd - power_rev, head,
    ))).T


def collect(books: list[bytes], prior: np.ndarray, warmup: int) -> dict:
    logps, targets, nonspace = [], [], []
    features = {name: [] for name in ARM_CONFIGS}
    char_count = 0
    t0 = time.perf_counter()

    for book in books:
        memory = ExactWindowMemory(prior)
        waves = {name: M3MirrorWave(config) for name, config in ARM_CONFIGS.items()}
        scoring_start = warmup
        while scoring_start < len(book) and book[scoring_start] & 0xC0 == 0x80:
            scoring_start += 1
        for pos in range(1, len(book)):
            a, b = book[pos - 1], book[pos]
            if pos >= scoring_start:
                logps.append(np.log(memory.predict(a)))
                targets.append(b)
                nonspace.append(b != 32)
                for name, wave in waves.items():
                    features[name].append(wave_features(wave, a))
            memory.observe(a, b)
            for wave in waves.values():
                wave.inject_transition(a, b, impulse=.2)
                if pos % STRIDE == 0:
                    wave.step()
        char_count += len(book[scoring_start:].decode("utf-8", "strict"))

    return {
        "base_logp": np.asarray(logps, dtype=np.float64),
        "targets": np.asarray(targets, dtype=np.int64),
        "nonspace": np.asarray(nonspace, dtype=bool),
        "unicode_chars": char_count,
        "features": {name: np.asarray(vals, dtype=np.float32) for name, vals in features.items()},
        "elapsed_s": time.perf_counter() - t0,
    }


def normalized_logits(base_logp: np.ndarray, features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    logits = base_logp + np.einsum("nbk,k->nb", features, weights, optimize=True)
    maxima = logits.max(axis=1, keepdims=True)
    return logits - maxima - np.log(np.exp(logits - maxima).sum(axis=1, keepdims=True))


def objective(base: np.ndarray, features: np.ndarray, targets: np.ndarray,
              weights: np.ndarray, penalty: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Exact convex multiclass log-loss, gradient and Fisher/Hessian (5D)."""
    lp = normalized_logits(base, features, weights)
    p = np.exp(lp)
    n = len(targets)
    rows = np.arange(n)
    weighted = np.einsum("nb,nbk->nk", p, features, optimize=True)
    target_features = features[rows, targets]
    grad = (weighted - target_features).mean(axis=0) + penalty * weights
    second = np.einsum("nb,nbk,nbl->kl", p, features, features, optimize=True) / n
    hess = second - weighted.T @ weighted / n + penalty * np.eye(len(weights))
    loss = float(-lp[rows, targets].mean() + 0.5 * penalty * weights @ weights)
    return loss, grad, hess


def fit_readout(data: dict, arm: str, penalty: float = L2_PENALTY) -> dict:
    x = data["features"][arm].astype(np.float64)
    base, labels = data["base_logp"], data["targets"]
    theta = np.zeros(x.shape[-1], dtype=np.float64)
    trajectory = []
    for iteration in range(15):
        loss, grad, hess = objective(base, x, labels, theta, penalty)
        trajectory.append(float(loss))
        if np.linalg.norm(grad) < 1e-7:
            break
        delta = np.linalg.solve(hess, grad)
        rate = 1.0
        for _ in range(12):
            candidate = theta - rate * delta
            new_loss, _, _ = objective(base, x, labels, candidate, penalty)
            if new_loss <= loss:
                theta = candidate
                break
            rate *= 0.5
        else:
            break
    return {"weights": theta, "objective": trajectory, "penalty": penalty}


def evaluate(train_books: list[bytes], dev_books: list[bytes], warmup: int = 1024) -> dict:
    if min(map(len, train_books + dev_books)) <= warmup:
        raise ValueError("each document must be longer than warmup")
    prior = train_unigram_prior(train_books)
    train = collect(train_books, prior, warmup)
    trained = {arm: fit_readout(train, arm) for arm in ARM_CONFIGS}
    dev = collect(dev_books, prior, warmup)
    baseline = metrics(dev["base_logp"], dev)
    arms = {"memory_only": baseline}
    for arm, model in trained.items():
        logp = normalized_logits(dev["base_logp"], dev["features"][arm], model["weights"])
        outcome = metrics(logp, dev)
        arms[arm] = {
            **outcome,
            "delta_bpb_vs_memory": baseline["bpb"] - outcome["bpb"],
            "fitted_weights": [float(v) for v in model["weights"]],
            "train_objective": model["objective"],
        }
    return {
        "round": "M3 R15 phase-independent learned wave readout",
        "status": "unpromoted experimental probe",
        "r12_checkpoint_loaded": False,
        "book_split": "TRAIN fit, book-disjoint DEV evaluated exactly once",
        "readout": "convex 5-feature logistic residual, TRAIN-only fit",
        "feature_names": list(FEATURE_NAMES),
        "l2": L2_PENALTY,
        "wave_stride": STRIDE,
        "wave_damping": .004,
        "train_positions": len(train["targets"]),
        "dev_positions": len(dev["targets"]),
        "train_elapsed_s": train["elapsed_s"],
        "dev_elapsed_s": dev["elapsed_s"],
        "dev_positions_per_s_all_arms": len(dev["targets"]) / dev["elapsed_s"],
        "dev": arms,
    }


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--train-books", type=int, default=4)
    parser.add_argument("--dev-books", type=int, default=4)
    parser.add_argument("--bytes-per-book", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.train_books < 1 or args.dev_books < 1 or args.bytes_per_book <= args.warmup:
        parser.error("need distinct train/dev books and bytes_per_book > warmup")
    from m3_r14.predictive_probe import resolve_books
    selected, books = resolve_books(args.corpus_root, args.bytes_per_book,
                                    args.train_books + args.dev_books, args.seed)
    result = evaluate(books[:args.train_books], books[args.train_books:], warmup=args.warmup)
    result["corpus"] = "NaNROUTER short stories batch 0004"
    result["seed"] = args.seed
    result["warmup"] = args.warmup
    result["bytes_per_book"] = args.bytes_per_book
    result["train_books"] = [p.name for p in selected[:args.train_books]]
    result["dev_books"] = [p.name for p in selected[args.train_books:]]
    result["source_sha256"] = [hashlib.sha256(book).hexdigest() for book in books]
    txt = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(txt)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    cli()
