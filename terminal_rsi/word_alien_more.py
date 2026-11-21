#!/usr/bin/env python3
"""Standalone alien word predictors for the Terminal-RSI word track.

This module collects the remaining "alien computation" word priors in one
dependency-light place.  It is deliberately self-contained: only the Python
standard library and NumPy are required.  Nothing here imports the training
stack, so the classes can be exercised from a notebook or wired into a larger
word-candidate union later.

Classes
-------
``PersistentHomologyFeatures``  H0 Vietoris-Rips merge-tree features (W5.9/W6.7)
``SpikingTraceMemory``          sparse codes, traces, STDP transitions (W5.4)
``EchoStateReservoir``          two-timescale fixed reservoir + ridge (W5.2)
``KuramotoPrior``               word oscillators and phase readout (W5.5)
``CellularAutomatonWordPrior``  local CA over word embeddings (W5.6)
``AntColonyPrior``              diffusion/ant flow on a word graph (W5.7)
``TensorTrainWordModel``        small TT/MPS joint word model (W6.6)
``RenormalizationBlockModel``   Re-Pair phrase blocks + block prediction (W6.4)
``FreeEnergyRegimeModel``       K-regime HMM trained by EM (W6.5)

Shared API
----------
* ``predict(...)`` methods return ``(ids, scores)`` as NumPy arrays, sorted
  by descending score.  ``ids`` are always in-range word IDs.
* ``fit(...)`` methods return a small dict of training statistics; fitted
  models also expose ``train_loss``/``train_top5`` where meaningful.
* Streaming models (STDP, Kuramoto) expose ``observe(word)`` and ``reset()``.
* Context-conditioned models replay ``context_ids`` from a fresh state, so a
  prediction is causal by construction: it never sees tokens after ``context``.

Compute and limits (CPU only; no GPU work is started)
------------------------------------------------------
* PH: O(L^2) distances and L Prim steps for a window of L words.
* STDP: observe is O(delays * neurons^2), predict is O(neurons^2 + V*neurons).
* Reservoir: fit is O(T * units^2) with units = n_fast + n_slow.  Keep
  ``max_fit_tokens`` at or below ~20k for interactive use.
* Kuramoto: O(T * V) to advance/couple phases plus a small linear readout.
* CA: O(T * cells * dim^2) for the local rule; linear in stream length.
* Ant colony: O(edges + steps * edges) per prediction.
* TT/MPS: builds a dense V**order count tensor.  This is a *prototype* for
  small vocabularies; the default guard allows about 2M tensor entries.  For
  4k-word vocabularies cluster IDs first or use a different estimator.
* Re-Pair: O(rules * T log T) counting plus a Python replacement pass.  It is
  comfortable for streams up to ~100k words, not for full corpora.
* Free energy: O(epochs * T * K^2), K=32-64.  The default fit cap is 20k
  tokens; full-corpus EM belongs in the offline runner, not here.

Quick use::

    from terminal_rsi.word_alien_more import EchoStateReservoir

    model = EchoStateReservoir(vocab_size=4096)
    stats = model.fit(train_ids)
    ids, scores = model.predict(dev_ids[-32:], top_k=5)

Tests::

    python -m unittest terminal_rsi.test_word_alien_more -v
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np

DEFAULT_SEED = 2026
ArrayLike = Sequence[int] | np.ndarray

__all__ = [
    "PersistentHomologyFeatures",
    "SpikingTraceMemory",
    "EchoStateReservoir",
    "KuramotoPrior",
    "CellularAutomatonWordPrior",
    "AntColonyPrior",
    "TensorTrainWordModel",
    "RenormalizationBlockModel",
    "FreeEnergyRegimeModel",
    "make_word_embeddings",
    "top_k_ids",
]


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def _validate_ids(values: ArrayLike, vocab_size: int, *,
                  name: str = "ids") -> np.ndarray:
    """Return a flat int64 array, rejecting negative or out-of-range IDs."""
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    if arr.size:
        low = int(arr.min())
        high = int(arr.max())
        if low < 0 or high >= int(vocab_size):
            raise ValueError(
                f"{name} must be in [0, {int(vocab_size) - 1}], "
                f"got [{low}, {high}]")
    return arr


def _single_id(value: int, vocab_size: int, *, name: str = "word") -> int:
    word = int(value)
    if word < 0 or word >= int(vocab_size):
        raise ValueError(f"{name} must be in [0, {int(vocab_size) - 1}], "
                         f"got {word}")
    return word


def top_k_ids(scores: np.ndarray, k: int, *,
              exclude: Iterable[int] = ()) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic top-k: score descending, then lower ID on ties."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    values = np.nan_to_num(values, nan=-np.inf,
                           posinf=np.finfo(np.float64).max,
                           neginf=-np.finfo(np.float64).max)
    if exclude:
        values = values.copy()
        for index in exclude:
            if 0 <= int(index) < values.size:
                values[int(index)] = -np.inf
    k = max(1, min(int(k), values.size))
    order = np.lexsort((np.arange(values.size), -values))[:k]
    return order.astype(np.int64), values[order]


def make_word_embeddings(vocab_size: int, dim: int = 32,
                         seed: int = DEFAULT_SEED) -> np.ndarray:
    """Fixed, unit-norm random word embeddings (float32)."""
    vocab_size = int(vocab_size)
    dim = int(dim)
    if vocab_size <= 0 or dim <= 0:
        raise ValueError("vocab_size and dim must be positive")
    rng = np.random.default_rng(int(seed))
    table = rng.normal(0.0, 1.0 / math.sqrt(dim),
                       (vocab_size, dim)).astype(np.float32)
    norms = np.linalg.norm(table, axis=1, keepdims=True) + 1e-8
    return (table / norms).astype(np.float32)


def _ridge_readout(features: np.ndarray, targets: np.ndarray,
                   ridge: float) -> np.ndarray:
    """Solve (X^T X + ridge I) W = X^T Y in float64."""
    gram = np.asarray(features, dtype=np.float64)
    rhs = np.asarray(targets, dtype=np.float64)
    size = gram.shape[0]
    gram = gram.copy()
    gram.flat[::size + 1] += float(max(ridge, 0.0))
    try:
        weight = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        weight = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    return weight.astype(np.float32)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values)
    return shifted - math.log(float(np.exp(shifted).sum()))


def _nll_from_logits(logits: np.ndarray, target: int) -> float:
    logp = _log_softmax(logits)
    return float(-logp[int(target)])


def _sparse_codes(vocab_size: int, neurons: int, active: int,
                  seed: int) -> np.ndarray:
    """Exactly ``active`` active neurons per word, unit L2 norm per row."""
    active = int(active)
    if active <= 0 or active > neurons:
        raise ValueError("active must be in [1, n_neurons]")
    rng = np.random.default_rng(int(seed))
    codes = np.zeros((int(vocab_size), int(neurons)), dtype=np.float32)
    scores = rng.random((int(vocab_size), int(neurons)))
    chosen = np.argpartition(scores, active - 1, axis=1)[:, :active]
    np.put_along_axis(codes, chosen, np.float32(1.0 / math.sqrt(active)),
                      axis=1)
    return codes


# ---------------------------------------------------------------------------
# 1. H0 persistence of the recent word-embedding cloud
# ---------------------------------------------------------------------------


class PersistentHomologyFeatures:
    """Fixed-width H0 Vietoris-Rips features for the last ``window`` words.

    In dimension 0 the Vietoris-Rips persistence diagram is the merge tree of
    single-linkage clustering.  We compute its merge times with a vectorised
    Prim minimum spanning tree over the pairwise distance matrix.

    Feature layout (width ``11 + len(thresholds)``, independent of window)::

        0 edge mean          1 edge max           2 edge std
        3 edge q25           4 edge q50           5 edge q75
        6 total persistence  7 normalized entropy
        8 .. 8+T-1           component fraction at each threshold
        8+T                  largest merge-time gap / max merge
        9+T                  n_points / window
        10+T                 fraction of merge times <= threshold[0]

    H0 bars are born at 0 and die at their single-linkage merge time, so the
    "persistence" statistics are the merge-time statistics.  Component counts
    are returned as fractions of the window size to keep the output scale-free.

    Parameters
    ----------
    vocab_size:
        Number of word IDs.
    dim:
        Embedding width used when ``embeddings`` is None.
    window:
        Number of most recent words to include (L).
    thresholds:
        Distance thresholds for component counts.
    metric:
        ``"cosine"`` (default) or ``"euclidean"``.
    embeddings:
        Optional ``[vocab_size, dim]`` table.  It is normalised for cosine.
    """

    def __init__(self, vocab_size: int, dim: int = 32, window: int = 16,
                 thresholds: Sequence[float] = (0.1, 0.25, 0.5, 0.75),
                 metric: str = "cosine", seed: int = DEFAULT_SEED,
                 embeddings: np.ndarray | None = None,
                 L: int | None = None):
        # ``L`` is an alias for ``window`` to match the persistence notation.
        if L is not None:
            window = int(L)
        self.vocab_size = int(vocab_size)
        self.window = int(window)
        if self.window < 1:
            raise ValueError("window must be positive")
        if metric not in ("cosine", "euclidean"):
            raise ValueError("metric must be 'cosine' or 'euclidean'")
        self.metric = metric
        self.thresholds = np.asarray(sorted(float(t) for t in thresholds),
                                     dtype=np.float64)
        if self.thresholds.size == 0:
            raise ValueError("at least one threshold is required")
        if (self.thresholds < 0).any():
            raise ValueError("thresholds must be non-negative")

        if embeddings is None:
            self.dim = int(dim)
            table = make_word_embeddings(self.vocab_size, self.dim, seed)
        else:
            table = np.asarray(embeddings, dtype=np.float32)
            if table.ndim != 2 or table.shape[0] != self.vocab_size:
                raise ValueError("embeddings must be [vocab_size, dim]")
            self.dim = int(table.shape[1])
        if metric == "cosine":
            table = table / (np.linalg.norm(table, axis=1, keepdims=True)
                             + 1e-8)
        self.embeddings = table.astype(np.float32)
        self.feature_dim = 11 + int(self.thresholds.size)

    # -- geometry ---------------------------------------------------------

    def _points(self, context_ids: ArrayLike) -> np.ndarray:
        ids = _validate_ids(context_ids, self.vocab_size,
                            name="context_ids")[-self.window:]
        return self.embeddings[ids]

    def _distances(self, points: np.ndarray) -> np.ndarray:
        if self.metric == "cosine":
            similarity = points @ points.T
            dist = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * similarity))
        else:
            diff = points[:, None, :] - points[None, :, :]
            dist = np.sqrt(np.maximum((diff * diff).sum(axis=-1), 0.0))
        np.fill_diagonal(dist, np.inf)
        return dist.astype(np.float64)

    def merge_times(self, context_ids: ArrayLike) -> np.ndarray:
        """Sorted H0 death times from a Prim MST (empty for <2 points)."""
        points = self._points(context_ids)
        n_points = len(points)
        if n_points <= 1:
            return np.zeros(0, dtype=np.float64)
        dist = self._distances(points)
        in_tree = np.zeros(n_points, dtype=bool)
        in_tree[0] = True
        key = dist[0].copy()
        key[0] = np.inf
        merges = np.empty(n_points - 1, dtype=np.float64)
        for step in range(n_points - 1):
            chosen = int(np.argmin(key))
            merges[step] = key[chosen]
            in_tree[chosen] = True
            key = np.minimum(key, dist[chosen])
            key[in_tree] = np.inf
        return np.sort(merges)

    def component_counts(self, context_ids: ArrayLike) -> np.ndarray:
        """Number of connected components at each configured threshold."""
        ids = _validate_ids(context_ids, self.vocab_size,
                            name="context_ids")[-self.window:]
        n_points = len(ids)
        if n_points == 0:
            return np.zeros(self.thresholds.size, dtype=np.int64)
        merges = self.merge_times(ids)
        counts = 1 + (merges[None, :] > self.thresholds[:, None]).sum(axis=1)
        return counts.astype(np.int64)

    # -- features ---------------------------------------------------------

    def features(self, context_ids: ArrayLike) -> np.ndarray:
        """One fixed-width float32 feature vector for a context."""
        ids = _validate_ids(context_ids, self.vocab_size,
                            name="context_ids")[-self.window:]
        out = np.zeros(self.feature_dim, dtype=np.float32)
        n_points = len(ids)
        if n_points == 0:
            return out
        merges = self.merge_times(ids)
        n_bars = merges.size
        if n_bars:
            out[0] = float(merges.mean())
            out[1] = float(merges.max())
            out[2] = float(merges.std())
            out[3], out[4], out[5] = np.quantile(
                merges, [0.25, 0.5, 0.75]).astype(np.float32)
            total = float(merges.sum())
            out[6] = total
            if total > 0.0 and n_bars > 1:
                prob = merges / total
                entropy = -float((prob * np.log(prob + 1e-12)).sum())
                out[7] = entropy / math.log(n_bars)
            max_merge = float(out[1]) + 1e-8
            gaps = np.diff(np.concatenate(([0.0], merges)))
            out[8 + self.thresholds.size] = float(gaps.max() / max_merge)
        counts = 1 + (merges[None, :] > self.thresholds[:, None]).sum(axis=1)
        start = 8
        out[start:start + self.thresholds.size] = (
            counts.astype(np.float32) / float(n_points))
        out[9 + self.thresholds.size] = float(n_points / self.window)
        if n_bars:
            out[10 + self.thresholds.size] = float(
                (merges <= self.thresholds[0]).mean())
        return out

    def batch_features(self, contexts: Iterable[ArrayLike]) -> np.ndarray:
        rows = [self.features(context) for context in contexts]
        if not rows:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        return np.stack(rows, axis=0)

    def transform(self, contexts) -> np.ndarray:
        """Features for one stream or a batch of streams.

        A 1-D sequence of IDs is treated as a single context and returns a
        ``[1, feature_dim]`` matrix, which mirrors sklearn-style transformers.
        """
        if isinstance(contexts, np.ndarray):
            if contexts.ndim == 1:
                return self.features(contexts)[None, :]
            return self.batch_features(contexts)
        items = list(contexts)
        if not items:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        if isinstance(items[0], (int, np.integer)):
            return self.features(items)[None, :]
        return self.batch_features(items)


# ---------------------------------------------------------------------------
# 2. Sparse spiking trace memory with STDP-style delayed transitions
# ---------------------------------------------------------------------------


class SpikingTraceMemory:
    """Sparse word codes, exponential traces and delayed Hebbian transitions.

    Each word owns a fixed sparse code (``active`` neurons, unit norm).  A
    bank of exponential traces keeps the recent code activity at ``delays``
    lags.  Observing a word performs an STDP-style outer-product update
    ``W[d] += lr * trace[d] (x) code[word]`` for every delay.  Prediction is
    the vectorised drive ``sum_d W[d].T trace[d]`` decoded by the word codes.

    No Python loop over neurons is used: the only loop is over delays (usually
    3-5), and each update is one NumPy outer product.

    Parameters
    ----------
    vocab_size:
        Number of word IDs.
    n_neurons:
        Size of the sparse coding population.
    active:
        Active neurons per word.  Smaller values reduce cross-talk.
    delays:
        Number of trace/transition lags.
    decay:
        Per-step trace decay.
    learning_rate:
        STDP increment.
    weight_decay:
        Optional multiplicative forgetting per observation (1.0 = none).
    unigram_weight:
        Optional log-count prior mixed into the decoded score.  Zero keeps
        the predictor purely transition driven.
    """

    def __init__(self, vocab_size: int, n_neurons: int = 192, active: int = 6,
                 delays: int = 4, decay: float = 0.55,
                 learning_rate: float = 0.2, weight_decay: float = 1.0,
                 unigram_weight: float = 0.0, seed: int = DEFAULT_SEED):
        self.vocab_size = int(vocab_size)
        self.n_neurons = int(n_neurons)
        self.active = int(active)
        self.delays = int(delays)
        if self.delays < 1:
            raise ValueError("delays must be positive")
        self.decay = float(decay)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.unigram_weight = float(unigram_weight)
        self.codes = _sparse_codes(self.vocab_size, self.n_neurons,
                                   self.active, seed)
        self.weights = np.zeros(
            (self.delays, self.n_neurons, self.n_neurons), dtype=np.float32)
        self.traces = np.zeros((self.delays, self.n_neurons),
                               dtype=np.float32)
        self.counts = np.zeros(self.vocab_size, dtype=np.float64)
        self.steps = 0
        self.last_word = -1

    def reset(self, clear_memory: bool = False) -> None:
        """Clear traces.  ``clear_memory=True`` also clears weights/counts."""
        self.traces.fill(0.0)
        self.steps = 0
        self.last_word = -1
        if clear_memory:
            self.weights.fill(0.0)
            self.counts.fill(0.0)

    def observe(self, word: int, *, learn: bool = True) -> None:
        """Present one word; apply delayed STDP updates unless ``learn=False``.

        ``learn=False`` is useful for causal evaluation: traces still advance,
        but the transition weights are held fixed.
        """
        word = _single_id(word, self.vocab_size)
        post = self.codes[word]
        if learn:
            if self.weight_decay != 1.0:
                self.weights *= np.float32(self.weight_decay)
            for delay in range(self.delays):
                self.weights[delay] += (
                    np.float32(self.learning_rate)
                    * np.outer(self.traces[delay], post))
            self.counts[word] += 1.0
        new_trace = np.float32(self.decay) * self.traces[0] + post
        self.traces[1:] = self.traces[:-1].copy()
        self.traces[0] = new_trace
        self.last_word = word
        self.steps += 1

    def _scores(self) -> np.ndarray:
        drive = np.zeros(self.n_neurons, dtype=np.float32)
        for delay in range(self.delays):
            drive += self.weights[delay].T @ self.traces[delay]
        scores = self.codes @ drive
        if self.unigram_weight:
            scores = scores + self.unigram_weight * np.log1p(self.counts)
        return scores

    def predict(self, top_k: int = 5, *,
                exclude_last: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Top-k word IDs and scores from the current trace state."""
        scores = self._scores()
        exclude = (self.last_word,) if exclude_last and self.last_word >= 0 else ()
        return top_k_ids(scores, top_k, exclude=exclude)

    def topk_accuracy(self, ids: ArrayLike, top_k: int = 5, *,
                      reset: bool = True,
                      clear_memory: bool = False) -> float:
        """Causal next-word top-k accuracy after replaying ``ids``.

        The model advances its traces with ``observe(ids[t], learn=False)``
        and is scored against ``ids[t+1]``; weights are not updated by this
        metric.  ``reset`` clears traces first.  Set ``clear_memory=True`` for
        a true cold-start baseline.
        """
        if reset:
            self.reset(clear_memory=clear_memory)
        sequence = _validate_ids(ids, self.vocab_size)
        hits = 0
        total = 0
        for index in range(len(sequence) - 1):
            self.observe(sequence[index], learn=False)
            predicted, _ = self.predict(top_k)
            hits += int(np.any(predicted == sequence[index + 1]))
            total += 1
        return hits / max(total, 1)


# ---------------------------------------------------------------------------
# 3. Two-timescale echo-state reservoir with a ridge readout
# ---------------------------------------------------------------------------


class EchoStateReservoir:
    """Fixed random reservoir with fast and slow leak rates.

    The state is a concatenation of a fast bank and a slow bank.  A single
    random recurrent matrix couples both banks, but each unit uses its own
    leak, giving two effective timescales.  Only the linear readout is trained,
    by ridge regression on next-word one-hot targets.

    Fit cost is O(T * units^2); prediction cost is O(context * units^2 + V *
    units).  The implementation accumulates ``X^T X`` and ``X^T Y`` online, so
    it does not store the full state matrix.

    Parameters
    ----------
    n_fast, n_slow:
        Units in each timescale bank.
    input_dim:
        Width of the fixed random word embedding.
    spectral_radius:
        Rescaled spectral radius of the recurrent matrix (<1 for stability).
    ridge:
        Ridge penalty for the readout.
    max_fit_tokens:
        Streams longer than this are fit on their most recent contiguous slice.
    """

    def __init__(self, vocab_size: int, n_fast: int = 64, n_slow: int = 32,
                 input_dim: int = 16, spectral_radius: float = 0.9,
                 leak_fast: float = 0.35, leak_slow: float = 0.08,
                 input_scale: float = 0.6, ridge: float = 1e-2,
                 seed: int = DEFAULT_SEED, max_fit_tokens: int = 20000,
                 context_limit: int = 128):
        self.vocab_size = int(vocab_size)
        self.n_fast = int(n_fast)
        self.n_slow = int(n_slow)
        self.n_units = self.n_fast + self.n_slow
        self.input_dim = int(input_dim)
        self.ridge = float(ridge)
        self.max_fit_tokens = int(max_fit_tokens)
        self.context_limit = int(context_limit)
        if self.n_fast < 1 or self.n_slow < 1:
            raise ValueError("n_fast and n_slow must be positive")
        rng = np.random.default_rng(int(seed))
        self.embeddings = make_word_embeddings(
            self.vocab_size, self.input_dim, seed + 1)
        self.input_weights = rng.normal(
            0.0, input_scale / math.sqrt(self.input_dim),
            (self.n_units, self.input_dim)).astype(np.float32)
        recurrent = rng.normal(
            0.0, 1.0 / math.sqrt(self.n_units),
            (self.n_units, self.n_units))
        recurrent[:self.n_fast, self.n_fast:] *= 0.25
        recurrent[self.n_fast:, :self.n_fast] *= 0.25
        radius = float(np.max(np.abs(np.linalg.eigvals(recurrent))))
        if radius > 0:
            recurrent *= float(spectral_radius) / radius
        self.recurrent = recurrent.astype(np.float32)
        self.bias = rng.normal(0.0, 0.01, self.n_units).astype(np.float32)
        self.leak = np.concatenate([
            np.full(self.n_fast, float(leak_fast), dtype=np.float32),
            np.full(self.n_slow, float(leak_slow), dtype=np.float32),
        ])
        self.readout: np.ndarray | None = None
        self.train_loss: float | None = None
        self.train_top5: float | None = None

    def _step(self, state: np.ndarray, word: int) -> np.ndarray:
        pre = (self.input_weights @ self.embeddings[word]
               + self.recurrent @ state + self.bias)
        return ((1.0 - self.leak) * state
                + self.leak * np.tanh(pre)).astype(np.float32)

    def run(self, context_ids: ArrayLike) -> np.ndarray:
        """Advance the reservoir from zero through ``context_ids``.

        Only the most recent ``context_limit`` words are replayed; longer
        contexts are truncated because the reservoir state already carries the
        older history.
        """
        sequence = _validate_ids(context_ids, self.vocab_size,
                                 name="context_ids")
        if self.context_limit > 0 and len(sequence) > self.context_limit:
            sequence = sequence[-self.context_limit:]
        state = np.zeros(self.n_units, dtype=np.float32)
        for word in sequence:
            state = self._step(state, int(word))
        return state.copy()

    state = run

    def _features(self, state: np.ndarray) -> np.ndarray:
        features = np.empty(self.n_units + 1, dtype=np.float32)
        features[:self.n_units] = state
        features[-1] = 1.0
        return features

    def fit(self, ids: ArrayLike, *, ridge: float | None = None) -> dict:
        """Train the ridge readout on a word-ID stream.

        The transition ``ids[t] -> ids[t+1]`` is used after the reservoir has
        consumed ``ids[t]``.  The target is therefore always one step in the
        future; no target leaks into its own feature row.
        """
        sequence = _validate_ids(ids, self.vocab_size)
        if len(sequence) < 2:
            raise ValueError("fit needs at least two word IDs")
        if len(sequence) > self.max_fit_tokens:
            sequence = sequence[-self.max_fit_tokens:]
        feature_dim = self.n_units + 1
        gram = np.zeros((feature_dim, feature_dim), dtype=np.float64)
        rhs = np.zeros((feature_dim, self.vocab_size), dtype=np.float64)
        state = np.zeros(self.n_units, dtype=np.float32)
        for index in range(len(sequence) - 1):
            state = self._step(state, int(sequence[index]))
            features = self._features(state)
            gram += np.outer(features, features)
            rhs[:, int(sequence[index + 1])] += features
        penalty = self.ridge if ridge is None else float(ridge)
        self.readout = _ridge_readout(gram, rhs, penalty)
        self.train_loss = self.loss(sequence)
        self.train_top5 = self.topk_accuracy(sequence, 5)
        return {"tokens": int(len(sequence)),
                "train_loss": float(self.train_loss),
                "train_top5": float(self.train_top5),
                "ridge": float(penalty)}

    def _logits(self, state: np.ndarray) -> np.ndarray:
        if self.readout is None:
            raise RuntimeError("fit() must be called before prediction")
        return (self.readout[:-1].T @ state + self.readout[-1]).astype(
            np.float64)

    def predict(self, context_ids: ArrayLike, top_k: int = 5
                ) -> tuple[np.ndarray, np.ndarray]:
        state = self.run(context_ids)
        return top_k_ids(self._logits(state), top_k)

    def loss(self, ids: ArrayLike) -> float:
        """Mean next-word negative log-likelihood."""
        sequence = _validate_ids(ids, self.vocab_size)
        if len(sequence) < 2:
            return float("nan")
        if len(sequence) > self.max_fit_tokens:
            sequence = sequence[-self.max_fit_tokens:]
        state = np.zeros(self.n_units, dtype=np.float32)
        total = 0.0
        for index in range(len(sequence) - 1):
            state = self._step(state, int(sequence[index]))
            total += _nll_from_logits(self._logits(state),
                                      int(sequence[index + 1]))
        return total / (len(sequence) - 1)

    def topk_accuracy(self, ids: ArrayLike, top_k: int = 5) -> float:
        sequence = _validate_ids(ids, self.vocab_size)
        if len(sequence) < 2:
            return float("nan")
        if len(sequence) > self.max_fit_tokens:
            sequence = sequence[-self.max_fit_tokens:]
        state = np.zeros(self.n_units, dtype=np.float32)
        hits = 0
        for index in range(len(sequence) - 1):
            state = self._step(state, int(sequence[index]))
            predicted, _ = top_k_ids(self._logits(state), top_k)
            hits += int(np.any(predicted == sequence[index + 1]))
        return hits / (len(sequence) - 1)


# ---------------------------------------------------------------------------
# 4. Kuramoto word oscillators
# ---------------------------------------------------------------------------


class KuramotoPrior:
    """Word phasors with phase advance, mean-field coupling and a linear readout.

    Every word owns a natural frequency ``omega`` and an initial phase.  At
    each observation all phases advance by ``omega * dt``; the observed word
    is then pulled toward the global Kuramoto mean phase and toward the
    decayed context field.  Features are the real/imaginary parts of the last
    ``window`` word phasors plus the current field.  A ridge readout maps those
    phase features to next-word logits, and a resonance term adds
    ``cos(angle(field) - predicted_phase_of_candidate)``.

    ``fit`` is O(T * V) for phase updates plus O(T * F^2) for the readout.
    ``predict`` is O(context * V + V * F).

    Parameters
    ----------
    coupling:
        Mean-field coupling strength in ``[0, 1]``.
    omega_scale:
        Standard deviation of natural frequencies.  Keep it small so phase
        features stay informative over a normal context.
    window:
        Number of recent phasors exposed to the readout.
    resonance_weight:
        Weight of the fixed resonance score added to the learned logits.
    """

    def __init__(self, vocab_size: int, coupling: float = 0.25,
                 dt: float = 1.0, omega_scale: float = 0.03,
                 decay: float = 0.7, window: int = 8,
                 ridge: float = 1e-2, resonance_weight: float = 0.1,
                 seed: int = DEFAULT_SEED):
        self.vocab_size = int(vocab_size)
        self.coupling = float(coupling)
        self.dt = float(dt)
        self.omega_scale = float(omega_scale)
        self.decay = float(decay)
        self.window = int(window)
        self.ridge = float(ridge)
        self.resonance_weight = float(resonance_weight)
        if self.window < 1:
            raise ValueError("window must be positive")
        rng = np.random.default_rng(int(seed))
        self.theta0 = rng.uniform(-math.pi, math.pi,
                                  self.vocab_size).astype(np.float64)
        self.omega = rng.normal(0.0, self.omega_scale,
                                self.vocab_size).astype(np.float64)
        self.feature_dim = 2 * self.window + 3
        self.readout: np.ndarray | None = None
        self.train_loss: float | None = None
        self.train_top5: float | None = None
        self.reset()

    def reset(self) -> None:
        self.time = 0.0
        self.theta = self.theta0.copy()
        self.field = 0.0 + 0.0j
        self._snapshots: list[complex] = []
        self._counts = np.zeros(self.vocab_size, dtype=np.float64)
        self.last_word = -1

    def observe(self, word: int) -> None:
        word = _single_id(word, self.vocab_size)
        self.time += self.dt
        self.theta += self.omega * self.dt
        mean_phasor = np.mean(np.exp(1j * self.theta))
        order = float(np.abs(mean_phasor))
        if self.field != 0.0 + 0.0j:
            pull = math.atan2(self.field.imag, self.field.real)
            self.theta[word] += self.coupling * math.sin(
                pull - self.theta[word])
        self.theta[word] += self.coupling * order * math.sin(
            math.atan2(mean_phasor.imag, mean_phasor.real)
            - self.theta[word])
        phasor = np.exp(1j * self.theta[word])
        self.field = self.decay * self.field + phasor
        self._snapshots.append(complex(phasor))
        if len(self._snapshots) > self.window:
            self._snapshots.pop(0)
        self._counts[word] += 1.0
        self.last_word = word

    def _features(self) -> np.ndarray:
        features = np.zeros(self.feature_dim, dtype=np.float64)
        offset = self.window - len(self._snapshots)
        for index, phasor in enumerate(self._snapshots):
            slot = offset + index
            features[2 * slot] = phasor.real
            features[2 * slot + 1] = phasor.imag
        magnitude = abs(self.field)
        features[-3] = magnitude
        if magnitude > 1e-12:
            features[-2] = self.field.real / magnitude
            features[-1] = self.field.imag / magnitude
        return features

    def _resonance(self) -> np.ndarray:
        magnitude = abs(self.field)
        if magnitude < 1e-12:
            return np.zeros(self.vocab_size, dtype=np.float64)
        predicted = np.exp(1j * (self.theta + self.omega * self.dt))
        return np.real(self.field * np.conj(predicted)) / magnitude

    def _logits(self, state_features: np.ndarray) -> np.ndarray:
        if self.readout is None:
            return self.resonance_weight * self._resonance()
        learned = state_features @ self.readout
        return learned + self.resonance_weight * self._resonance()

    def fit(self, ids: ArrayLike, *,
            ridge: float | None = None) -> dict:
        sequence = _validate_ids(ids, self.vocab_size)
        if len(sequence) < 2:
            raise ValueError("fit needs at least two word IDs")
        self.reset()
        gram = np.zeros((self.feature_dim, self.feature_dim),
                        dtype=np.float64)
        rhs = np.zeros((self.feature_dim, self.vocab_size),
                       dtype=np.float64)
        for index in range(len(sequence) - 1):
            self.observe(int(sequence[index]))
            features = self._features()
            gram += np.outer(features, features)
            rhs[:, int(sequence[index + 1])] += features
        penalty = self.ridge if ridge is None else float(ridge)
        self.readout = _ridge_readout(gram, rhs, penalty)
        self.train_loss = self.loss(sequence)
        self.train_top5 = self.topk_accuracy(sequence, 5)
        self.reset()
        return {"tokens": int(len(sequence)),
                "train_loss": float(self.train_loss),
                "train_top5": float(self.train_top5),
                "ridge": float(penalty),
                "feature_dim": int(self.feature_dim)}

    def predict(self, context_ids: ArrayLike, top_k: int = 5
                ) -> tuple[np.ndarray, np.ndarray]:
        self.reset()
        sequence = _validate_ids(context_ids, self.vocab_size,
                                 name="context_ids")
        for word in sequence:
            self.observe(int(word))
        features = self._features()
        return top_k_ids(self._logits(features), top_k)

    def loss(self, ids: ArrayLike) -> float:
        sequence = _validate_ids(ids, self.vocab_size)
        if len(sequence) < 2:
            return float("nan")
        self.reset()
        total = 0.0
        for index in range(len(sequence) - 1):
            self.observe(int(sequence[index]))
            total += _nll_from_logits(self._logits(self._features()),
                                      int(sequence[index + 1]))
        return total / (len(sequence) - 1)

    def topk_accuracy(self, ids: ArrayLike, top_k: int = 5) -> float:
        sequence = _validate_ids(ids, self.vocab_size)
        if len(sequence) < 2:
            return float("nan")
        self.reset()
        hits = 0
        for index in range(len(sequence) - 1):
            self.observe(int(sequence[index]))
            predicted, _ = top_k_ids(self._logits(self._features()), top_k)
            hits += int(np.any(predicted == sequence[index + 1]))
        return hits / (len(sequence) - 1)


# ---------------------------------------------------------------------------
# 5. Local cellular automaton over word embeddings
# ---------------------------------------------------------------------------


class CellularAutomatonWordPrior:
    """A short 1-D lattice of word embeddings updated by a fixed local rule.

    Observing a word shifts the lattice one cell and injects its embedding at
    cell 0.  Each step then applies

        state <- (1 - mix) * state + mix * tanh(W_self state + W_near shift)

    with fixed random local matrices.  The flattened lattice is a context
    feature, and a ridge readout is trained on next words.  The rule itself is
    never trained; only the readout is.

    Cost: O(context * cells * dim^2) to encode and O(T * cells * dim^2) to fit.
    """

    def __init__(self, vocab_size: int, dim: int = 12, cells: int = 6,
                 steps: int = 1, mix: float = 0.55, ridge: float = 1e-2,
                 seed: int = DEFAULT_SEED, max_fit_tokens: int = 20000):
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.cells = int(cells)
        self.steps = int(steps)
        self.mix = float(mix)
        self.ridge = float(ridge)
        self.max_fit_tokens = int(max_fit_tokens)
        if self.cells < 1 or self.dim < 1 or self.steps < 0:
            raise ValueError("cells/dim must be positive and steps >= 0")
        rng = np.random.default_rng(int(seed))
        self.embeddings = make_word_embeddings(
            self.vocab_size, self.dim, seed + 1)
        self.rule_self = rng.normal(
            0.0, 0.8 / math.sqrt(self.dim),
            (self.dim, self.dim)).astype(np.float32)
        self.rule_neighbor = rng.normal(
            0.0, 0.5 / math.sqrt(self.dim),
            (self.dim, self.dim)).astype(np.float32)
        self.rule_bias = rng.normal(0.0, 0.01, self.dim).astype(np.float32)
        self.state = np.zeros((self.cells, self.dim), dtype=np.float32)
        self.readout: np.ndarray | None = None
        self.train_loss: float | None = None
        self.train_top5: float | None = None

    @property
    def feature_dim(self) -> int:
        return self.cells * self.dim + 1

    def reset(self) -> None:
        self.state.fill(0.0)

    def _update(self) -> None:
        state = self.state
        pre = (state @ self.rule_self
               + np.roll(state, 1, axis=0) @ self.rule_neighbor
               + np.roll(state, -1, axis=0) @ self.rule_neighbor
               + self.rule_bias)
        updated = np.tanh(pre)
        self.state = ((1.0 - self.mix) * state
                      + self.mix * updated).astype(np.float32)

    def observe(self, word: int) -> None:
        word = _single_id(word, self.vocab_size)
        if self.cells > 1:
            self.state[1:] = self.state[:-1].copy()
        self.state[0] = self.embeddings[word]
        for _ in range(self.steps):
            self._update()

    def _features(self) -> np.ndarray:
        features = np.empty(self.feature_dim, dtype=np.float32)
        features[:-1] = self.state.reshape(-1)
        features[-1] = 1.0
        return features

    def fit(self, ids: ArrayLike, *, ridge: float | None = None) -> dict:
        sequence = _validate_ids(ids, self.vocab_size)
        if len(sequence) < 2:
            raise ValueError("fit needs at least two word IDs")
        if len(sequence) > self.max_fit_tokens:
            sequence = sequence[-self.max_fit_tokens:]
        self.reset()
        gram = np.zeros((self.feature_dim, self.feature_dim),
                        dtype=np.float64)
        rhs = np.zeros((self.feature_dim, self.vocab_size),
                       dtype=np.float64)
        for index in range(len(sequence) - 1):
            self.observe(int(sequence[index]))
            features = self._features()
            gram += np.outer(features, features)
            rhs[:, int(sequence[index + 1])] += features
        penalty = self.ridge if ridge is None else float(ridge)
        self.readout = _ridge_readout(gram, rhs, penalty)
        self.train_loss = self.loss(sequence)
        self.train_top5 = self.topk_accuracy(sequence, 5)
        self.reset()
        return {"tokens": int(len(sequence)),
                "train_loss": float(self.train_loss),
                "train_top5": float(self.train_top5),
                "ridge": float(penalty),
                "feature_dim": int(self.feature_dim)}

    def _logits(self) -> np.ndarray:
        if self.readout is None:
            raise RuntimeError("fit() must be called before prediction")
        return self._features() @ self.readout

    def predict(self, context_ids: ArrayLike, top_k: int = 5
                ) -> tuple[np.ndarray, np.ndarray]:
        self.reset()
        sequence = _validate_ids(context_ids, self.vocab_size,
                                 name="context_ids")
        for word in sequence:
            self.observe(int(word))
        return top_k_ids(self._logits(), top_k)

    def loss(self, ids: ArrayLike) -> float:
        sequence = _validate_ids(ids, self.vocab_size)
        if len(sequence) < 2:
            return float("nan")
        if len(sequence) > self.max_fit_tokens:
            sequence = sequence[-self.max_fit_tokens:]
        self.reset()
        total = 0.0
        for index in range(len(sequence) - 1):
            self.observe(int(sequence[index]))
            total += _nll_from_logits(self._logits(),
                                      int(sequence[index + 1]))
        return total / (len(sequence) - 1)

    def topk_accuracy(self, ids: ArrayLike, top_k: int = 5) -> float:
        sequence = _validate_ids(ids, self.vocab_size)
        if len(sequence) < 2:
            return float("nan")
        if len(sequence) > self.max_fit_tokens:
            sequence = sequence[-self.max_fit_tokens:]
        self.reset()
        hits = 0
        for index in range(len(sequence) - 1):
            self.observe(int(sequence[index]))
            predicted, _ = top_k_ids(self._logits(), top_k)
            hits += int(np.any(predicted == sequence[index + 1]))
        return hits / (len(sequence) - 1)


# ---------------------------------------------------------------------------
# 6. Ant-colony / diffusion prior on the word transition graph
# ---------------------------------------------------------------------------


class AntColonyPrior:
    """Deterministic ant-flow diffusion over a word transition graph.

    ``fit`` counts observed transitions and stores them as a compact COO edge
    list.  Prediction seeds pheromone at the recent context, repeatedly moves
    mass along outgoing edges and restarts a fraction at the seed.  Accumulated
    visits are the score.  This is the vectorised mean-field limit of an ant
    walk: one ``bincount`` move per step, no per-ant loop.

    Cost: O(E) memory and O(steps * E) per prediction, where E is the number
    of distinct observed transitions.
    """

    def __init__(self, vocab_size: int, steps: int = 3,
                 evaporation: float = 0.35, seed_decay: float = 0.6,
                 restart: float = 0.35, unigram_weight: float = 0.05,
                 seed: int = DEFAULT_SEED):
        self.vocab_size = int(vocab_size)
        self.steps = int(steps)
        self.evaporation = float(evaporation)
        self.seed_decay = float(seed_decay)
        self.restart = float(restart)
        self.unigram_weight = float(unigram_weight)
        # ``seed`` is accepted for a uniform constructor signature; the
        # diffusion/ant-flow itself is deterministic.
        self.seed = int(seed)
        self.rows = np.zeros(0, dtype=np.int64)
        self.cols = np.zeros(0, dtype=np.int64)
        self.values = np.zeros(0, dtype=np.float64)
        self.unigram = np.zeros(self.vocab_size, dtype=np.float64)
        self.unigram_prob = np.full(self.vocab_size, 1.0 / self.vocab_size)
        self.word_bigram: dict[int, dict[int, float]] = {}
        self.edges = 0
        self.train_top5: float | None = None

    def fit(self, ids: ArrayLike) -> dict:
        """Count transitions and build the compact edge list."""
        sequence = _validate_ids(ids, self.vocab_size)
        if len(sequence) < 2:
            raise ValueError("fit needs at least two word IDs")
        self.unigram = np.bincount(sequence, minlength=self.vocab_size).astype(
            np.float64)
        self.unigram_prob = (self.unigram + 1e-6) / (
            self.unigram.sum() + 1e-6 * self.vocab_size)
        left = sequence[:-1]
        right = sequence[1:]
        pairs = left * self.vocab_size + right
        unique, counts = np.unique(pairs, return_counts=True)
        self.rows = (unique // self.vocab_size).astype(np.int64)
        self.cols = (unique % self.vocab_size).astype(np.int64)
        self.values = counts.astype(np.float64)
        self.edges = int(unique.size)
        bigram: dict[int, dict[int, float]] = defaultdict(dict)
        for key, count in zip(unique, counts):
            bigram[int(key // self.vocab_size)][int(key % self.vocab_size)] = (
                float(count))
        self.word_bigram = dict(bigram)
        self.train_top5 = self.topk_accuracy(sequence, 5)
        return {"tokens": int(len(sequence)), "edges": self.edges,
                "train_top5": float(self.train_top5)}

    def _seed_distribution(self, context_ids: np.ndarray) -> np.ndarray:
        seed = np.zeros(self.vocab_size, dtype=np.float64)
        if context_ids.size == 0:
            return self.unigram_prob.copy()
        for lag, word in enumerate(reversed(context_ids[-8:])):
            seed[int(word)] += self.seed_decay ** lag
        total = seed.sum()
        if total <= 0:
            return self.unigram_prob.copy()
        return seed / total

    def diffuse(self, context_ids: ArrayLike, *,
                steps: int | None = None) -> np.ndarray:
        """Return accumulated ant-visit mass over words."""
        sequence = _validate_ids(context_ids, self.vocab_size,
                                 name="context_ids")
        seed = self._seed_distribution(sequence)
        pheromone = seed.copy()
        visits = np.zeros(self.vocab_size, dtype=np.float64)
        n_steps = self.steps if steps is None else int(steps)
        for _ in range(max(0, n_steps)):
            if self.edges:
                flow = np.bincount(
                    self.cols,
                    weights=self.values * pheromone[self.rows],
                    minlength=self.vocab_size)
                total = float(flow.sum())
                if total > 0.0:
                    flow /= total
                else:
                    flow = self.unigram_prob.copy()
            else:
                flow = self.unigram_prob.copy()
            pheromone = ((1.0 - self.evaporation) * flow
                         + self.evaporation * seed)
            visits += pheromone
        if n_steps <= 0:
            visits = pheromone
        return visits

    def predict(self, context_ids: ArrayLike, top_k: int = 5
                ) -> tuple[np.ndarray, np.ndarray]:
        sequence = _validate_ids(context_ids, self.vocab_size,
                                 name="context_ids")
        scores = self.diffuse(sequence)
        if self.unigram_weight:
            scores = scores + self.unigram_weight * self.unigram_prob
        if sequence.size and sequence[-1] in self.word_bigram:
            for word, count in self.word_bigram[int(sequence[-1])].items():
                scores[word] += 0.05 * count / max(
                    sum(self.word_bigram[int(sequence[-1])].values()), 1.0)
        return top_k_ids(scores, top_k)

    def topk_accuracy(self, ids: ArrayLike, top_k: int = 5) -> float:
        sequence = _validate_ids(ids, self.vocab_size)
        if len(sequence) < 2:
            return float("nan")
        hits = 0
        for index in range(1, len(sequence)):
            context = sequence[max(0, index - 8):index]
            predicted, _ = self.predict(context, top_k)
            hits += int(np.any(predicted == sequence[index]))
        return hits / (len(sequence) - 1)


# ---------------------------------------------------------------------------
# 7. Tensor-train / matrix-product-state joint word model
# ---------------------------------------------------------------------------


class TensorTrainWordModel:
    """A small TT/MPS model of the joint distribution of ``order`` words.

    ``fit`` builds the dense empirical ``V x ... x V`` count tensor (guarded by
    ``max_tensor_entries``), applies TT-SVD, and stores the cores.  Conditioning
    on the last ``order - 1`` words contracts the prefix cores and leaves a
    vector of scores for the next word.

    This is intentionally a prototype: V**order must stay small.  Use it on
    clustered IDs or a modest vocabulary, not directly on 4k words with order
    4-6.  The default rank cap is 8; ranks up to 16 are supported.

    Joint probability is the TT contraction of the cores; the partition
    function is normalised at fit time.  Contexts shorter than ``order - 1``
    are left-padded with word 0 (a documented approximation for cold starts).
    """

    def __init__(self, vocab_size: int, order: int = 4, rank: int = 8,
                 smoothing: float = 0.25, seed: int = DEFAULT_SEED,
                 max_tensor_entries: int = 2_000_000,
                 k: int | None = None, chi: int | None = None):
        # ``k`` and ``chi`` are the TT/MPS notation aliases for order/rank.
        if k is not None:
            order = int(k)
        if chi is not None:
            rank = int(chi)
        self.vocab_size = int(vocab_size)
        self.order = int(order)
        if not 2 <= self.order <= 8:
            raise ValueError("order must be in [2, 8]")
        self.rank = int(rank)
        if not 1 <= self.rank <= 16:
            raise ValueError("rank (TT chi) must be in [1, 16]")
        self.smoothing = float(smoothing)
        self.max_tensor_entries = int(max_tensor_entries)
        self.seed = int(seed)
        self.cores = self._random_cores()
        self.fitted = False
        self.train_nll: float | None = None
        self.train_top5: float | None = None
        self.reconstruction_error: float | None = None

    # -- TT mechanics -----------------------------------------------------

    def _random_cores(self) -> list[np.ndarray]:
        rng = np.random.default_rng(self.seed)
        ranks = [1] + [self.rank] * (self.order - 1) + [1]
        cores = [
            rng.normal(0.0, 0.5,
                       (ranks[i], self.vocab_size, ranks[i + 1])).astype(
                           np.float64)
            for i in range(self.order)
        ]
        partition = self._partition(cores)
        if partition != 0.0:
            cores[0] /= partition
        return cores

    @staticmethod
    def _partition(cores: list[np.ndarray]) -> float:
        boundary = np.ones(1, dtype=np.float64)
        for core in reversed(cores):
            boundary = np.einsum("rvq,q->rv", core, boundary).sum(axis=1)
        return float(boundary[0])

    @staticmethod
    def _tt_svd(tensor: np.ndarray, order: int,
                rank: int) -> list[np.ndarray]:
        cores: list[np.ndarray] = []
        current = np.asarray(tensor, dtype=np.float64)
        left_rank = 1
        for _mode in range(order - 1):
            current = current.reshape(left_rank * tensor.shape[0], -1)
            u_mat, singular, vh_mat = np.linalg.svd(
                current, full_matrices=False)
            new_rank = int(min(rank, singular.size))
            u_mat = u_mat[:, :new_rank]
            singular = singular[:new_rank]
            vh_mat = vh_mat[:new_rank]
            cores.append(u_mat.reshape(left_rank, tensor.shape[0],
                                       new_rank))
            current = singular[:, None] * vh_mat
            left_rank = new_rank
        cores.append(current.reshape(left_rank, tensor.shape[0], 1))
        return cores

    def _to_dense(self) -> np.ndarray:
        output = self.cores[0][0]
        for core in self.cores[1:]:
            output = np.tensordot(output, core, axes=([-1], [0]))
        return np.squeeze(output, axis=-1)

    def joint_probability(self, sequence: ArrayLike) -> float:
        """TT mass for a full ``order``-word sequence.

        Truncated TT-SVD can overshoot slightly below zero for unseen
        sequences; the returned probability is clipped at zero.
        """
        ids = _validate_ids(sequence, self.vocab_size)
        if len(ids) != self.order:
            raise ValueError(f"sequence must have length {self.order}")
        boundary = np.ones((1, 1), dtype=np.float64)
        for core, word in zip(self.cores, ids):
            boundary = boundary @ np.take(core, int(word), axis=1)
        return max(0.0, float(boundary[0, 0]))

    def _context_logits(self, context_ids: ArrayLike) -> np.ndarray:
        ids = _validate_ids(context_ids, self.vocab_size,
                            name="context_ids")[-(self.order - 1):]
        if ids.size < self.order - 1:
            padding = np.zeros(self.order - 1 - ids.size, dtype=np.int64)
            ids = np.concatenate((padding, ids))
        left = np.ones((1, 1), dtype=np.float64)
        for core, word in zip(self.cores[:-1], ids):
            # Keep the left rank axis: the first core has shape (1, V, r).
            left = left @ np.take(core, int(word), axis=1)
        logits = left @ self.cores[-1][:, :, 0]
        return logits.reshape(-1).astype(np.float64)

    # -- training / inference --------------------------------------------

    def fit(self, ids: ArrayLike) -> dict:
        sequence = _validate_ids(ids, self.vocab_size)
        if len(sequence) < self.order:
            raise ValueError(f"need at least {self.order} word IDs")
        entries = self.vocab_size ** self.order
        if entries > self.max_tensor_entries:
            raise ValueError(
                f"dense V**order = {entries} exceeds max_tensor_entries="
                f"{self.max_tensor_entries}; cluster the vocabulary first")
        windows = np.lib.stride_tricks.sliding_window_view(
            sequence, self.order)
        flat = np.ravel_multi_index(
            windows.T, (self.vocab_size,) * self.order)
        counts = np.bincount(flat, minlength=entries).reshape(
            (self.vocab_size,) * self.order).astype(np.float64)
        tensor = counts + self.smoothing
        tensor /= tensor.sum()
        initial_nll = self.nll(sequence)
        cores = self._tt_svd(tensor, self.order, self.rank)
        partition = self._partition(cores)
        if partition != 0.0:
            cores[0] /= partition
        self.cores = cores
        self.fitted = True
        dense = self._to_dense()
        self.reconstruction_error = float(
            np.linalg.norm(dense - tensor) / (np.linalg.norm(tensor) + 1e-12))
        self.train_nll = self.nll(sequence)
        self.train_top5 = self.topk_accuracy(sequence, 5)
        word_counts = np.bincount(sequence, minlength=self.vocab_size)
        unigram = (word_counts + self.smoothing)
        unigram /= unigram.sum()
        unigram_nll = float(-np.log(unigram[sequence]).mean())
        return {"tokens": int(len(sequence)),
                "tensor_entries": int(entries),
                "rank": int(self.rank),
                "initial_nll": float(initial_nll),
                "train_nll": float(self.train_nll),
                "unigram_nll": unigram_nll,
                "reconstruction_error": self.reconstruction_error,
                "train_top5": float(self.train_top5)}

    def nll(self, ids: ArrayLike) -> float:
        sequence = _validate_ids(ids, self.vocab_size)
        if len(sequence) <= self.order - 1:
            return float("nan")
        total = 0.0
        for index in range(self.order - 1, len(sequence)):
            logits = self._context_logits(sequence[index - self.order + 1:
                                                    index])
            total += _nll_from_logits(logits, int(sequence[index]))
        return total / (len(sequence) - self.order + 1)

    def predict(self, context_ids: ArrayLike, top_k: int = 5
                ) -> tuple[np.ndarray, np.ndarray]:
        return top_k_ids(self._context_logits(context_ids), top_k)

    def topk_accuracy(self, ids: ArrayLike, top_k: int = 5) -> float:
        sequence = _validate_ids(ids, self.vocab_size)
        if len(sequence) <= self.order - 1:
            return float("nan")
        hits = 0
        total = 0
        for index in range(self.order - 1, len(sequence)):
            predicted, _ = self.predict(
                sequence[index - self.order + 1:index], top_k)
            hits += int(np.any(predicted == sequence[index]))
            total += 1
        return hits / max(total, 1)


# ---------------------------------------------------------------------------
# 8. Re-Pair renormalisation blocks
# ---------------------------------------------------------------------------


def _replace_pair(symbols: np.ndarray, first: int, second: int,
                  new_symbol: int) -> tuple[np.ndarray, int]:
    """Non-overlapping left-to-right replacement of one adjacent pair."""
    positions = np.flatnonzero((symbols[:-1] == first)
                               & (symbols[1:] == second))
    if positions.size == 0:
        return symbols, 0
    chosen: list[int] = []
    last = -2
    for position in positions:
        position = int(position)
        if position > last:
            chosen.append(position)
            last = position + 1
    output: list[int] = []
    cursor = 0
    for position in chosen:
        output.extend(symbols[cursor:position].tolist())
        output.append(int(new_symbol))
        cursor = position + 2
    output.extend(symbols[cursor:].tolist())
    return np.asarray(output, dtype=np.int64), len(chosen)


class RenormalizationBlockModel:
    """Re-Pair phrase induction plus block-level next-word prediction.

    ``fit`` repeatedly replaces the most frequent adjacent symbol pair with a
    new block symbol.  The final block stream is used to count block n-grams
    and the first terminal word that follows them.  Prediction encodes the
    context with the learned rules, matches the longest available block
    suffix, and backs off to word bigrams and unigrams.

    This is the "renormalisation group" prior: higher-level blocks are the
    coarse-grained units of the stream.  Re-Pair's pair counting is NumPy
    vectorised, but the replacement pass is a Python loop; keep training
    streams to roughly 100k words or less for interactive use.
    """

    def __init__(self, vocab_size: int, max_rules: int = 24,
                 min_count: int = 2, block_order: int = 3,
                 context_limit: int = 8, block_weight: float = 1.0,
                 word_weight: float = 0.25,
                 unigram_weight: float = 0.05,
                 seed: int = DEFAULT_SEED):
        # ``seed`` is accepted for a uniform constructor signature; Re-Pair is
        # deterministic and has no random parameters.
        self.seed = int(seed)
        self.vocab_size = int(vocab_size)
        self.max_rules = int(max_rules)
        self.min_count = int(min_count)
        self.block_order = int(block_order)
        self.context_limit = int(context_limit)
        self.block_weight = float(block_weight)
        self.word_weight = float(word_weight)
        self.unigram_weight = float(unigram_weight)
        self.rules: list[tuple[int, int, int]] = []
        self.first_terminal = np.arange(self.vocab_size, dtype=np.int64)
        self.block_length = np.ones(self.vocab_size, dtype=np.int64)
        self.block_ngram_counts: dict[tuple[int, ...], dict[int, int]] = {}
        self.word_bigram: dict[int, dict[int, float]] = {}
        self.unigram = np.zeros(self.vocab_size, dtype=np.float64)
        self.unigram_prob = np.full(self.vocab_size,
                                    1.0 / self.vocab_size)
        self.final_symbols = np.zeros(0, dtype=np.int64)
        self.fitted = False
        self.train_top5: float | None = None

    # -- grammar ----------------------------------------------------------

    def fit(self, ids: ArrayLike) -> dict:
        sequence = _validate_ids(ids, self.vocab_size)
        if len(sequence) < 2:
            raise ValueError("fit needs at least two word IDs")
        symbols = sequence.copy()
        rules: list[tuple[int, int, int]] = []
        first_terminal = list(range(self.vocab_size))
        block_length = [1] * self.vocab_size
        for _rule_index in range(self.max_rules):
            if symbols.size < 2:
                break
            base = max(self.vocab_size, int(symbols.max()) + 1)
            pairs = symbols[:-1] * base + symbols[1:]
            unique, counts = np.unique(pairs, return_counts=True)
            best = int(np.argmax(counts))
            if int(counts[best]) < self.min_count:
                break
            first = int(unique[best] // base)
            second = int(unique[best] % base)
            new_symbol = self.vocab_size + len(rules)
            replaced, n_replaced = _replace_pair(
                symbols, first, second, new_symbol)
            if n_replaced == 0:
                break
            symbols = replaced
            rules.append((first, second, new_symbol))
            first_terminal.append(first_terminal[first])
            block_length.append(block_length[first] + block_length[second])
        self.rules = rules
        self.first_terminal = np.asarray(first_terminal, dtype=np.int64)
        self.block_length = np.asarray(block_length, dtype=np.int64)
        self.final_symbols = symbols

        self.unigram = np.bincount(sequence, minlength=self.vocab_size).astype(
            np.float64)
        self.unigram_prob = (self.unigram + 1e-6) / (
            self.unigram.sum() + 1e-6 * self.vocab_size)
        pairs = sequence[:-1] * self.vocab_size + sequence[1:]
        unique, counts = np.unique(pairs, return_counts=True)
        bigram: dict[int, dict[int, float]] = defaultdict(dict)
        for key, count in zip(unique, counts):
            bigram[int(key // self.vocab_size)][int(key % self.vocab_size)] = (
                float(count))
        self.word_bigram = dict(bigram)

        block_counts: dict[tuple[int, ...], dict[int, int]] = {}
        for position in range(len(symbols) - 1):
            next_word = int(self.first_terminal[int(symbols[position + 1])])
            for order in range(1, min(self.block_order, position + 1) + 1):
                key = tuple(int(x) for x in symbols[position - order + 1:
                                                     position + 1])
                row = block_counts.get(key)
                if row is None:
                    row = {}
                    block_counts[key] = row
                row[next_word] = row.get(next_word, 0) + 1
        self.block_ngram_counts = block_counts
        self.fitted = True
        self.train_top5 = self.topk_accuracy(sequence, 5)
        compression = len(symbols) / max(len(sequence), 1)
        return {"tokens": int(len(sequence)),
                "rules": len(self.rules),
                "compression": float(compression),
                "mean_block_length": float(self.block_length[-len(self.rules):
                                                             ].mean())
                if self.rules else 1.0,
                "train_top5": float(self.train_top5)}

    def encode(self, ids: ArrayLike) -> np.ndarray:
        """Apply the learned rules in order to a word-ID sequence."""
        symbols = _validate_ids(ids, self.vocab_size,
                                name="context_ids").copy()
        for first, second, new_symbol in self.rules:
            if symbols.size < 2:
                break
            symbols, _ = _replace_pair(symbols, first, second, new_symbol)
        return symbols

    def expand(self, symbol: int) -> list[int]:
        """Expand a block symbol into its terminal word sequence."""
        output: list[int] = []
        stack = [int(symbol)]
        while stack:
            current = stack.pop()
            if current < self.vocab_size:
                output.append(current)
            else:
                index = current - self.vocab_size
                if index < 0 or index >= len(self.rules):
                    raise ValueError(f"unknown block symbol {current}")
                first, second, _ = self.rules[index]
                stack.append(second)
                stack.append(first)
        return output

    # -- prediction -------------------------------------------------------

    def _block_distribution(self, encoded: np.ndarray) -> np.ndarray:
        distribution = np.zeros(self.vocab_size, dtype=np.float64)
        found = False
        for order in range(min(self.block_order, len(encoded)), 0, -1):
            key = tuple(int(x) for x in encoded[-order:])
            row = self.block_ngram_counts.get(key)
            if row:
                total = float(sum(row.values()))
                for word, count in row.items():
                    distribution[int(word)] += count / total
                found = True
                break
        if not found:
            return distribution
        total = distribution.sum()
        return distribution / total if total > 0 else distribution

    def predict(self, context_ids: ArrayLike, top_k: int = 5
                ) -> tuple[np.ndarray, np.ndarray]:
        sequence = _validate_ids(context_ids, self.vocab_size,
                                 name="context_ids")[-self.context_limit:]
        scores = np.zeros(self.vocab_size, dtype=np.float64)
        block = np.zeros(self.vocab_size, dtype=np.float64)
        if sequence.size:
            encoded = self.encode(sequence)
            block = self._block_distribution(encoded)
        word = np.zeros(self.vocab_size, dtype=np.float64)
        if sequence.size:
            last = int(sequence[-1])
            row = self.word_bigram.get(last)
            if row:
                total = float(sum(row.values()))
                for word_id, count in row.items():
                    word[int(word_id)] = count / total
        scores += self.block_weight * block
        scores += self.word_weight * word
        scores += self.unigram_weight * self.unigram_prob
        return top_k_ids(scores, top_k)

    def topk_accuracy(self, ids: ArrayLike, top_k: int = 5) -> float:
        sequence = _validate_ids(ids, self.vocab_size)
        if len(sequence) < 2:
            return float("nan")
        hits = 0
        for index in range(1, len(sequence)):
            context = sequence[max(0, index - self.context_limit):index]
            predicted, _ = self.predict(context, top_k)
            hits += int(np.any(predicted == sequence[index]))
        return hits / (len(sequence) - 1)


# ---------------------------------------------------------------------------
# 9. Free-energy switching regimes (HMM / Baum-Welch)
# ---------------------------------------------------------------------------


class FreeEnergyRegimeModel:
    """A K-regime latent-variable word model trained with EM.

    Hidden regime ``z_t`` follows a sticky Markov chain.  Emissions are
    regime-specific unigram distributions over words.  Baum-Welch maximises
    the exact evidence lower bound (for an HMM the ELBO is the log evidence),
    and prediction uses the filtered regime posterior followed by one
    transition and emission step.

    Cost: O(epochs * T * K^2), with K in the requested 32-64 range.  The fit
    cap defaults to 20k tokens; longer streams are fit on their first slice.
    """

    def __init__(self, vocab_size: int, n_regimes: int = 32,
                 transition_smoothing: float = 0.01,
                 emission_smoothing: float = 0.01,
                 seed: int = DEFAULT_SEED, max_fit_tokens: int = 20000,
                 K: int | None = None):
        # ``K`` is an alias for the number of latent regimes.
        if K is not None:
            n_regimes = int(K)
        self.vocab_size = int(vocab_size)
        self.n_regimes = int(n_regimes)
        if not 2 <= self.n_regimes <= 128:
            raise ValueError("n_regimes must be in [2, 128]")
        self.transition_smoothing = float(transition_smoothing)
        self.emission_smoothing = float(emission_smoothing)
        self.max_fit_tokens = int(max_fit_tokens)
        rng = np.random.default_rng(int(seed))
        self.pi = np.full(self.n_regimes, 1.0 / self.n_regimes)
        self.A = np.full((self.n_regimes, self.n_regimes),
                         0.15 / max(self.n_regimes - 1, 1))
        np.fill_diagonal(self.A, 0.85)
        emissions = rng.random((self.n_regimes, self.vocab_size)) + 0.05
        self.B = emissions / emissions.sum(axis=1, keepdims=True)
        self.history: list[float] = []
        self.train_loss: float | None = None
        self.train_top5: float | None = None

    # -- inference --------------------------------------------------------

    def _forward(self, sequence: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, float]:
        length = len(sequence)
        alpha = np.empty((length, self.n_regimes), dtype=np.float64)
        scales = np.empty(length, dtype=np.float64)
        current = self.pi * self.B[:, int(sequence[0])]
        scale = max(float(current.sum()), 1e-300)
        alpha[0] = current / scale
        scales[0] = scale
        for step in range(1, length):
            current = (alpha[step - 1] @ self.A) * self.B[
                :, int(sequence[step])]
            scale = max(float(current.sum()), 1e-300)
            alpha[step] = current / scale
            scales[step] = scale
        return alpha, scales, float(np.log(scales).sum())

    def _forward_backward(self, sequence: np.ndarray
                          ) -> tuple[float, np.ndarray, np.ndarray]:
        alpha, scales, log_likelihood = self._forward(sequence)
        length = len(sequence)
        beta = np.ones((length, self.n_regimes), dtype=np.float64)
        for step in range(length - 2, -1, -1):
            beta[step] = (
                self.A @ (self.B[:, int(sequence[step + 1])]
                          * beta[step + 1])) / scales[step + 1]
        gamma = alpha * beta
        gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)
        xi_sum = np.zeros((self.n_regimes, self.n_regimes),
                          dtype=np.float64)
        for step in range(length - 1):
            xi = (np.outer(alpha[step],
                           self.B[:, int(sequence[step + 1])]
                           * beta[step + 1])
                  * self.A / scales[step + 1])
            xi_sum += xi
        return log_likelihood, gamma, xi_sum

    def log_likelihood(self, ids: ArrayLike) -> float:
        sequence = _validate_ids(ids, self.vocab_size)
        if sequence.size == 0:
            return float("nan")
        return self._forward(sequence)[2]

    def elbo(self, ids: ArrayLike) -> float:
        """Evidence lower bound; exact log evidence for this HMM."""
        return self.log_likelihood(ids)

    def filtered_state(self, context_ids: ArrayLike) -> np.ndarray:
        sequence = _validate_ids(context_ids, self.vocab_size,
                                 name="context_ids")
        if sequence.size == 0:
            return self.pi.copy()
        state = self.pi * self.B[:, int(sequence[0])]
        state /= max(float(state.sum()), 1e-300)
        for word in sequence[1:]:
            state = (state @ self.A) * self.B[:, int(word)]
            state /= max(float(state.sum()), 1e-300)
        return state

    # -- training ---------------------------------------------------------

    def fit(self, ids: ArrayLike, epochs: int = 20,
            tol: float = 1e-4) -> dict:
        sequence = _validate_ids(ids, self.vocab_size)
        if len(sequence) < 2:
            raise ValueError("fit needs at least two word IDs")
        if len(sequence) > self.max_fit_tokens:
            sequence = sequence[:self.max_fit_tokens]
        self.history = []
        previous = -np.inf
        for epoch in range(max(1, int(epochs))):
            log_likelihood, gamma, xi_sum = self._forward_backward(sequence)
            per_token = log_likelihood / len(sequence)
            self.history.append(float(per_token))
            # M-step.
            self.pi = gamma[0] + 1e-9
            self.pi /= self.pi.sum()
            self.A = xi_sum + self.transition_smoothing
            self.A /= self.A.sum(axis=1, keepdims=True)
            emission_counts = np.empty_like(self.B)
            for regime in range(self.n_regimes):
                emission_counts[regime] = np.bincount(
                    sequence, weights=gamma[:, regime],
                    minlength=self.vocab_size)
            self.B = emission_counts + self.emission_smoothing
            self.B /= self.B.sum(axis=1, keepdims=True)
            if epoch > 0 and per_token - previous < tol:
                break
            previous = per_token
        # Evaluate the final parameters, not the pre-M-step snapshot that was
        # appended to the history.
        final_elbo = -self.loss(sequence)
        self.train_loss = -final_elbo
        self.train_top5 = self.topk_accuracy(sequence, 5)
        return {"tokens": int(len(sequence)),
                "epochs": int(len(self.history)),
                "final_elbo_per_token": float(final_elbo),
                "elbo_gain": float(final_elbo - self.history[0]),
                "train_loss": float(self.train_loss),
                "train_top5": float(self.train_top5)}

    # -- prediction / metrics --------------------------------------------

    def predict(self, context_ids: ArrayLike, top_k: int = 5
                ) -> tuple[np.ndarray, np.ndarray]:
        state = self.filtered_state(context_ids)
        probabilities = (state @ self.A) @ self.B
        probabilities = np.maximum(probabilities, 1e-300)
        return top_k_ids(np.log(probabilities), top_k)

    def loss(self, ids: ArrayLike) -> float:
        sequence = _validate_ids(ids, self.vocab_size)
        if sequence.size == 0:
            return float("nan")
        return -self.log_likelihood(sequence) / len(sequence)

    def topk_accuracy(self, ids: ArrayLike, top_k: int = 5) -> float:
        sequence = _validate_ids(ids, self.vocab_size)
        if len(sequence) < 2:
            return float("nan")
        # One filtering pass instead of replaying every prefix (O(T) not
        # O(T^2) in the number of forward calls).
        state = self.pi * self.B[:, int(sequence[0])]
        state /= max(float(state.sum()), 1e-300)
        hits = 0
        for index in range(len(sequence) - 1):
            probabilities = (state @ self.A) @ self.B
            predicted, _ = top_k_ids(np.log(np.maximum(probabilities,
                                                       1e-300)), top_k)
            target = int(sequence[index + 1])
            hits += int(np.any(predicted == target))
            state = (state @ self.A) * self.B[:, target]
            state /= max(float(state.sum()), 1e-300)
        return hits / (len(sequence) - 1)
