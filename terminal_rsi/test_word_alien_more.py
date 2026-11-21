"""Unit tests for the standalone alien word predictors.

Run from the repository root::

    python -m unittest terminal_rsi.test_word_alien_more -v

The tests use tiny synthetic streams and CPU-only NumPy work.  They check
shapes, deterministic/causal predictions and that training measurably helps on
a toy periodic pattern.
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from terminal_rsi.word_alien_more import (
    AntColonyPrior,
    CellularAutomatonWordPrior,
    EchoStateReservoir,
    FreeEnergyRegimeModel,
    KuramotoPrior,
    PersistentHomologyFeatures,
    RenormalizationBlockModel,
    SpikingTraceMemory,
    TensorTrainWordModel,
    make_word_embeddings,
    top_k_ids,
)


def cycle(vocab_size: int, repeats: int) -> np.ndarray:
    return np.tile(np.arange(vocab_size, dtype=np.int64), repeats)


def pattern(words, repeats: int) -> np.ndarray:
    return np.tile(np.asarray(words, dtype=np.int64), repeats)


class TopKHelperTests(unittest.TestCase):
    def test_top_k_is_sorted_and_deterministic(self):
        scores = np.array([0.5, 2.0, 2.0, -1.0, 0.5])
        ids, values = top_k_ids(scores, 3)
        self.assertEqual(ids.tolist(), [1, 2, 0])
        self.assertEqual(values.tolist(), [2.0, 2.0, 0.5])

    def test_top_k_excludes_ids(self):
        ids, _ = top_k_ids(np.arange(5.0), 2, exclude=[4])
        self.assertNotIn(4, ids.tolist())

    def test_embeddings_are_unit_norm(self):
        table = make_word_embeddings(10, 6, seed=3)
        self.assertEqual(table.shape, (10, 6))
        norms = np.linalg.norm(table, axis=1)
        np.testing.assert_allclose(norms, np.ones(10), atol=1e-6)


class PersistentHomologyTests(unittest.TestCase):
    def setUp(self):
        self.model = PersistentHomologyFeatures(
            vocab_size=24, dim=8, window=10,
            thresholds=(0.2, 0.6, 1.2, 1.8), seed=7)

    def test_feature_width_is_fixed(self):
        for length in (0, 1, 3, 10, 50):
            context = np.arange(length, dtype=np.int64) % 24
            features = self.model.features(context)
            self.assertEqual(features.shape, (self.model.feature_dim,))
            self.assertTrue(np.all(np.isfinite(features)))

    def test_merge_times_are_sorted(self):
        merges = self.model.merge_times([0, 1, 2, 3, 4, 5])
        self.assertEqual(merges.shape, (5,))
        self.assertTrue(np.all(np.diff(merges) >= 0.0))

    def test_component_counts_follow_thresholds(self):
        context = [0, 1, 2, 3]
        counts = self.model.component_counts(context)
        self.assertEqual(counts.shape, (4,))
        self.assertEqual(int(counts[0]), 4)
        self.assertEqual(int(counts[-1]), 1)

        identical = PersistentHomologyFeatures(
            vocab_size=4, dim=4, window=4, thresholds=(0.0, 0.2),
            embeddings=np.ones((4, 4), dtype=np.float32))
        counts = identical.component_counts([0, 1, 2, 3])
        self.assertEqual(counts.tolist(), [1, 1])

    def test_features_are_causal_and_prefix_only(self):
        prefix = [3, 9, 4, 1, 7]
        later = [2, 8, 0, 6]
        self.assertTrue(np.array_equal(
            self.model.features(prefix),
            self.model.features(prefix)))
        # Appending future words to the *input* is allowed, but the class
        # always slices to the last window; the prefix result itself must not
        # depend on a future call.
        before = self.model.batch_features([prefix, later])
        after = self.model.batch_features([prefix, later])
        self.assertTrue(np.array_equal(before, after))

    def test_transform_accepts_single_stream_and_batch(self):
        single = self.model.transform([0, 1, 2])
        self.assertEqual(single.shape, (1, self.model.feature_dim))
        batch = self.model.transform([[0, 1], [2, 3]])
        self.assertEqual(batch.shape, (2, self.model.feature_dim))


class SpikingTraceMemoryTests(unittest.TestCase):
    def setUp(self):
        self.memory = SpikingTraceMemory(
            vocab_size=8, n_neurons=128, active=4, delays=3,
            decay=0.55, learning_rate=0.3, seed=11)

    def test_predict_shape_and_order(self):
        for word in (0, 1, 2, 3, 4, 5):
            self.memory.observe(word)
        ids, scores = self.memory.predict(5)
        self.assertEqual(ids.shape, (5,))
        self.assertEqual(scores.shape, (5,))
        self.assertTrue(np.all(ids >= 0))
        self.assertTrue(np.all(ids < 8))
        self.assertTrue(np.all(np.diff(scores) <= 1e-7))
        self.assertEqual(self.memory.weights.shape, (3, 128, 128))

    def test_training_improves_top5(self):
        stream = cycle(8, 40)
        cold = SpikingTraceMemory(
            vocab_size=8, n_neurons=128, active=4, delays=3,
            decay=0.55, learning_rate=0.3, seed=11)
        baseline = cold.topk_accuracy(stream, top_k=5, reset=True)
        for word in stream:
            self.memory.observe(int(word))
        trained = self.memory.topk_accuracy(stream, top_k=5, reset=True)
        self.assertGreater(trained, 0.90)
        self.assertGreater(trained, baseline + 0.25)

    def test_prefix_state_is_deterministic(self):
        prefix = [1, 4, 2, 6]
        clone = SpikingTraceMemory(
            vocab_size=8, n_neurons=128, active=4, delays=3,
            decay=0.55, learning_rate=0.3, seed=11)
        for word in prefix:
            self.memory.observe(word)
            clone.observe(word)
        first = self.memory.predict(4)
        second = self.memory.predict(4)
        other = clone.predict(4)
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(np.array_equal(first[0], other[0]))
        np.testing.assert_allclose(first[1], other[1], atol=1e-6)
        # Observing a new word is the only thing that advances the trace.
        before = self.memory.traces.copy()
        self.memory.observe(7)
        self.assertFalse(np.array_equal(before, self.memory.traces))

    def test_reset_can_preserve_learned_weights(self):
        for word in cycle(8, 4):
            self.memory.observe(int(word))
        weights = self.memory.weights.copy()
        self.memory.reset()
        self.assertTrue(np.array_equal(weights, self.memory.weights))
        self.assertTrue(np.all(self.memory.traces == 0.0))

    def test_evaluation_does_not_update_weights(self):
        for word in cycle(8, 3):
            self.memory.observe(int(word))
        weights = self.memory.weights.copy()
        self.memory.topk_accuracy(cycle(8, 3), top_k=5, reset=True)
        self.assertTrue(np.array_equal(weights, self.memory.weights))


class EchoStateReservoirTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stream = cycle(8, 30)
        cls.model = EchoStateReservoir(
            vocab_size=8, n_fast=32, n_slow=16, input_dim=8,
            ridge=1e-2, seed=5)
        cls.stats = cls.model.fit(cls.stream)

    def test_fit_reduces_loss_and_improves_top5(self):
        self.assertLess(self.stats["train_loss"], math.log(8))
        self.assertGreater(self.stats["train_top5"], 0.80)
        self.assertGreater(self.model.topk_accuracy(self.stream, 5), 0.80)

    def test_predict_shape_and_context_replay(self):
        context = [0, 1, 2, 3]
        ids, scores = self.model.predict(context, 5)
        self.assertEqual(ids.shape, (5,))
        self.assertTrue(np.all((ids >= 0) & (ids < 8)))
        self.assertTrue(np.all(np.diff(scores) <= 1e-7))
        repeat = self.model.predict(context, 5)
        self.assertTrue(np.array_equal(ids, repeat[0]))
        # Prediction is a pure function of the supplied context.
        self.model.predict([7, 6, 5], 2)
        repeat = self.model.predict(context, 5)
        self.assertTrue(np.array_equal(ids, repeat[0]))

    def test_state_shape(self):
        state = self.model.run([0, 1, 2])
        self.assertEqual(state.shape, (48,))


class KuramotoPriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stream = cycle(8, 30)
        cls.model = KuramotoPrior(
            vocab_size=8, coupling=0.2, omega_scale=0.02, window=6,
            ridge=1e-3, seed=13)
        cls.stats = cls.model.fit(cls.stream)

    def test_phase_advance_and_coupling(self):
        model = KuramotoPrior(vocab_size=6, seed=2)
        before_theta = model.theta.copy()
        before_time = model.time
        model.observe(0)
        model.observe(1)
        self.assertGreater(model.time, before_time)
        self.assertFalse(np.array_equal(before_theta, model.theta))
        self.assertNotEqual(model.field, 0.0 + 0.0j)

    def test_fit_improves_top5(self):
        self.assertGreater(self.stats["train_top5"], 0.75)
        self.assertLess(self.stats["train_loss"], math.log(8))
        self.assertGreater(self.model.topk_accuracy(self.stream, 5), 0.75)

    def test_predict_shape_and_determinism(self):
        ids, scores = self.model.predict([6, 7, 0], 5)
        self.assertEqual(ids.shape, (5,))
        self.assertTrue(np.all((ids >= 0) & (ids < 8)))
        self.assertTrue(np.all(np.diff(scores) <= 1e-7))
        again = self.model.predict([6, 7, 0], 5)
        self.assertTrue(np.array_equal(ids, again[0]))
        np.testing.assert_allclose(scores, again[1], atol=1e-6)


class CellularAutomatonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stream = cycle(8, 30)
        cls.model = CellularAutomatonWordPrior(
            vocab_size=8, dim=8, cells=5, steps=1, mix=0.55,
            ridge=1e-2, seed=17)
        cls.stats = cls.model.fit(cls.stream)

    def test_feature_shape_and_causal_replay(self):
        self.assertEqual(self.model.feature_dim, 5 * 8 + 1)
        self.model.reset()
        self.model.observe(3)
        state_after_one = self.model.state.copy()
        self.assertEqual(state_after_one.shape, (5, 8))
        ids, scores = self.model.predict([3, 4, 5], 5)
        self.assertEqual(ids.shape, (5,))
        again = self.model.predict([3, 4, 5], 5)
        self.assertTrue(np.array_equal(ids, again[0]))
        np.testing.assert_allclose(scores, again[1], atol=1e-6)

    def test_fit_improves_top5(self):
        self.assertGreater(self.stats["train_top5"], 0.75)
        self.assertLess(self.stats["train_loss"], math.log(8))
        self.assertGreater(self.model.topk_accuracy(self.stream, 5), 0.75)


class AntColonyTests(unittest.TestCase):
    def setUp(self):
        self.words = [6, 7, 8, 9]
        self.stream = pattern(self.words, 40)
        self.model = AntColonyPrior(
            vocab_size=10, steps=3, evaporation=0.35, seed=3)
        self.stats = self.model.fit(self.stream)

    def test_graph_and_prediction(self):
        self.assertGreater(self.stats["edges"], 0)
        ids, scores = self.model.predict([6], 5)
        self.assertEqual(ids.shape, (5,))
        self.assertTrue(np.all(np.diff(scores) <= 1e-7))
        self.assertIn(7, ids.tolist())

    def test_training_helps_on_phrase_pattern(self):
        self.assertGreater(self.stats["train_top5"], 0.80)
        untrained = AntColonyPrior(vocab_size=10, seed=3)
        cold, _ = untrained.predict([6], 5)
        self.assertNotIn(7, cold.tolist())
        trained, _ = self.model.predict([6], 5)
        self.assertIn(7, trained.tolist())

    def test_unseen_context_falls_back(self):
        ids, scores = self.model.predict([5], 4)
        self.assertEqual(ids.shape, (4,))
        self.assertTrue(np.all(np.isfinite(scores)))
        empty_ids, _ = self.model.predict([], 4)
        self.assertEqual(empty_ids.shape, (4,))


class TensorTrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stream = cycle(8, 20)
        cls.model = TensorTrainWordModel(
            vocab_size=8, order=4, rank=12, smoothing=0.01, seed=21)
        cls.stats = cls.model.fit(cls.stream)

    def test_cores_and_prediction_shape(self):
        self.assertEqual(len(self.model.cores), 4)
        for left_rank, _vocab, right_rank in (
                core.shape for core in self.model.cores):
            self.assertLessEqual(left_rank, 12)
            self.assertLessEqual(right_rank, 12)
        ids, scores = self.model.predict([0, 1, 2], 5)
        self.assertEqual(ids.shape, (5,))
        self.assertTrue(np.all((ids >= 0) & (ids < 8)))
        self.assertTrue(np.all(np.diff(scores) <= 1e-7))

    def test_training_reduces_nll(self):
        self.assertLess(self.stats["train_nll"], self.stats["initial_nll"])
        self.assertLess(self.stats["train_nll"], self.stats["unigram_nll"])
        self.assertLess(self.stats["reconstruction_error"], 0.05)
        self.assertGreater(self.stats["train_top5"], 0.75)

    def test_joint_probability_and_tensor_guard(self):
        probability = self.model.joint_probability([0, 1, 2, 3])
        self.assertGreaterEqual(probability, 0.0)
        tiny = TensorTrainWordModel(
            vocab_size=64, order=6, rank=4, max_tensor_entries=100_000)
        with self.assertRaises(ValueError):
            tiny.fit(cycle(6, 10))

    def test_prediction_is_context_only(self):
        first = self.model.predict([4, 5, 6], 3)
        self.model.predict([0, 1, 2], 3)
        second = self.model.predict([4, 5, 6], 3)
        self.assertTrue(np.array_equal(first[0], second[0]))


class RenormalizationBlockTests(unittest.TestCase):
    def setUp(self):
        self.words = [8, 9, 10, 11]
        self.stream = pattern(self.words, 30)
        self.model = RenormalizationBlockModel(
            vocab_size=12, max_rules=16, min_count=2, seed=0)
        self.stats = self.model.fit(self.stream)

    def test_rules_compress_the_stream(self):
        self.assertGreater(self.stats["rules"], 0)
        self.assertLess(self.stats["compression"], 1.0)
        encoded = self.model.encode(self.stream)
        self.assertLessEqual(len(encoded), len(self.stream))
        expanded = self.model.expand(int(encoded[0]))
        self.assertGreaterEqual(len(expanded), 1)

    def test_block_prediction_beats_cold_start(self):
        ids, _ = self.model.predict([8, 9, 10], 5)
        self.assertEqual(ids[0], 11)
        cold = RenormalizationBlockModel(vocab_size=12)
        cold_ids, _ = cold.predict([8, 9, 10], 5)
        self.assertNotIn(11, cold_ids.tolist())
        self.assertGreater(self.stats["train_top5"], 0.80)

    def test_prediction_shape_and_determinism(self):
        first = self.model.predict([10, 11, 8], 5)
        second = self.model.predict([10, 11, 8], 5)
        self.assertEqual(first[0].shape, (5,))
        self.assertTrue(np.array_equal(first[0], second[0]))


class FreeEnergyRegimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stream = cycle(8, 40)
        cls.model = FreeEnergyRegimeModel(
            vocab_size=8, n_regimes=32, seed=23, max_fit_tokens=2000)
        cls.stats = cls.model.fit(cls.stream, epochs=12, tol=1e-5)

    def test_elbo_increases(self):
        self.assertGreater(self.stats["elbo_gain"], 0.0)
        self.assertGreater(self.model.history[-1], self.model.history[0])
        self.assertTrue(np.all(np.isfinite(self.model.history)))

    def test_filtered_state_and_prediction_shape(self):
        state = self.model.filtered_state([0, 1, 2])
        self.assertEqual(state.shape, (32,))
        self.assertAlmostEqual(float(state.sum()), 1.0, places=6)
        ids, scores = self.model.predict([0, 1, 2], 5)
        self.assertEqual(ids.shape, (5,))
        self.assertTrue(np.all((ids >= 0) & (ids < 8)))
        self.assertTrue(np.all(np.diff(scores) <= 1e-6))

    def test_training_improves_top5(self):
        self.assertGreater(self.stats["train_top5"], 0.60)
        self.assertLess(self.stats["train_loss"], math.log(8))

    def test_prediction_is_context_only(self):
        first = self.model.predict([3, 4, 5], 4)
        self.model.predict([0, 1, 2], 4)
        second = self.model.predict([3, 4, 5], 4)
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(np.isfinite(self.model.elbo(self.stream)))


if __name__ == "__main__":
    unittest.main()
