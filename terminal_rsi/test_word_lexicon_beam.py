#!/usr/bin/env python3
"""Pure-Python tests for ``word_lexicon_beam``.

Run with::

    python -m unittest terminal_rsi.test_word_lexicon_beam

No torch is imported unless the optional scorer tests can run.  The torch
tests are marked skipped in this Termux environment (torch is not installed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

try:
    from . import word_lexicon_beam as wlb
except ImportError:  # direct execution
    import word_lexicon_beam as wlb  # type: ignore


def fake_logp(context: bytes, child: int) -> float:
    """Deterministic proper 256-way distribution depending on the context."""
    seed = 0
    for index, byte in enumerate(context):
        seed = (seed * 131 + byte + index) & 0xFFFFFFFF
    logits = [(-((value * 37 + seed) % 113)) / 10.0 for value in range(256)]
    largest = max(logits)
    logz = largest + math.log(sum(math.exp(v - largest) for v in logits))
    return logits[child] - logz


class FakeScorer:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, list[int]]] = []

    def score_pairs(self, pairs):
        output = []
        for context, children in pairs:
            self.calls.append((bytes(context), [int(c) for c in children]))
            output.append([fake_logp(context, int(c)) for c in children])
        return output


def brute_force(base: bytes, word: bytes) -> float:
    return sum(fake_logp(base + word[:index], word[index])
               for index in range(len(word)))


class TestWordTrie(unittest.TestCase):
    def setUp(self) -> None:
        self.vocab = {b"ab": 1, b"abc": 2, b"b": 3, b"ba": 4,
                      b"c": 5, b"a": 6, b"ca": 7}
        self.trie = wlb.WordTrie(self.vocab)

    def test_structure(self) -> None:
        trie = self.trie
        self.assertEqual(len(trie), 7)
        self.assertEqual(trie.node_count, 8)
        self.assertEqual(trie.prefix_node(b"ab"), 2)
        self.assertEqual(trie.node_prefix(2), b"ab")
        self.assertEqual(trie.node_prefix(3), b"abc")
        self.assertEqual(trie.prefix_node(b"zz"), None)
        self.assertEqual(trie.children_of(0), [(97, 1), (98, 4), (99, 6)])
        self.assertEqual(trie.word_ids[2], 1)
        self.assertEqual(trie.words(), sorted(self.vocab.items(), key=lambda x: x[1]))

    def test_skips(self) -> None:
        trie = wlb.WordTrie({b"": 1, b"x" * 30: 2, b"ok": 0, b"good": 3})
        self.assertEqual(len(trie), 1)
        self.assertEqual(trie.skipped["empty"], 1)
        self.assertEqual(trie.skipped["too_long"], 1)
        self.assertEqual(trie.skipped["unk"], 1)

    def test_duplicate_id_rejected(self) -> None:
        with self.assertRaises(ValueError):
            wlb.WordTrie({b"a": 1, b"b": 1})
        with self.assertRaises(ValueError):
            wlb.WordTrie({b"a": 5000})

    def test_score_prefixes_uniform(self) -> None:
        logp = [-math.log(256.0)] * 256
        scores = self.trie.score_prefixes(logp, top_k=None)
        self.assertEqual(len(scores), 7)
        for score in scores:
            self.assertAlmostEqual(
                score.score, -score.depth * math.log(256.0), places=12)
            self.assertAlmostEqual(
                score.mean_logprob, -math.log(256.0), places=12)
        # Shorter words rank first under raw, unnormalized scores; ties break
        # on word id (b=3, c=5, a=6).
        self.assertEqual([item.prefix for item in scores[:3]],
                         [b"b", b"c", b"a"])

    def test_score_prefixes_includes_nonterminals(self) -> None:
        trie = wlb.WordTrie({b"abc": 1, b"b": 2})
        logp = [-math.log(256.0)] * 256
        scores = trie.score_prefixes(logp, top_k=None, terminal_only=False)
        prefixes = {item.prefix: item for item in scores}
        self.assertIn(b"a", prefixes)
        self.assertIn(b"ab", prefixes)
        self.assertEqual(prefixes[b"a"].word_id, 0)
        self.assertAlmostEqual(prefixes[b"a"].score, -math.log(256.0))
        self.assertAlmostEqual(prefixes[b"ab"].score, -2 * math.log(256.0))

    def test_score_prefixes_normalize(self) -> None:
        # Mass 2 at the boundary: normalization must shift each byte by
        # -log(2), i.e. a word of length d by -d*log(2).
        logp = [-math.log(256.0) - math.log(2.0)] * 256
        raw = self.trie.score_prefixes(logp, top_k=None)
        normalized = self.trie.score_prefixes(logp, top_k=None, normalize=True)
        for left, right in zip(raw, normalized):
            self.assertAlmostEqual(
                right.score, left.score + left.depth * math.log(2.0), places=12)

    def test_score_prefixes_rejects_bad_vector(self) -> None:
        with self.assertRaises(ValueError):
            self.trie.score_prefixes([0.0] * 255)
        with self.assertRaises(ValueError):
            self.trie.score_prefixes([math.nan] * 256)


class TestBeamSearch(unittest.TestCase):
    def setUp(self) -> None:
        self.vocab = {b"ab": 1, b"abc": 2, b"b": 3, b"ba": 4,
                      b"c": 5, b"a": 6, b"ca": 7}
        self.trie = wlb.WordTrie(self.vocab)
        self.base = b"hello "

    def test_exhaustive_scores_are_exact(self) -> None:
        result = self.trie.beam_search(
            FakeScorer(), self.base, beam_width=None, top_k=100,
            max_edges=None)
        expected = sorted(
            ((brute_force(self.base, word), word_id, word)
             for word, word_id in self.vocab.items()),
            key=lambda item: (-item[0], item[1]))
        self.assertEqual([item.word_id for item in result.candidates],
                         [item[1] for item in expected])
        for candidate, (score, _word_id, word) in zip(
                result.candidates, expected):
            self.assertEqual(candidate.word, word)
            self.assertAlmostEqual(candidate.score, score, places=9)
            self.assertAlmostEqual(
                candidate.mean_logprob, score / len(word), places=9)
        self.assertEqual(result.edges_scored, self.trie.node_count - 1)
        self.assertFalse(result.truncated)

    def test_finite_beam_scores_are_still_exact(self) -> None:
        result = self.trie.beam_search(
            FakeScorer(), self.base, beam_width=2, top_k=100)
        self.assertTrue(result.candidates)
        for candidate in result.candidates:
            self.assertAlmostEqual(
                candidate.score, brute_force(self.base, candidate.word),
                places=9)

    def test_causal_prefix_context(self) -> None:
        trie = wlb.WordTrie({b"abc": 1, b"abcd": 2})
        scorer = FakeScorer()
        base = b"pre "
        result = trie.beam_search(
            scorer, base, beam_width=None, top_k=10)
        self.assertEqual(len(result.candidates), 2)
        found_last_byte = False
        for context, children in scorer.calls:
            self.assertTrue(context.startswith(base))
            if context == base + b"ab" and ord("c") in children:
                found_last_byte = True
            # The full target word must never be used to score its own last
            # byte.  It may appear when scoring a *longer* candidate's next
            # byte ("abc" -> "d"), which is legitimate.
            if context == base + b"abc" and ord("c") in children:
                self.fail("target word bytes were used as input for its own byte")
        self.assertTrue(found_last_byte)

    def test_max_edges_truncates(self) -> None:
        result = self.trie.beam_search(
            FakeScorer(), self.base, beam_width=None, top_k=10,
            max_edges=1)
        self.assertTrue(result.truncated)
        self.assertEqual(result.stop_reason, "max_edges")
        self.assertEqual(result.candidates, [])
        self.assertEqual(result.edges_scored, 0)

    def test_max_depth_marks_truncated(self) -> None:
        result = self.trie.beam_search(
            FakeScorer(), self.base, beam_width=None, top_k=100,
            max_depth=1)
        self.assertTrue(result.truncated)
        self.assertEqual(result.stop_reason, "max_depth")
        self.assertTrue(result.candidates)
        self.assertTrue(all(len(item.word) == 1
                            for item in result.candidates))

    def test_plain_callable_scorer(self) -> None:
        def scorer(pairs):
            return [[fake_logp(context, child) for child in children]
                    for context, children in pairs]
        result = self.trie.beam_search(
            scorer, self.base, beam_width=None, top_k=10)
        self.assertEqual(len(result.candidates), len(self.vocab))

    def test_invalid_beam_width(self) -> None:
        with self.assertRaises(ValueError):
            self.trie.beam_search(FakeScorer(), self.base, beam_width=0)

    def test_lexicon_beam_with_custom_scorer(self) -> None:
        results = wlb.lexicon_beam(
            None, [self.base], self.trie, mode="prefix",
            scorer=FakeScorer(), top_k=10, beam_width=None)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].mode, "prefix")
        self.assertIsNone(results[0].fallback_reason)
        self.assertEqual(len(results[0].candidates), len(self.vocab))
        for candidate in results[0].candidates:
            self.assertAlmostEqual(
                candidate.score, brute_force(self.base, candidate.word),
                places=9)

    def test_lexicon_beam_requires_model_or_scorer(self) -> None:
        with self.assertRaises(ValueError):
            wlb.lexicon_beam(None, [self.base], self.trie, mode="teacher")


class TestHelpers(unittest.TestCase):
    def test_word_spans(self) -> None:
        self.assertEqual(
            wlb.word_spans(b"  ab\tc\r\nd "),
            [(2, 4, b"ab"), (5, 6, b"c"), (8, 9, b"d")])
        self.assertEqual(wlb.word_spans(b""), [])
        self.assertEqual(wlb.word_spans(b"abc"), [(0, 3, b"abc")])

    def test_evenly_spaced(self) -> None:
        self.assertEqual(wlb._evenly_spaced(list(range(10)), 3), [0, 4, 9])
        self.assertEqual(wlb._evenly_spaced(list(range(3)), 5), [0, 1, 2])
        self.assertEqual(wlb._evenly_spaced([], 3), [])
        self.assertEqual(wlb._evenly_spaced([1, 2, 3], 1), [2])

    def test_stats_delta(self) -> None:
        before = {"model_calls": 1, "forward_rows": 4,
                  "context_window": 256, "teacher_forced": False}
        after = {"model_calls": 3, "forward_rows": 10,
                 "context_window": 256, "teacher_forced": False,
                 "mass_error": 1.0}
        delta = wlb._stats_delta(before, after)
        self.assertEqual(delta["model_calls"], 2)
        self.assertEqual(delta["forward_rows"], 6)
        self.assertEqual(delta["context_window"], 256)
        self.assertIs(delta["teacher_forced"], False)
        self.assertEqual(delta["mass_error"], 1.0)

    def test_dev_bin_round_trip(self) -> None:
        payload = (struct.pack("<II", 2, 0)
                   + struct.pack("<I", 3) + b"abc"
                   + struct.pack("<I", 4) + b"defg")
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "dev.bin"
            path.write_bytes(payload)
            excerpts, skip, fmt, raw = wlb.read_dev_bin(path)
        self.assertEqual(fmt, "nanro")
        self.assertEqual(skip, 0)
        self.assertEqual(excerpts, [b"abc", b"defg"])
        self.assertEqual(raw, payload)

    def test_dev_bin_raw_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "raw.bin"
            path.write_bytes(b"hello world")
            excerpts, skip, fmt, _raw = wlb.read_dev_bin(path)
        self.assertEqual(fmt, "raw")
        self.assertEqual(skip, 0)
        self.assertEqual(excerpts, [b"hello world"])

    def test_manifest_check(self) -> None:
        payload = b"abc"
        manifest = {
            "dev_sha256": hashlib.sha256(payload).hexdigest(),
            "books": 1,
            "skip": 0,
            "selected_books": [{"relative_path": "x.txt"}],
        }
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "manifest.json"
            path.write_text(json.dumps(manifest))
            loaded, notes = wlb._manifest_check(path, [payload], 0, payload)
            self.assertEqual(loaded["books"], 1)
            self.assertEqual(notes, [])
            with self.assertRaises(ValueError):
                wlb._manifest_check(path, [b"other"], 0, b"other")

    def test_vocab_json_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "vocab.json"
            path.write_text(json.dumps({"hello": 1, "world": 2}))
            self.assertEqual(wlb.load_vocab_json(path),
                             {b"hello": 1, b"world": 2})
            path.write_text(json.dumps({"vocab": ["hello", "world"]}))
            self.assertEqual(wlb.load_vocab_json(path),
                             {b"hello": 1, b"world": 2})
            path.write_text(json.dumps(
                {"words": [{"word": "hello", "id": 3},
                           {"token": "world", "index": 4}]}))
            self.assertEqual(wlb.load_vocab_json(path),
                             {b"hello": 3, b"world": 4})
            path.write_text(json.dumps({"<unk>": 0, "x": 1}))
            self.assertEqual(wlb.load_vocab_json(path), {b"x": 1})
            path.write_text(json.dumps({"hex:00ff": 1, "y": 2}))
            self.assertEqual(wlb.load_vocab_json(path),
                             {b"\x00\xff": 1, b"y": 2})

    def test_vocab_json_rejects_duplicate_id(self) -> None:
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "vocab.json"
            path.write_text(json.dumps({"a": 1, "b": 1}))
            with self.assertRaises(ValueError):
                wlb.load_vocab_json(path)
            path.write_text(json.dumps({"a": 5000}))
            with self.assertRaises(ValueError):
                wlb.load_vocab_json(path, max_vocab=4096)

    def test_recall_and_unique_hits(self) -> None:
        rankings = [[1, 2], [3, 4], []]
        targets = [2, 4, 5]
        self.assertAlmostEqual(wlb.recall_at_k(rankings, targets, 1), 0.0)
        self.assertAlmostEqual(wlb.recall_at_k(rankings, targets, 2), 2 / 3)
        self.assertEqual(wlb.unique_hits(rankings, targets, 2), 2)
        self.assertAlmostEqual(
            wlb.recall_at_k([[0], [2]], [0, 2], 1, in_vocab_only=True), 1.0)

    def test_enumerate_boundaries(self) -> None:
        excerpt = b"x" * 40 + b" aa bb"
        vocab = {b"aa": 1, b"bb": 2}
        boundaries, words, ids = wlb.enumerate_boundaries(
            [excerpt], vocab, min_context=32, per_book=0, max_boundaries=10)
        # words_by_book/ids_by_book deliberately cover every word, including
        # the pre-context run, so PPM can reconstruct any boundary prefix.
        self.assertEqual(words, [[b"x" * 40, b"aa", b"bb"]])
        self.assertEqual(ids, [[0, 1, 2]])
        self.assertEqual([(item.position, item.word, item.target_id)
                          for item in boundaries],
                         [(41, b"aa", 1), (44, b"bb", 2)])

    def test_ppm(self) -> None:
        ids = [1, 2, 1, 2, 2, 2]
        counts = wlb.build_ppm(ids, max_order=2)
        ranking = wlb.ppm_ranking(
            [1, 2], counts, [2, 1], max_order=2, top_k=5)
        self.assertEqual(ranking[:2], [1, 2])
        # Unknown context falls back to the unigram list.
        self.assertEqual(
            wlb.ppm_ranking([9, 9], counts, [2, 1], max_order=2, top_k=5),
            [2, 1])

    def test_json_safe(self) -> None:
        safe = wlb._json_safe({"a": math.nan, "b": [math.inf, 1.0]})
        self.assertEqual(safe, {"a": None, "b": [None, 1.0]})

    def test_cached_baseline(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "cache.npz"
            head = np.zeros((2, 4096), dtype=np.float16)
            head[0, 7] = 10
            head[1, 9] = 10
            ppm = np.full((2, 100), -1, dtype=np.int32)
            ppm[0, 0] = 7
            ppm[1, 0] = 9
            np.savez_compressed(
                path, head=head, ppm=ppm,
                target=np.array([7, 9], dtype=np.int32),
                book=np.array([0, 0], dtype=np.int16),
                pos=np.array([41, 44], dtype=np.int32))
            boundaries = [
                wlb.Boundary(book=0, word_index=0, position=41,
                             word=b"aa", target_id=7),
                wlb.Boundary(book=0, word_index=1, position=44,
                             word=b"bb", target_id=9),
            ]
            result = wlb.load_cached_baseline(path, boundaries, kmax=5)
        self.assertTrue(result["available"])
        self.assertEqual(result["matched"], 2)
        self.assertEqual(result["head"][0][0], 7)
        self.assertEqual(result["head"][1][0], 9)
        self.assertEqual(result["ppm"][0][0], 7)

    def test_cached_baseline_rejects_vocab_mismatch(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "cache.npz"
            np.savez_compressed(
                path, head=np.zeros((1, 4096), dtype=np.float16),
                target=np.array([99], dtype=np.int32),
                book=np.array([0], dtype=np.int16),
                pos=np.array([41], dtype=np.int32))
            boundaries = [wlb.Boundary(
                book=0, word_index=0, position=41, word=b"aa", target_id=7)]
            result = wlb.load_cached_baseline(path, boundaries, kmax=5)
        self.assertFalse(result["available"])
        self.assertIn("disagree", result["reason"])


class TestExactScorerRunner(unittest.TestCase):
    """The exact scorer's batching/window logic with an injected runner."""

    def test_exact_prefix_scorer_batches_and_windows(self) -> None:
        calls: list[tuple[int, int]] = []

        def runner(previous_rows, targets):
            calls.append((len(previous_rows), len(previous_rows[0])))
            return ([fake_logp(row, int(target))
                     for row, target in zip(previous_rows, targets)], 0)

        scorer = wlb.ExactPrefixScorer(
            runner=runner, context_window=32, batch_size=2)
        first_context = b"x" * 40 + b"a"
        second_context = b"y" * 40 + b"a"
        result = scorer.score_pairs([
            (first_context, [ord("b"), ord("c")]),
            (second_context, [ord("d")]),
        ])
        self.assertEqual([len(row) for row in result], [2, 1])
        first_window = first_context[-32:]
        second_window = second_context[-32:]
        self.assertAlmostEqual(
            result[0][0], fake_logp(first_window, ord("b")), places=12)
        self.assertAlmostEqual(
            result[0][1], fake_logp(first_window, ord("c")), places=12)
        self.assertAlmostEqual(
            result[1][0], fake_logp(second_window, ord("d")), places=12)
        self.assertEqual(calls, [(2, 32), (1, 32)])
        self.assertEqual(scorer.model_calls, 2)
        self.assertEqual(scorer.forward_rows, 3)
        self.assertEqual(scorer.edges_scored, 3)
        scorer.close()

    def test_pads_short_context_to_minimum(self) -> None:
        seen: list[bytes] = []

        def runner(previous_rows, targets):
            seen.extend(previous_rows)
            return ([fake_logp(row, int(target))
                     for row, target in zip(previous_rows, targets)], 0)

        scorer = wlb.ExactPrefixScorer(
            runner=runner, context_window=32, batch_size=8)
        result = scorer.score_pairs([(b"abc", [ord("d")])])
        self.assertEqual(len(seen[0]), 32)
        self.assertEqual(seen[0][-3:], b"abc")
        self.assertAlmostEqual(
            result[0][0], fake_logp(seen[0], ord("d")), places=12)

    def test_boundary_scorer_with_runner(self) -> None:
        def runner(previous_rows, targets):
            return ([fake_logp(row, int(target))
                     for row, target in zip(previous_rows, targets)], 0)

        base = b"z" * 40
        boundary = wlb.BoundaryByteScorer(
            runner=runner, base_context=base, context_window=32,
            batch_size=64)
        self.assertEqual(len(boundary.logp), 256)
        window = base[-32:]
        for child in (0, 7, 65, 255):
            self.assertAlmostEqual(
                boundary.logp[child], fake_logp(window, child), places=12)
        # fake_logp is a proper normalized distribution.
        self.assertAlmostEqual(boundary.mass_error, 0.0, places=9)
        boundary.close()

    def test_lexicon_beam_with_runner_scorer(self) -> None:
        def runner(previous_rows, targets):
            return ([fake_logp(row, int(target))
                     for row, target in zip(previous_rows, targets)], 0)

        trie = wlb.WordTrie({b"ab": 1, b"b": 2, b"c": 3})
        scorer = wlb.ExactPrefixScorer(
            runner=runner, context_window=32, batch_size=8)
        results = wlb.lexicon_beam(
            None, [b"q" * 40], trie, mode="prefix", scorer=scorer,
            top_k=10, beam_width=None)
        self.assertEqual(results[0].mode, "prefix")

        def windowed_score(word: bytes) -> float:
            total = 0.0
            for index, byte in enumerate(word):
                context = (b"q" * 40 + word[:index])[-32:]
                total += fake_logp(context, byte)
            return total

        expected = sorted(
            trie.words(), key=lambda pair: (-windowed_score(pair[0]), pair[1]))
        self.assertEqual(
            [candidate.word for candidate in results[0].candidates],
            [word for word, _word_id in expected])
        for candidate in results[0].candidates:
            self.assertAlmostEqual(
                candidate.score, windowed_score(candidate.word), places=9)
        self.assertEqual(results[0].cost["forward_rows"], 4.0)
        scorer.close()

    def test_lexicon_beam_teacher_with_runner(self) -> None:
        def runner(previous_rows, targets):
            return ([fake_logp(row, int(target))
                     for row, target in zip(previous_rows, targets)], 0)

        trie = wlb.WordTrie({b"ab": 1, b"b": 2, b"c": 3})
        results = wlb.lexicon_beam(
            None, [b"q" * 40], trie, mode="teacher", runner=runner,
            top_k=10, batch_size=64, context_window=32)
        self.assertEqual(results[0].mode, "teacher")
        window = (b"q" * 40)[-32:]
        expected = trie.score_prefixes(
            [fake_logp(window, child) for child in range(256)], top_k=10)
        self.assertEqual(
            [candidate.word_id for candidate in results[0].candidates],
            [score.word_id for score in expected])
        for candidate, score in zip(results[0].candidates, expected):
            self.assertAlmostEqual(candidate.score, score.score, places=12)
        self.assertEqual(results[0].cost["forward_rows"], 256.0)
        self.assertEqual(results[0].cost["trie_edges"], trie.node_count - 1)
        self.assertFalse(results[0].truncated)

    def test_auto_falls_back_to_teacher_on_prefix_error(self) -> None:
        def runner(previous_rows, targets):
            return ([fake_logp(row, int(target))
                     for row, target in zip(previous_rows, targets)], 0)

        class BadScorer:
            def score_pairs(self, _pairs):
                raise RuntimeError("intentional prefix failure")

        trie = wlb.WordTrie({b"ab": 1, b"b": 2})
        results = wlb.lexicon_beam(
            object(), [b"q" * 40], trie, mode="auto", scorer=BadScorer(),
            runner=runner, top_k=10, batch_size=64, context_window=32)
        self.assertEqual(results[0].mode, "teacher")
        self.assertIn("intentional prefix failure", results[0].fallback_reason)
        self.assertTrue(results[0].candidates)


class TestReport(unittest.TestCase):
    """Exercise the CLI report builder without importing torch."""

    def test_build_report_with_stubs(self) -> None:
        vocabulary = {b"aa": 1, b"bb": 2}
        trie = wlb.WordTrie(vocabulary)
        excerpts = [b"x" * 40 + b" aa bb"]
        boundaries, words, ids = wlb.enumerate_boundaries(
            excerpts, vocabulary, min_context=32, per_book=0,
            max_boundaries=10)
        args = argparse.Namespace(
            checkpoint=None, dev_bin=Path("dev.bin"), dev_manifest=None,
            vocab_json=Path("vocab.json"), topk=100, context_window=32,
            min_context=32, per_book=0, max_boundaries=10, exact="prefix",
            beam=2, max_edges=1000, batch_size=4, head_batch=2,
            max_word_bytes=24, max_vocab=4096,
            details=False, progress=False, train_bin=None,
            ppm_max_order=8, ppm_max_train_words=0, baseline_cache=None)

        original_head = wlb.word_head_topk
        original_beam = wlb.lexicon_beam

        def fake_head(_model, _windows, _kmax, **_kwargs):
            return [[2, 1], [1, 2]]

        def fake_beam(_model, histories, _trie, **_kwargs):
            return [wlb.LexiconResult(
                candidates=[wlb.WordCandidate(1, b"aa", -1.0, 2, -0.5)],
                requested_mode="prefix", mode="prefix",
                context_bytes=len(history), truncated=False,
                stop_reason="complete", fallback_reason=None,
                cost={"seconds": 0.1, "model_calls": 1,
                      "forward_rows": 2, "edges_scored": 2},
                notes=[]) for history in histories]

        try:
            wlb.word_head_topk = fake_head
            wlb.lexicon_beam = fake_beam
            report = wlb._build_report(
                args, boundaries=boundaries, words_by_book=words,
                ids_by_book=ids, trie=trie, vocabulary=vocabulary,
                manifest=None, dev_format="raw", skip=0,
                excerpts=excerpts, model=object(), device=None, notes=[])
        finally:
            wlb.word_head_topk = original_head
            wlb.lexicon_beam = original_beam

        self.assertEqual(report["boundaries"]["units"], 2)
        self.assertEqual(report["mode"]["used"], ["prefix"])
        # targets are [1, 2]; the fake beam returns only id 1 for both.
        self.assertAlmostEqual(report["recall"]["beam"]["recall@5"], 0.5)
        # the fake head finds both targets.
        self.assertAlmostEqual(report["recall"]["head"]["recall@5"], 1.0)
        self.assertAlmostEqual(
            report["in_vocab_recall"]["beam"]["recall@5"], 0.5)
        self.assertEqual(report["cost"]["forward_rows"], 4)


try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class TestTorchScorers(unittest.TestCase):
    """Plumbing tests; not executed in the Termux environment (no torch)."""

    def test_uniform_scorer_and_boundary_mass(self) -> None:
        class UniformModel(torch.nn.Module):
            def forward(self, previous, target):
                bits = ((target[:, None] >> (
                    7 - torch.arange(8, device=target.device)[None, :])) & 1
                ).float()
                return torch.zeros_like(bits), bits

        model = UniformModel()
        scorer = wlb.ExactPrefixScorer(
            model, context_window=32, batch_size=2)
        result = scorer.score_pairs([
            (b"a" * 40, [65, 66]),
            (b"b" * 40, [67]),
        ])
        self.assertEqual([len(row) for row in result], [2, 1])
        expected = 8 * math.log(0.5)
        for row in result:
            for value in row:
                self.assertAlmostEqual(value, expected, places=6)
        self.assertEqual(scorer.forward_rows, 3)
        scorer.close()

        boundary = wlb.BoundaryByteScorer(
            model, b"z" * 40, context_window=32)
        self.assertEqual(len(boundary.logp), 256)
        self.assertAlmostEqual(boundary.mass_error, 0.0, places=5)
        boundary.close()

    def test_lexicon_beam_prefix_with_torch(self) -> None:
        class UniformModel(torch.nn.Module):
            def forward(self, previous, target):
                bits = ((target[:, None] >> (
                    7 - torch.arange(8, device=target.device)[None, :])) & 1
                ).float()
                return torch.zeros_like(bits), bits

        trie = wlb.WordTrie({b"ab": 1, b"b": 2})
        results = wlb.lexicon_beam(
            UniformModel(), [b"z" * 40], trie, mode="prefix",
            top_k=5, beam_width=2, context_window=32)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].mode, "prefix")
        self.assertEqual(len(results[0].candidates), 2)


if __name__ == "__main__":
    unittest.main()
