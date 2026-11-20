"""Small unit tests for hierarchical span construction."""
from __future__ import annotations

import unittest

from terminal_rsi.hierarchical_metrics import (
    _bpe_spans, _safe_eval_batch, _sentence_spans, _word_spans,
)


class HierarchicalSpanTests(unittest.TestCase):
    def test_word_spans(self):
        self.assertEqual(_word_spans(b"hi there"), [(0, 2), (3, 8)])
        self.assertEqual(_word_spans(b"  hi\n"), [(2, 4)])

    def test_sentence_spans(self):
        self.assertEqual(_sentence_spans(b"Hi. Bye!"), [(0, 3), (4, 8)])

    def test_candidate_batch_is_capped(self):
        self.assertEqual(_safe_eval_batch(128), 8)
        self.assertEqual(_safe_eval_batch(10000), 8)
        self.assertEqual(_safe_eval_batch(1), 1)
        self.assertEqual(_safe_eval_batch(0), 1)

    def test_empty_bpe_returns_bytes(self):
        self.assertEqual(_bpe_spans(b"ab", {}), [(0, 1), (1, 2)])

    def test_bpe_merge_collapses_pair(self):
        spans = _bpe_spans(b"ab", {(ord("a"), ord("b")): 0})
        self.assertEqual(spans, [(0, 2)])


if __name__ == "__main__":
    unittest.main()
