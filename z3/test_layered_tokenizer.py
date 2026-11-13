#!/usr/bin/env python3
"""Tests for the six-level tokenizer."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layered_tokenizer import (
    BitLayer,
    ByteLayer,
    LayeredTokenizer,
    SentenceLayer,
    WordLayer,
    _char_byte_offsets,
    load_artifact,
    write_artifact,
)


TEXT = (
    "Hello world! This is a layered tokenizer. "
    "It keeps naïve café bytes, 你好, and punctuation aligned. "
    "A second paragraph has another sentence?\nYes.\n"
)


class LayeredTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = LayeredTokenizer(
            prefer_native=False,
            word_vocab_size=0,
            sentence_max_chars=64,
        )

    def test_bit_layer_lossless_both_orders(self):
        data = b"\x00\x01\x7f\x80\xffhello"
        for msb in (True, False):
            layer = BitLayer(msb_first=msb)
            bits = layer.encode(data)
            self.assertEqual(len(bits), len(data) * 8)
            self.assertEqual(layer.decode(bits), data)
            self.assertEqual(layer.pack(bits), data)
        with self.assertRaises(ValueError):
            BitLayer().decode([1, 0, 1])

    def test_byte_layer_unicode(self):
        layer = ByteLayer()
        raw = TEXT.encode("utf-8")
        self.assertEqual(layer.encode(TEXT), raw)
        self.assertEqual(layer.decode(raw), TEXT)

    def test_subword_roundtrip_and_alignment(self):
        encoded = self.tokenizer.encode(TEXT, include_bits=True)
        self.assertEqual(
            self.tokenizer.subword.decode(encoded.subword.ids), TEXT
        )
        self.assertEqual(encoded.subword.n_bytes, len(TEXT.encode("utf-8")))
        self.assertEqual(len(encoded.bits), len(encoded.bytes) * 8)
        cursor = 0
        for start, end in encoded.subword.spans:
            self.assertEqual(start, cursor)
            self.assertLess(start, end)
            cursor = end
        self.assertEqual(cursor, len(encoded.bytes))
        self.assertEqual(len(encoded.token_ids), len(encoded.subword.ids) + 2)
        self.assertEqual(encoded.token_ids[0], self.tokenizer.doc_id)
        self.assertEqual(encoded.token_ids[-1], self.tokenizer.eos_id)
        self.assertEqual(self.tokenizer.decode_token_ids(encoded.token_ids), TEXT)

    def test_words_and_sentences_tile_the_input(self):
        encoded = self.tokenizer.encode(TEXT)
        self.assertEqual(
            self.tokenizer.decode_words(encoded.word_units), TEXT
        )
        self.assertEqual(
            self.tokenizer.decode_sentences(TEXT, encoded.sentence_spans), TEXT
        )
        self.assertEqual(
            sum(end - start for start, end in encoded.sentence_spans),
            len(encoded.bytes),
        )
        self.assertGreater(len(encoded.word_units), 5)
        self.assertGreaterEqual(len(encoded.sentence_spans), 3)
        for unit in encoded.word_units:
            self.assertGreaterEqual(unit.start, 0)
            self.assertLessEqual(unit.end, len(encoded.bytes))
            self.assertGreaterEqual(unit.subword_start, 1)
            self.assertLessEqual(unit.subword_end, len(encoded.subword.ids) + 1)

    def test_word_tuning_and_fit(self):
        layer = WordLayer()
        size, detail = layer.tune(TEXT, [2, 8, 64], min_coverage=0.5)
        self.assertIn(size, (2, 8, 64))
        self.assertTrue(detail["gate"]["passed"])
        fitted = layer.fit(TEXT, size)
        self.assertGreater(fitted.vocab_size, 0)
        self.assertLessEqual(fitted.vocab_size, size)
        subword = self.tokenizer.subword.encode(TEXT)
        units = fitted.encode_with_offsets(
            TEXT, _char_byte_offsets(TEXT), subword.spans
        )
        self.assertEqual(
            "".join(unit.text for unit in units), TEXT
        )

    def test_sentence_tuning_and_long_split(self):
        text = "One short sentence. " + ("word " * 100) + "End."
        max_chars, detail = SentenceLayer.tune(
            text, [32, 64, 256], target_max_chars=64
        )
        self.assertIn(max_chars, (32, 64, 256))
        self.assertTrue(detail["gate"]["passed"])
        layer = SentenceLayer(max_chars=64)
        spans = layer.split(text)
        self.assertEqual("".join(text[s:e] for s, e in spans), text)
        self.assertLessEqual(max(e - s for s, e in spans), 64)

    def test_artifact_roundtrip(self):
        fitted = WordLayer().fit(TEXT, 32)
        tokenizer = LayeredTokenizer(
            prefer_native=False,
            word_vocab_size=fitted.vocab_size,
            word_vocab=fitted.vocab,
            sentence_max_chars=64,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "layered.json"
            write_artifact(tokenizer, path)
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(raw["layers"]["subword"]["vocab_size"], 64000)
            loaded = load_artifact(path)
            self.assertEqual(loaded.word.vocab, tokenizer.word.vocab)
            self.assertEqual(loaded.sentence.max_chars, 64)
            encoded = loaded.encode(TEXT)
            self.assertEqual(
                loaded.decode_token_ids(encoded.token_ids), TEXT
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
