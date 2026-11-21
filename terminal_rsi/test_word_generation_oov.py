"""Word-level generation must spell UNK with the byte fallback, not stop."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from terminal_rsi.word_bridge import generate_words


class StubModel:
    def __init__(self):
        self._word_vocab = {b"hello": 1}

    def parameters(self):
        yield torch.zeros(1)

    def eval(self):
        return self


class WordGenerationOOVTests(unittest.TestCase):
    def test_unk_is_spelled_by_the_byte_fallback(self):
        model = StubModel()
        values = torch.zeros((1, 1))
        indices = torch.zeros((1, 1), dtype=torch.long)
        with patch("terminal_rsi.word_bridge.word_candidates",
                   return_value=(values, indices)), \
                patch("terminal_rsi.oov_fallback.spell_oov",
                      return_value=[("zephyr", -1.0), ("zephyrs", -2.0)]):
            result = generate_words(model, b"hello ", steps=1,
                                    device=torch.device("cpu"))
        self.assertEqual(result["words"], ["zephyr"])
        self.assertTrue(result["records"][0]["oov"])
        self.assertEqual(result["records"][0]["candidates"],
                         ["zephyr", "zephyrs"])

    def test_known_word_does_not_call_the_fallback(self):
        model = StubModel()
        values = torch.zeros((1, 1))
        indices = torch.ones((1, 1), dtype=torch.long)
        with patch("terminal_rsi.word_bridge.word_candidates",
                   return_value=(values, indices)), \
                patch("terminal_rsi.oov_fallback.spell_oov") as spell:
            result = generate_words(model, b"hello ", steps=1,
                                    device=torch.device("cpu"))
        spell.assert_not_called()
        self.assertEqual(result["words"], ["hello"])


if __name__ == "__main__":
    unittest.main()
