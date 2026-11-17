#!/usr/bin/env python3
"""Tests for document framing and the domain-bin builder helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from text_documents import iter_documents, write_documents
from train_v3 import LayeredFlatTokenizer, load_text_ids


TOKENIZER_PATH = (
    Path(__file__).resolve().parent
    / "tokenizers" / "layered_v1" / "layered_tokenizer.json"
)


class TextDocumentTests(unittest.TestCase):
    def test_iter_documents_preserves_blank_line_split(self):
        with tempfile.TemporaryDirectory(prefix="z3_docs_") as tmp:
            path = Path(tmp) / "corpus.txt"
            path.write_text(
                "First document.\nline two.\n\n"
                "Second document.\n\n\n"
                "Third document.\n",
                encoding="utf-8",
            )
            documents = [text for text, _raw in iter_documents(path)]
            self.assertEqual(len(documents), 3)
            self.assertEqual(documents[0], "First document.\nline two.")
            self.assertEqual(documents[1], "Second document.")
            self.assertEqual(documents[2], "Third document.")

    def test_min_bytes_filters(self):
        with tempfile.TemporaryDirectory(prefix="z3_docs_") as tmp:
            path = Path(tmp) / "corpus.txt"
            path.write_text(
                "tiny\n\n" + ("A" * 200) + "\n\nlast\n",
                encoding="utf-8",
            )
            documents = [
                text
                for text, _raw in iter_documents(path, min_bytes=20)
            ]
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0], "A" * 200)

    def test_write_documents_roundtrip(self):
        with tempfile.TemporaryDirectory(prefix="z3_docs_") as tmp:
            path = Path(tmp) / "out.txt"
            write_documents(path, ["one", "two\nlines", "three"])
            documents = [text for text, _raw in iter_documents(path)]
            self.assertEqual(documents, ["one", "two\nlines", "three"])

    @unittest.skipUnless(
        TOKENIZER_PATH.is_file(), "layered tokenizer artifact not present"
    )
    def test_load_text_ids_document_framing(self):
        with tempfile.TemporaryDirectory(prefix="z3_docs_") as tmp:
            path = Path(tmp) / "corpus.txt"
            path.write_text("First doc.\n\nSecond doc.\n", encoding="utf-8")
            tokenizer = LayeredFlatTokenizer(TOKENIZER_PATH)
            ids = load_text_ids(
                [str(path)], tokenizer, 0, document_framing=True
            )
            self.assertGreater(len(ids), 2)
            self.assertEqual(int(ids[0]), int(tokenizer.layered.doc_id))
            self.assertIn(int(tokenizer.layered.eos_id), ids.tolist())


if __name__ == "__main__":
    unittest.main(verbosity=2)
