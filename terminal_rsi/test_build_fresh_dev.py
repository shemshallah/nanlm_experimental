"""CPU-only acceptance for fresh Terminal-RSI NaNROUTER DEV creation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import torch

from terminal_rsi.build_fresh_dev import build, validate_fresh_dev
from terminal_rsi.legacy_v103_reference import heldout_windows, PARITY_CONTEXT


def english_book(book_id: int) -> bytes:
    # Use distinct natural-English sentences; 64-byte shingles contain ID/line
    # information and do not accidentally match across books.
    sentences = [
        (f"The traveler in book {book_id} had a dream at chapter {i:05d}. "
         f"She walked to the forest and the river was quiet. \n")
        for i in range(180)
    ]
    return "".join(sentences).encode()


class FreshNaNROUTERDevTest(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory(prefix="terminal_rsi_new_dev_")
        self.addCleanup(self.t.cleanup)
        self.root = Path(self.t.name)
        self.corpus = self.root / "corpus" / "batch-0004"
        self.corpus.mkdir(parents=True)
        for bid in range(1000, 1022):
            (self.corpus / f"{bid}--unknown--story.txt").write_bytes(english_book(bid))
        # Russian book by Cyrillic content, even with English filename.
        (self.corpus / "2010--unknown--story.txt").write_bytes(
            english_book(2010) + "Глава первая. Это русский текст.".encode())
        # Russian-language metadata on an all-Latin text.
        (self.corpus / "2011--unknown--story.txt").write_bytes(english_book(2011))
        # Russian-language writer transliterated, with English text.
        (self.corpus / "2012--Chekhov--english-translation.txt").write_bytes(english_book(2012))
        (self.corpus / "SOURCES.jsonl").write_text(
            json.dumps({"gutenberg_id": 2011, "language": "ru"}) + "\n")
        self.train = self.root / "train_10000000.bin"
        self.train.write_bytes(english_book(1000) + b"unrelated original training source")
        self.train_ids = self.root / "train_ids.json"
        self.train_ids.write_text(json.dumps({"book_ids": [1000, 1001]}))
        self.test_ids = self.root / "sealed_test_ids.txt"
        self.test_ids.write_text("1002\n")
        self.output = self.root / "dev"

    def build_data(self, **changes):
        args = dict(corpus_dir=self.corpus.parent, train_stream=self.train,
                    output_dir=self.output, book_count=18,
                    positions_per_book=80, skip=64, seed=103,
                    train_book_ids=self.train_ids,
                    sealed_test_book_ids=self.test_ids)
        args.update(changes)
        return build(**args)

    def test_new_eighteen_book_dev_excludes_russian_and_train_and_test_ids(self):
        manifest = self.build_data()
        self.assertEqual(manifest["books"], 18)
        self.assertEqual(manifest["scored_positions"], 1440)
        self.assertTrue(manifest["id_level_train_disjoint_proven"])
        self.assertTrue(manifest["id_level_test_disjoint_proven"])
        self.assertGreaterEqual(manifest["excluded_reason_counts"]["cyrillic_text"], 1)
        self.assertGreaterEqual(manifest["excluded_reason_counts"]["russian_language_metadata"], 1)
        self.assertGreaterEqual(manifest["excluded_reason_counts"]["russian_author_conservative_exclusion"], 1)
        excluded = {"1000", "1001", "1002", "2010", "2011", "2012"}
        self.assertFalse(excluded & {e["gutenberg_id"] for e in manifest["selected_books"]})
        valid = validate_fresh_dev(
            self.output / "win_dev.bin", self.output / "win_dev.manifest.json",
            train_sha256=hashlib.sha256(self.train.read_bytes()).hexdigest())
        self.assertEqual(valid["dev_sha256"], manifest["dev_sha256"])
        windows, digest = heldout_windows(
            str(self.output / "win_dev.bin"), torch.device("cpu"),
            history=PARITY_CONTEXT)
        self.assertEqual(tuple(windows.shape), (18, 80, 33))
        self.assertEqual(digest, manifest["dev_sha256"])
        # Exact reproducibility, no overwrites or restored historic DEV.
        self.assertEqual(self.build_data()["dev_sha256"], digest)

    def test_sha_train_identity_is_binding(self):
        self.build_data()
        with self.assertRaisesRegex(ValueError, "DIFFERENT TRAIN"):
            validate_fresh_dev(self.output / "win_dev.bin",
                               self.output / "win_dev.manifest.json",
                               train_sha256="0" * 64)

    def test_corrupt_dev_rejected(self):
        self.build_data()
        raw = bytearray((self.output / "win_dev.bin").read_bytes())
        raw[-1] ^= 1
        (self.output / "win_dev.bin").write_bytes(bytes(raw))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            validate_fresh_dev(self.output / "win_dev.bin",
                               self.output / "win_dev.manifest.json")

    def test_never_overwrite_changed_dev(self):
        self.build_data()
        with self.assertRaises(FileExistsError):
            self.build_data(seed=104)

    def test_too_many_books_fails_without_artifacts(self):
        with self.assertRaisesRegex(ValueError, "eligible English"):
            self.build_data(book_count=300)
        self.assertFalse((self.output / "win_dev.bin").exists())

    def test_provenance_not_overclaimed_without_id_lists(self):
        manifest = self.build_data(book_count=18,
                                   train_book_ids=None,
                                   sealed_test_book_ids=None)
        self.assertFalse(manifest["id_level_train_disjoint_proven"])
        self.assertFalse(manifest["id_level_test_disjoint_proven"])
        self.assertTrue(manifest["sealed_test_opened"] is False)


if __name__ == "__main__":
    unittest.main()
