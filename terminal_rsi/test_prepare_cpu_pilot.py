"""CPU-only verified public-corpus pilot preparation with strict book IDs."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from terminal_rsi.test_build_fresh_dev import english_book
from terminal_rsi.prepare_cpu_pilot import build_cpu_pilot
from terminal_rsi.build_fresh_dev import validate_fresh_dev


class FreshRealPilotCorpusTests(unittest.TestCase):
    def test_fresh_train_and_dev_are_distinct_verified_english_books(self):
        with tempfile.TemporaryDirectory(prefix="terminal_rsi_corpus_") as tmp:
            root = Path(tmp)
            corpus = root / "batch-0004"
            corpus.mkdir()
            for bid in range(2000, 2030):
                (corpus / f"{bid}--unknown--english.txt").write_bytes(english_book(bid))
            (corpus / "2200--unknown--russian.txt").write_bytes(
                english_book(2200) + "запрещённый русский текст".encode())
            output = root / "prepared"
            receipt = build_cpu_pilot(
                corpus_dir=root, output_dir=output,
                train_bytes=8192, train_chunk_bytes=4096,
                dev_books=18, dev_positions=80, seed=103, min_remaining_books=18)
            self.assertEqual(receipt["train_bytes"], 8192)
            self.assertEqual(receipt["dev_books"], 18)
            self.assertEqual(receipt["dev_scored_bytes"], 1440)
            self.assertTrue(receipt["book_disjoint_train_dev_verified_by_gutenberg_ids"])
            self.assertEqual(receipt["source_rejection_counts"]["cyrillic_text"], 1)
            manifest = validate_fresh_dev(
                output / "win_dev.bin", output / "win_dev.manifest.json",
                train_sha256=hashlib.sha256((output / "train_cpu_pilot.bin").read_bytes()).hexdigest())
            self.assertEqual(manifest["books"], 18)
            self.assertTrue(manifest["id_level_train_disjoint_proven"])
            self.assertEqual(receipt["dev_sha256"], manifest["dev_sha256"])
            self.assertEqual(build_cpu_pilot(
                corpus_dir=root, output_dir=output,
                train_bytes=8192, train_chunk_bytes=4096,
                dev_books=18, dev_positions=80, seed=103, min_remaining_books=18)["dev_sha256"],
                receipt["dev_sha256"])


if __name__ == "__main__":
    unittest.main()
