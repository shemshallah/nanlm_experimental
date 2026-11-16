"""R18 software tests use only synthetic bytes, never untouched TEST labels."""

import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from m3_r18 import locked_replication as m


class R18FrozenTests(unittest.TestCase):
    def test_manifest_is_precommitted_and_disjoint(self):
        manifest = json.loads(m.MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["seed"], m.SEED)
        self.assertEqual(manifest["warmup"], m.WARMUP)
        self.assertEqual(len(manifest["r17_test_sha256"]), m.TEST_BOOKS)
        self.assertEqual(len(manifest["r18_test_sha256"]), m.NEW_TEST_BOOKS)
        all_hashes = manifest["r17_test_sha256"] + manifest["r18_test_sha256"]
        self.assertEqual(len(set(all_hashes)), len(all_hashes))
        for fingerprint in all_hashes:
            self.assertRegex(fingerprint, r"\A[0-9a-f]{64}\Z")

    def test_rejects_new_test_mutation_even_when_original_prefix_matches(self):
        books = [f"fixture {i:02d} unique".encode() for i in range(52)]
        paths = [Path(f"fixture-{i:02d}.txt") for i in range(52)]
        hashes = [hashlib.sha256(b).hexdigest() for b in books]
        manifest = {
            "seed": m.SEED, "bytes_per_book": m.BYTES_PER_BOOK,
            "warmup": m.WARMUP,
            "r17_test_sha256": hashes[14:20],
            "r18_test_sha256": hashes[20:],
        }
        with patch.object(m, "verify_r16_prefix", return_value=hashes[:20]):
            self.assertEqual(m.verify_manifest(paths, books, manifest), hashes)
            books[31] += b" changed"
            with self.assertRaisesRegex(ValueError, "R18 new TEST identities changed"):
                m.verify_manifest(paths, books, manifest)
            books[31] = books[0]
            with self.assertRaises(ValueError):
                m.verify_manifest(paths, books, manifest)

    def test_streamed_aggregate_preserves_exact_oscillator_when_residual_zero(self):
        book = b"aa aaaaa"  # 8 bytes; four scored positions after warmup 4
        fake = {
            "base_logp": np.full((4, 256), -np.log(256), dtype=np.float64),
            "targets": np.array([32, 97, 97, 32], dtype=np.int64),
            "nonspace": np.array([False, True, True, False]),
            "unicode_chars": 4, "elapsed_s": 1.0,
            "features": {
                "uncoupled": np.zeros((4, 256, 5), dtype=np.float32),
                "spatial_wave": np.zeros((4, 256, 5), dtype=np.float32),
            },
        }
        with patch.object(m, "collect", return_value=fake) as collect:
            result = m.evaluate_streamed([book] * 8, [book] * 32, warmup=4)
        self.assertEqual(collect.call_count, 32)
        self.assertEqual(result["new_test_positions"], 128)
        self.assertEqual(result["new_test_characters"], 128)
        self.assertEqual(len(result["per_book"]), 32)
        rows = result["test"]
        self.assertEqual(rows["memory_only"]["bpb"], 8.0)
        self.assertEqual(rows["oscillator_carrier"]["bpb"], rows["oscillator_plus_spatial_wave"]["bpb"])
        self.assertAlmostEqual(rows["oscillator_plus_spatial_wave"]["nonspace_gain_bpb"], 0.0)
        self.assertAlmostEqual(rows["oscillator_plus_spatial_wave"]["space_gain_bpb"], 0.0)
        self.assertEqual(rows["oscillator_plus_spatial_wave"]["books_improved"], 0)

    def test_wrong_train_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "original 8 TRAIN"):
            m.evaluate_streamed([b"abcdefgh"] * 7, [b"abcdefghi"] * 32, warmup=4)


if __name__ == "__main__":
    unittest.main()
