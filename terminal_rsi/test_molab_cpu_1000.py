"""Offline Molab CPU launch checks; never touches trained checkpoints."""
from __future__ import annotations

import hashlib
import io
import zipfile
import tarfile
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from terminal_rsi.run_molab_cpu_1000 import ARCHIVE, CHECKSUMS, source_dir, verify_release, create_download_bundle


class MolabCPU1000Tests(unittest.TestCase):
    def test_sha256_release_accepts_exact_archive(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            path = base / "short_stories-0004.tar.zst"
            contents = b"sample fixture archive bytes only"
            path.write_bytes(contents)
            digest = hashlib.sha256(contents).hexdigest()
            sums = base / "short_stories-0004.SHA256SUMS"
            sums.write_text(f"{digest}  {path.name}\n")
            self.assertEqual(verify_release(path, sums), digest)

    def test_archive_tampering_rejected(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            path = base / "short_stories-0004.tar.zst"
            path.write_bytes(b"original")
            digest = hashlib.sha256(b"original").hexdigest()
            sums = base / "short_stories-0004.SHA256SUMS"
            sums.write_text(f"{digest}  {path.name}\n")
            path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                verify_release(path, sums)

    def test_pure_python_zstandard_extraction_without_system_tools(self):
        import zstandard
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "nanrouter_batch0004"
            cache.mkdir(parents=True)
            data = b"English story book source"
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w") as tar:
                info = tarfile.TarInfo(name="books/123--english.txt")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            compressed = zstandard.ZstdCompressor().compress(buffer.getvalue())
            (cache / ARCHIVE).write_bytes(compressed)
            (cache / CHECKSUMS).write_text(
                f"{hashlib.sha256(compressed).hexdigest()}  {ARCHIVE}\n"
            )
            with patch("terminal_rsi.run_molab_cpu_1000.shutil.which", return_value=None):
                extracted = source_dir(root, None)
            self.assertEqual((extracted / "books" / "123--english.txt").read_bytes(), data)
            self.assertTrue((extracted / "extracted.complete").exists())

    def test_download_bundle_has_exact_checkpoint_once_and_verified_sha(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "champions").mkdir()
            files = [
                "terminal_rsi_cpu_1000_metrics.json",
                "terminal_rsi_cpu_1000_console.log",
                "cpu_pilot.manifest.json",
                "win_dev.manifest.json",
                "train_book_ids.json",
                "train_cpu_pilot.bin",
                "win_dev.bin",
                "champions/best.json",
                "champions/step1000_abc.pt",
            ]
            for path in files:
                (output / path).write_bytes(f"fixture:{path}".encode())
            receipt = {
                "champion_guard": {
                    "champion_checkpoint": str(output / "champions/step1000_abc.pt")
                },
            }
            bundle = create_download_bundle(output, receipt)
            with zipfile.ZipFile(bundle) as archive:
                names = archive.namelist()
                self.assertEqual(names.count("champions/step1000_abc.pt"), 1)
                self.assertEqual(len([n for n in names if n.endswith(".pt")]), 1)
                self.assertIn("RECOVERY.txt", names)
                self.assertIn("SHA256SUMS.txt", names)
                for line in archive.read("SHA256SUMS.txt").decode().splitlines():
                    digest, name = line.split("  ", 1)
                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), digest)
            with self.assertRaisesRegex(ValueError, "already exists"):
                create_download_bundle(output, receipt)

    def test_missing_archive_entry_rejected(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            path = base / "short_stories-0004.tar.zst"
            path.write_bytes(b"sample")
            sums = base / "short_stories-0004.SHA256SUMS"
            sums.write_text("0" * 64 + "  unknown.tar.zst\n")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                verify_release(path, sums)


if __name__ == "__main__":
    unittest.main()
