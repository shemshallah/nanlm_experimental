"""Offline Molab CPU launch checks; never touches trained checkpoints."""
from __future__ import annotations

import hashlib
import io
import tarfile
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from terminal_rsi.run_molab_cpu_1000 import ARCHIVE, CHECKSUMS, source_dir, verify_release


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
                f"{hashlib.sha256(compressed).hexdigest()}  {ARCHIVE}\\n"
            )
            with patch("terminal_rsi.run_molab_cpu_1000.shutil.which", return_value=None):
                extracted = source_dir(root, None)
            self.assertEqual((extracted / "books" / "123--english.txt").read_bytes(), data)
            self.assertTrue((extracted / "extracted.complete").exists())

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
