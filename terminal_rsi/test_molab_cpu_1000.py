"""Offline Molab CPU launch checks; never touches trained checkpoints."""
from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from terminal_rsi.run_molab_cpu_1000 import verify_release


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
