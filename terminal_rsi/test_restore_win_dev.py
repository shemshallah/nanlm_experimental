"""CPU-only proof the original notebook reference is reconstructed losslessly."""
import base64
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zlib

from terminal_rsi import restore_win_dev as restore_module


class OriginalDEVRestoration(unittest.TestCase):
    def test_original_embedded_dev_and_frozen_c_sha_idempotence(self):
        with tempfile.TemporaryDirectory(prefix="v103_archive_") as d:
            root = Path(d)
            skip = 32
            books = [(b"private independent sample " * 3)[:42] for _ in range(16)]
            win = struct.pack("<II", 16, skip) + b"".join(
                struct.pack("<I", len(book)) + book for book in books)
            frozen = b"frozen C reference test fixture"
            records = {"win_dev.bin": win,
                       "curve_dev_frozen_10000000.bin": frozen,
                       "win_test.bin": b"sealed TEST must never restore"}
            compressed = {
                name: (base64.b64encode(zlib.compress(data)).decode("ascii"),
                       hashlib.sha256(data).hexdigest())
                for name, data in records.items()}
            notebook = root / "original_v103_fixture.ipynb"
            notebook.write_text(json.dumps({"cells": [{
                "cell_type": "code", "source": [
                    "ORIGINAL_V103_INDEPENDENT_REFERENCES = " +
                    repr(compressed)]}]}))
            output = root / "restored"
            with patch.dict(restore_module.PINNED_REFERENCES,
                            {name: info[1] for name, info in compressed.items()}):
                first = restore_module.restore(notebook, output, include_frozen_c=True)
                self.assertEqual(first["win_dev.bin"]["status"], "restored_original")
                self.assertEqual((output / "win_dev.bin").read_bytes(), win)
                self.assertEqual((output / "curve_dev_frozen_10000000.bin").read_bytes(), frozen)
                self.assertFalse((output / "win_test.bin").exists())
                second = restore_module.restore(notebook, output, include_frozen_c=True)
                self.assertEqual(second["win_dev.bin"]["status"], "verified_existing")
                (output / "win_dev.bin").write_bytes(b"wrong split")
                with self.assertRaisesRegex(FileExistsError, "non-identical"):
                    restore_module.restore(notebook, output)
                self.assertEqual((output / "win_dev.bin").read_bytes(), b"wrong split")

    def test_rejects_altered_archive_digest_before_any_write(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            notebook = root / "bad.ipynb"
            notebook.write_text(json.dumps({"cells": [{
                "cell_type": "code",
                "source": ["ORIGINAL_V103_INDEPENDENT_REFERENCES = "
                           "{'win_dev.bin': ('eNpjYGBgAAAABAAB', 'bad')}"]
            }]}))
            with self.assertRaisesRegex(ValueError, "SHA"):
                restore_module.restore(notebook, root / "output")
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
