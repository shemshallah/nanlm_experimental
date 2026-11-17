"""R68 promotion records its failed gates and makes active loading fail closed."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fa2_active import resolve_active_checkpoint


class ActiveRecoveryTests(unittest.TestCase):
    def test_explicit_path_and_digest_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path=root/'FA2_ACTIVE.pt'
            payload=b'unified-FA2-test-placeholder'
            path.write_bytes(payload)
            manifest=root/'FA2_ACTIVE.json'
            manifest.write_text(json.dumps({'architecture':'FA2_UNIFIED',
                'active_step':4209,'checkpoint_filename':path.name,
                'checkpoint_sha256':hashlib.sha256(payload).hexdigest()}))
            chosen,record=resolve_active_checkpoint(path,manifest)
            self.assertEqual(chosen,path)
            self.assertEqual(record['active_step'],4209)
            path.write_bytes(b'wrong-weights')
            with self.assertRaisesRegex(ValueError,'SHA-256'):
                resolve_active_checkpoint(path,manifest)

    def test_default_search_requires_real_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest=root/'FA2_ACTIVE.json'
            manifest.write_text(json.dumps({'architecture':'FA2_UNIFIED',
                'active_step':4209,'checkpoint_filename':'does_not_exist.pt',
                'checkpoint_sha256':'0'*64}))
            with self.assertRaises(FileNotFoundError):
                resolve_active_checkpoint(manifest_path=manifest)


if __name__=='__main__':
    unittest.main()
