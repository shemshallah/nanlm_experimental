import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from import_nanrouter_actions_artifact import import_archive

class NanrouterArtifactTests(unittest.TestCase):
    def setUp(self):
        self.zip_path=Path('/mnt/data/nanrouter_raw_prose_books.zip')
    def test_real_artifact_split_and_hashes(self):
        if not self.zip_path.exists():self.skipTest('GitHub artifact not mounted')
        with tempfile.TemporaryDirectory() as t:
            m=import_archive(self.zip_path,Path(t),require_counts=True)
            self.assertEqual((m['train_books'],m['valid_books'],m['test_books']),(12,12,14))
            self.assertEqual(sum(m['train_doc_lengths']),m['train_tokens'])
            self.assertEqual(sum(m['valid_doc_lengths']),m['valid_tokens'])
            self.assertEqual(sum(m['test_doc_lengths']),m['test_tokens'])
            self.assertEqual(hashlib.sha256((Path(t)/'split_manifest.json').read_bytes()).hexdigest(),m['split_manifest_sha256'])
            man=json.loads((Path(t)/'split_manifest.json').read_text())
            self.assertEqual(len({r['gutenberg_id'] for r in man['books']}),38)
            self.assertEqual(len({r['sha256'] for r in man['books']}),38)
    def test_corrupt_book_rejected(self):
        if not self.zip_path.exists():self.skipTest('GitHub artifact not mounted')
        with tempfile.TemporaryDirectory() as t:
            changed=Path(t)/'bad.zip'
            with zipfile.ZipFile(self.zip_path) as src, zipfile.ZipFile(changed,'w',compression=zipfile.ZIP_DEFLATED) as dst:
                for item in src.infolist():
                    b=src.read(item.filename)
                    if item.filename=='train/10466.txt':b=b'CORRUPTED '+b
                    dst.writestr(item.filename,b)
            with self.assertRaisesRegex(ValueError,'SHA256'):
                import_archive(changed,Path(t)/'bad_data',require_counts=True)

if __name__=='__main__':unittest.main()
