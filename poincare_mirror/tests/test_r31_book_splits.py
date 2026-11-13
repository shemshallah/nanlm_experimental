"""Protect held-out books across new NaNROUTER corpus adoption."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poincare_frontier import prepare_books


class TestBookSplits(unittest.TestCase):
    def test_held_out_split_and_excluded_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            books = root / 'books'
            books.mkdir()
            for i in range(20):
                (books / f'{1000 + i}--book.txt').write_text((f'Book {i} text. ' * 25))
            (root / 'old_ids.json').write_text('[1000, 1001]')
            args = argparse.Namespace(source_dir=str(books), out_dir=str(root / 'data'),
                valid_fraction=.2, test_fraction=.2, min_bytes=50,
                max_doc_bytes=100000, seed=2026, tokenizer_model=None,
                exclude_gutenberg_ids=str(root / 'old_ids.json'))
            prepare_books(args)
            meta = json.loads((root / 'data' / 'meta.json').read_text())
            manifest = json.loads((root / 'data' / 'split_manifest.json').read_text())
            self.assertEqual(meta['unique_books'], 18)
            self.assertEqual(meta['train_books'] + meta['valid_books'] + meta['test_books'], 18)
            self.assertEqual(len(manifest['excluded_files']), 2)
            splits = [{b['sha256'] for b in manifest[f'{split}_books']}
                for split in ('train', 'valid', 'test')]
            self.assertFalse(splits[0] & splits[1])
            self.assertFalse(splits[0] & splits[2])
            self.assertFalse(splits[1] & splits[2])
            self.assertEqual(sum(meta[f'{s}_doc_lengths'][i] for s in ('train', 'valid', 'test')
                for i in range(len(meta[f'{s}_doc_lengths']))),
                sum(meta[f'{s}_tokens'] for s in ('train', 'valid', 'test')))


if __name__ == '__main__':
    unittest.main()
