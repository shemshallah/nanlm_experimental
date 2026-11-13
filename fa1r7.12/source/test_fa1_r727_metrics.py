#!/usr/bin/env python3
"""R7.27 standalone probability and whole-book metadata regressions."""
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from fa1_r727_eval import normalized_trunk_word_topk, validate_book_corpus


class UniformTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(7, 96))
        self.B = nn.Parameter(torch.zeros(7))

    def nll(self, hidden, target):
        return torch.full((len(target),), 3 * math.log(2), dtype=torch.float32)


class TestMetrics(unittest.TestCase):
    def test_exact_normalized_word_tree(self):
        # Eight leaves, seven independent binary decisions. A uniform tree
        # must assign exactly 1/8 to every word, not a target-specific score.
        base = type('Vocabulary', (), {})()
        base.vocab = list('abcdefgh')
        base.children = np.asarray([(0, 1), (2, 3), (4, 5), (6, 7),
                                    (8, 9), (10, 11), (12, 13)])
        model = type('Model', (), {})()
        model.core = type('Core', (), {})()
        model.core.trunk = UniformTrunk()
        out = normalized_trunk_word_topk(model, base, torch.randn(8, 96), torch.arange(8))
        self.assertEqual(out['trunk_vocabulary_size'], 8)
        self.assertAlmostEqual(out['trunk_word_loss_nats_per_word'], 3 * math.log(2), places=5)
        self.assertAlmostEqual(out['trunk_next_word_top5'], 5/8, places=6)
        self.assertTrue(out['trunk_normalization_verified'])
        self.assertIsNone(out['full_model_word_top1'])

    def test_book_split_disjoint_and_malformed_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, specs in {'train': [('train', 4)],
                                'dev': [('dev1', 3), ('dev2', 5)],
                                'test': [('test', 2)]}.items():
                starts = []
                docs = []
                first = 0
                for label, words in specs:
                    starts.extend([first] * words)
                    docs.append({'sha256': label, 'words': words, 'first_record': first})
                    first += words
                np.asarray(starts, dtype='<u4').tofile(root / (name + '_doc_start.u32'))
                (root / (name + '_documents.json')).write_text(json.dumps({
                    'source': 'individual Gutenberg text files',
                    'documents': docs, 'total_word_targets': first}))
            self.assertEqual(validate_book_corpus(root, 'dev')['dev']['books'], 2)
            doc = json.loads((root / 'test_documents.json').read_text())
            doc['documents'][0]['sha256'] = 'dev1'
            (root / 'test_documents.json').write_text(json.dumps(doc))
            with self.assertRaises(ValueError):
                validate_book_corpus(root, 'dev')


if __name__ == '__main__':
    unittest.main()
