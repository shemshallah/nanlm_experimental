"""Exercise stochastic multi-window continuation with an on-disk canonical checkpoint."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poincare_frontier import prepare_books
from argparse import Namespace


class ContextContinuation(unittest.TestCase):
    def test_same_checkpoint_with_mixed_context_and_exact_token_counter(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            books = root / 'books'
            books.mkdir()
            for j in range(5):
                (books / f'{j}.txt').write_text(('A unique book number %d for testing. ' % j) * 50)
            data = root / 'data'
            prepare_books(Namespace(source_dir=str(books), out_dir=str(data),
                tokenizer_model=None, valid_fraction=.2, min_bytes=100,
                max_doc_bytes=250000, seed=2026))
            source = Path(__file__).resolve().parents[1] / 'poincare_frontier.py'
            run = root / 'run'
            cmd = [sys.executable, str(source), 'train', '--data-dir',str(data),
                '--run-dir',str(run), '--cpu', '--cpu-threads','1',
                '--dim','32', '--heads','4', '--layers','2', '--pairs','4',
                '--expert-hidden','16', '--context','16', '--batch','1',
                '--steps','3', '--context-mix','8:2,16:1', '--eval-context','8',
                '--eval-batches','1', '--eval-every','3', '--save-every','3']
            env=dict(os.environ, OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            result=subprocess.run(cmd,capture_output=True,text=True,env=env,timeout=45)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            saved=torch.load(run/'latest.pt',weights_only=False)
            self.assertEqual(saved['step'],3)
            self.assertGreaterEqual(saved['seen_tokens'],3*8)
            self.assertLessEqual(saved['seen_tokens'],3*16)
            self.assertEqual(saved['seen_tokens'] % 8,0)
            self.assertTrue(any(h.get('kind')=='cpu_context_mixture'
                                for h in saved['growth_history']))
            self.assertEqual(len(list(run.glob('*.pt'))),1)
            # Resume on the same file and optimizer, with the same fixed tokenizer.
            cmd[cmd.index('--steps')+1]='2'
            result=subprocess.run(cmd,capture_output=True,text=True,env=env,timeout=45)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            continued=torch.load(run/'latest.pt',weights_only=False)
            self.assertEqual(continued['step'],5)
            self.assertGreater(continued['seen_tokens'],saved['seen_tokens'])
            self.assertTrue(continued['optimizer']['state'])
            self.assertEqual(len(list(run.glob('*.pt'))),1)

if __name__=='__main__':
    unittest.main()
