"""R28 migration and true book-corpus adoption regression tests."""
import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poincare_frontier import (Config, MirrorLM, file_hash, grow_checkpoint,
    prepare_books, save_checkpoint, restore_checkpoint)


class Round28Continuity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_checkpoint(self, directory):
        old = Config(vocab_size=257, dim=24, layers=2, heads=2,
                     pairs=4, topk_pairs=2, expert_hidden=16, context=16,
                     future_mode='rollout')
        model = MirrorLM(old)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        ids = torch.randint(0, 257, (1, 19))
        optimizer.zero_grad()
        model.loss_with_future(ids)['loss'].backward()
        optimizer.step()
        meta = {'vocab_size':257, 'tokenizer_sha256':None,
                'unit':'UTF8_bytes_plus_eos', 'split_unit':'whole_book',
                'train_sha256':'pre-book-hash', 'valid_sha256':'pre-book-valid'}
        path = Path(directory) / 'latest.pt'
        save_checkpoint(path, model, optimizer, 88, 1408, meta,
                        np.random.default_rng(12), old)
        return path

    def test_growth_retains_training_state_and_new_width_learns(self):
        torch.manual_seed(19)
        with tempfile.TemporaryDirectory() as d:
            path = self.make_checkpoint(d)
            pre = torch.load(path, weights_only=False)
            optargs = argparse.Namespace(run_dir=d, widen_dim=48,
                add_layers=2, grow_pairs=8, grow_expert_hidden=32,
                set_context=128,dispatch_mode='gather',gather_chunk=8,
                widen_noise=1e-4, initial_pair_bias=-12.,grow_warmup=200,
                max_logit_drift=5e-4,threads=1,report=None)
            grow_checkpoint(optargs)
            post = torch.load(path, weights_only=False)
            self.assertEqual(post['step'], 88)
            self.assertEqual(post['seen_tokens'], 1408)
            self.assertEqual(post['data'], pre['data'])
            self.assertTrue(torch.equal(post['rng_torch'],pre['rng_torch']))
            self.assertEqual(post['rng_python'],pre['rng_python'])
            self.assertEqual(post['rng_numpy'],pre['rng_numpy'])
            self.assertEqual(post['config']['context'],128)
            self.assertEqual(post['config']['pairs'],8)
            self.assertEqual(post['config']['dim'],48)
            self.assertEqual([p.name for p in Path(d).iterdir()],['latest.pt'])
            cfg=Config(**post['config'])
            model=MirrorLM(cfg)
            optimizer=torch.optim.AdamW(model.parameters())
            restore_checkpoint(path,model,optimizer)
            x=torch.randint(0,257,(1,19))
            loss=model.loss_with_future(x)['loss']
            self.assertTrue(torch.isfinite(loss))
            optimizer.zero_grad()
            loss.backward()
            self.assertGreater(model.embed.weight.grad[:,24:].abs().sum().item(),0)
            self.assertGreater(model.blocks[3].att.out.weight.grad.abs().sum().item(),0)
            self.assertGreater(model.blocks[1].ff.out_shared.grad[:4,:,24:].abs().sum().item(),0)

    def test_adopt_book_corpus_with_v27_dtype_missing(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            path=self.make_checkpoint(root/'run') if False else None
            (root/'run').mkdir()
            path=self.make_checkpoint(root/'run')
            books=root/'books'
            books.mkdir()
            for i in range(6):
                (books/f'gutenberg-{i}.txt').write_text((f'Book {i} a sentence of prose. '*40), encoding='utf-8')
            prep=argparse.Namespace(source_dir=str(books),out_dir=str(root/'data'),
                                   valid_fraction=.25,min_bytes=100,
                                   max_doc_bytes=100000,seed=2026,tokenizer_model=None)
            prepare_books(prep)
            cmd=[sys.executable,str(Path(__file__).resolve().parents[1]/'poincare_frontier.py'),
                  'train','--data-dir',str(root/'data'),'--run-dir',str(root/'run'),
                  '--batch','1','--train-context','8','--grad-accum','2',
                  '--eval-batches','1','--eval-every','1','--save-every','1','--cpu']
            env={'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
            proc=subprocess.run(cmd+['--adopt-data','--steps','0'],
                    stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
            self.assertEqual(proc.returncode,0,proc.stderr[-2500:])
            self.assertIn('zero_training_updates',proc.stdout)
            adopted=torch.load(path,weights_only=False)
            self.assertEqual(adopted['step'],88)
            self.assertEqual(adopted['seen_tokens'],1408)
            proc=subprocess.run(cmd+['--steps','1'],
                    stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
            self.assertEqual(proc.returncode,0,proc.stderr[-2500:])
            updated=torch.load(path, weights_only=False)
            self.assertEqual(updated['step'],89)
            self.assertEqual(updated['seen_tokens'],1424)
            self.assertEqual(updated['data']['split_unit'],'whole_book')
            self.assertEqual(updated['data']['token_dtype'],'<u2')
            self.assertEqual(updated['data']['train_sha256'],file_hash(root/'data'/'train.bin'))
            self.assertTrue(any(e.get('kind')=='data_transition' for e in updated['growth_history']))


if __name__=='__main__':
    unittest.main()
