"""Checkpoint and causal invariants for auxiliary-only multi-byte forecasting."""
import argparse
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poincare_frontier import (Config, MirrorLM, count_parameters_exact,
                              expand_architecture, grow_checkpoint,
                              prepare_books, save_checkpoint)


def config(isolated=False):
    return Config(vocab_size=48, dim=32, layers=2, heads=4,
                  pairs=4, topk_pairs=2, expert_hidden=24, context=16,
                  moe_every=2, future_mode='isolated' if isolated else 'rollout',
                  future_adapter_rank=8 if isolated else 0)


class IsolatedFutureAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_migration_preserves_all_now_weights_and_adam_moments(self):
        torch.manual_seed(911)
        old_cfg = config()
        old = MirrorLM(old_cfg)
        old_opt = torch.optim.AdamW(old.parameters(), lr=3e-4)
        seq = torch.randint(0, old_cfg.vocab_size, (2, 11))
        old.loss_with_future(seq)['loss'].backward()
        old_opt.step()
        saved = {'model':copy.deepcopy(old.state_dict()),
                 'optimizer':copy.deepcopy(old_opt.state_dict())}

        new = MirrorLM(config(True))
        new_opt = torch.optim.AdamW(new.parameters(), lr=3e-4)
        expand_architecture(new, new_opt, saved, old_cfg)
        old.eval(); new.eval()
        with torch.no_grad():
            now_before = old(seq[:, :-3])[0]
            now_after = new(seq[:, :-3])[0]
            future = new.future_logits(new(seq[:, :-3])[1])
        self.assertTrue(torch.equal(now_before, now_after))
        self.assertEqual(future[0].shape, now_before.shape)
        self.assertTrue(torch.equal(future[0], now_before))
        self.assertTrue(torch.equal(future[1], now_before))
        self.assertEqual(count_parameters_exact(new.cfg),
                         sum(p.numel() for p in new.parameters()))
        self.assertEqual(len(old_opt.state), len(new_opt.state))
        for before, after in zip(old.parameters(), new.parameters()):
            self.assertTrue(torch.equal(before, after))
            if before in old_opt.state:
                for key, val in old_opt.state[before].items():
                    self.assertTrue(torch.equal(val, new_opt.state[after][key]))

    def test_future_training_cannot_update_now_path_or_optimizer(self):
        torch.manual_seed(919)
        model = MirrorLM(config(True))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.1)
        for param in model.parameters():
            param.requires_grad_(False)
        for param in model.future_adapter.parameters():
            param.requires_grad_(True)
        before = {name: p.detach().clone() for name,p in model.named_parameters()
                  if not name.startswith('future_adapter.')}
        x = torch.randint(0, 48, (2, 12))
        with torch.no_grad():
            base_logits = model(x[:, :-3])[0].clone()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            losses = model.loss_with_future(x)
            (losses['future2_ce'] + losses['future3_ce']).backward()
            optimizer.step()
        self.assertGreater(float(model.future_adapter.up.weight.abs().sum()),0.)
        for name, p in model.named_parameters():
            if name in before:
                self.assertTrue(torch.equal(before[name],p), name)
                self.assertIsNone(p.grad, name)
                self.assertNotIn(p, optimizer.state, name)
        self.assertTrue(torch.equal(base_logits,model(x[:, :-3])[0]))

    def test_multi_horizon_causality_and_no_label_access(self):
        torch.manual_seed(929)
        model = MirrorLM(config(True)).eval()
        with torch.no_grad():
            model.future_adapter.up.weight.normal_(0, .03)
            model.future_adapter.step_offsets.normal_(0, .02)
            a = torch.randint(0,48,(1,12))
            b = a.clone(); b[:,6:] = torch.randint(0,48,b[:,6:].shape)
            ah = model(a)[1]
            bh = model(b)[1]
            a2,a3 = model.future_logits(ah)
            b2,b3 = model.future_logits(bh)
        self.assertTrue(torch.allclose(a2[:,:6], b2[:,:6], atol=2e-5,rtol=0))
        self.assertTrue(torch.allclose(a3[:,:6], b3[:,:6], atol=2e-5,rtol=0))
        self.assertFalse(torch.equal(a2,a3))

    def test_cli_growth_keeps_step_and_rng_and_records_switch(self):
        torch.manual_seed(937)
        with tempfile.TemporaryDirectory() as tmp:
            cfg=config()
            model=MirrorLM(cfg)
            opt=torch.optim.AdamW(model.parameters())
            ids=torch.randint(0,48,(1,9))
            model.loss_with_future(ids)['loss'].backward();opt.step()
            p=Path(tmp)/'latest.pt'
            save_checkpoint(p,model,opt,2865,123456,{'vocab_size':48},
                            np.random.default_rng(33),cfg)
            previous=torch.load(p,map_location='cpu',weights_only=False)
            args=argparse.Namespace(run_dir=tmp,widen_dim=None,add_layers=0,
                grow_pairs=None,grow_expert_hidden=None,set_context=None,
                dispatch_mode=None,gather_chunk=None,geometry_memory_chunk=None,
                symbiotic_rank=None,hierarchy_rank=None,attention_window=None,
                future_adapter_rank=8,future_mode='isolated',
                widen_noise=0.,initial_pair_bias=-12.,grow_warmup=400,
                max_logit_drift=1e-6,threads=1,report=None)
            grow_checkpoint(args)
            current=torch.load(p,map_location='cpu',weights_only=False)
            self.assertEqual(current['step'],2865)
            self.assertEqual(current['seen_tokens'],123456)
            self.assertTrue(torch.equal(current['rng_torch'],previous['rng_torch']))
            self.assertEqual(current['rng_numpy'],previous['rng_numpy'])
            self.assertEqual(current['config']['future_mode'],'isolated')
            self.assertEqual(current['config']['future_adapter_rank'],8)
            self.assertTrue(current['growth_history'][-1]
                            ['adam_moment_fields_verified'] > 0)

    def test_real_train_cli_future_only_keeps_existing_weights_and_adam(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            books=root/'books';books.mkdir()
            for i in range(5):
                (books/f'book-{i}.txt').write_text(
                    (f'Chapter {i}: A quiet river runs into the sea. '*40), encoding='utf-8')
            data=root/'data'
            prepare_books(argparse.Namespace(source_dir=str(books),out_dir=str(data),
                          valid_fraction=.2,test_fraction=0.,min_bytes=100,
                          max_doc_bytes=100000,seed=2026,tokenizer_model=None,
                          exclude_gutenberg_ids=None))
            meta=json.loads((data/'meta.json').read_text())
            cfg=Config(vocab_size=257,dim=32,layers=2,heads=4,pairs=4,
                       topk_pairs=2,expert_hidden=24,context=16,moe_every=2,
                       future_mode='rollout')
            model=MirrorLM(cfg)
            optimizer=torch.optim.AdamW(model.parameters())
            sample=torch.randint(0,257,(1,12))
            model.loss_with_future(sample)['loss'].backward();optimizer.step()
            run=root/'candidate';run.mkdir()
            ckpt=run/'latest.pt'
            save_checkpoint(ckpt,model,optimizer,2865,123456,meta,
                            np.random.default_rng(3),cfg)
            before=torch.load(ckpt,map_location='cpu',weights_only=False)
            command=[sys.executable,str(Path(__file__).resolve().parents[1]/'poincare_frontier.py'),
                     'train','--data-dir',str(data),'--run-dir',str(run),
                     '--steps','1','--batch','1','--train-context','8',
                     '--eval-context','8','--cpu','--cpu-threads','1',
                     '--eval-batches','1','--eval-every','1','--save-every','1',
                     '--future-mode','isolated','--future-adapter-rank','8','--future-only']
            result=subprocess.run(command,text=True,capture_output=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            after=torch.load(ckpt,map_location='cpu',weights_only=False)
            self.assertEqual(after['step'],2866)
            for name,p in before['model'].items():
                self.assertTrue(torch.equal(p,after['model'][name]),name)
            old_state=before['optimizer']['state']
            new_state=after['optimizer']['state']
            for index,state in old_state.items():
                for key,val in state.items():
                    self.assertTrue(torch.equal(val,new_state[index][key]),str((index,key)))
            self.assertTrue(any(name.startswith('future_adapter.')
                                for name in after['model']))


if __name__ == '__main__':
    unittest.main()
