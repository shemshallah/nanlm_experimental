"""Small correctness tests; not a frontier-scale performance benchmark."""
import argparse
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poincare_frontier import Config, MirrorLM, MirrorPairMoE, file_hash, prepare_books, \
    restore_checkpoint, save_checkpoint


class TestMirrorFrontier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, mode='rollout'):
        return Config(vocab_size=64, dim=32, heads=4, layers=2, pairs=4,
                      topk_pairs=2, expert_hidden=24, context=8,
                      moe_every=2, future_mode=mode)

    def test_exact_mirror_identity(self):
        torch.manual_seed(0)
        m = MirrorPairMoE(self.config())
        plus = m.in_shared + m.in_anti
        minus = m.in_shared - m.in_anti
        self.assertTrue(torch.allclose((plus + minus)/2, m.in_shared))
        self.assertTrue(torch.allclose((plus - minus)/2, m.in_anti))
        self.assertTrue(torch.allclose(m.out_shared,
                                       ((m.out_shared + m.out_anti) +
                                        (m.out_shared - m.out_anti))/2))

    def test_rollout_gradients_and_causality(self):
        torch.manual_seed(1)
        model = MirrorLM(self.config())
        batch = torch.randint(0, 64, (2, 11))
        losses = model.loss_with_future(batch)
        self.assertTrue(torch.isfinite(losses['loss']).item())
        losses['loss'].backward()
        ff = model.blocks[-1].ff
        self.assertGreater(ff.in_anti.grad.abs().sum().item(), 0)
        self.assertGreater(ff.centers.grad.abs().sum().item(), 0)
        model.eval()
        with torch.no_grad():
            x = batch[:, :-3]
            y = x.clone()
            y[:, 5:] = torch.randint(0, 64, y[:, 5:].shape)
            h1 = model(x)[1]
            h2 = model(y)[1]
            l1 = model.future_logits(h1)
            l2 = model.future_logits(h2)
            for a, b in zip(l1, l2):
                self.assertTrue(torch.allclose(a[:, :5], b[:, :5], atol=2e-5))

    def test_books_are_split_by_content_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'books'
            root.mkdir()
            for k in range(12):
                (root/f'{k:02}.txt').write_text((f'Book {k} has unique words. '*12)+'\n')
            (root/'copied.txt').write_bytes((root/'01.txt').read_bytes())
            output = Path(d)/'data'
            args = argparse.Namespace(source_dir=str(root), out_dir=str(output),
                valid_fraction=.25, min_bytes=100, max_doc_bytes=10000,
                seed=2026, tokenizer_model=None)
            prepare_books(args)
            meta = json.loads((output/'meta.json').read_text())
            manifest = json.loads((output/'split_manifest.json').read_text())
            self.assertEqual(meta['train_books'], 9)
            self.assertEqual(meta['valid_books'], 3)
            self.assertEqual(meta['duplicates_ignored'], 1)
            self.assertEqual(meta['token_dtype'], '<u2')
            train_hashes = {r['sha256'] for r in manifest['train_books']}
            valid_hashes = {r['sha256'] for r in manifest['valid_books']}
            self.assertFalse(train_hashes.intersection(valid_hashes))
            self.assertEqual(meta['train_tokens'], len(np.memmap(output/'train.bin', dtype='<u2')))
            old_hash = file_hash(output/'train.bin')
            prepare_books(args)
            self.assertEqual(file_hash(output/'train.bin'), old_hash)

    def test_atomic_single_weight_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self.config()
            model = MirrorLM(cfg)
            optimizer = torch.optim.AdamW(model.parameters())
            rng = np.random.default_rng(7)
            path = Path(d)/'latest.pt'
            save_checkpoint(path, model, optimizer, 3, 900, {'dataset':'test'}, rng, cfg)
            restored = MirrorLM(cfg)
            opt2 = torch.optim.AdamW(restored.parameters())
            p = restore_checkpoint(path, restored, opt2)
            self.assertEqual(p['step'], 3)
            self.assertEqual(p['seen_tokens'], 900)
            self.assertEqual(p['config']['future_mode'], 'rollout')
            self.assertTrue(torch.equal(model.embed.weight, restored.embed.weight))
            self.assertEqual(sorted(x.name for x in Path(d).iterdir()), ['latest.pt'])


if __name__ == '__main__':
    unittest.main()


class TestCapacityAndBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_document_windows_never_cross_boundaries(self):
        from poincare_frontier import DocumentWindowSampler, legacy_document_lengths
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'tokens.bin'
            pieces = [np.array([7]*36 + [256], dtype='<u2'),
                      np.array([12]*35 + [256], dtype='<u2'),
                      np.array([25]*22 + [256], dtype='<u2')]
            np.concatenate(pieces).tofile(f)
            mem = np.memmap(f, dtype='<u2', mode='r')
            lengths = [len(p) for p in pieces]
            sampler = DocumentWindowSampler(mem, context=12, lengths=lengths)
            rng = np.random.default_rng(12)
            batches = [sampler.windows(16, rng, 'cpu') for _ in range(10)]
            samples = torch.cat(batches, dim=0).numpy()
            for sample in samples:
                self.assertEqual(len(set(sample[:-1])), 1)
                self.assertTrue(sample[-1] in (7, 12, 25, 256))
                # 256 only when the EOS is the last target, never in the middle.
                self.assertNotIn(256, sample[:-1])
            self.assertEqual(legacy_document_lengths(mem,
                             {'split_unit':'whole_book','train_books':3,
                              'eos_id':256}, 'train'), lengths)
            checked = 0
            for b in sampler.exhaustive(3, 'cpu', stride=12):
                checked += len(b)
                for row in b:
                    self.assertNotIn(256, row[:-1].tolist())
            self.assertGreater(checked, 0)

    def test_function_preserving_depth_pair_growth_with_adam(self):
        import copy
        from poincare_frontier import expand_architecture, load_v25_compatible
        torch.manual_seed(21)
        oldcfg = Config(vocab_size=40, dim=32, layers=2, heads=4,
                        pairs=4, topk_pairs=2, expert_hidden=24,
                        context=8, future_mode='rollout')
        old = MirrorLM(oldcfg)
        opt = torch.optim.AdamW(old.parameters(), lr=3e-4)
        x = torch.randint(0, 40, (2, 11))
        opt.zero_grad()
        old.loss_with_future(x)['loss'].backward()
        opt.step()
        old.eval()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'latest.pt'
            save_checkpoint(p, old, opt, 17, 1000, {'training':'test'},
                            np.random.default_rng(11), oldcfg)
            saved = torch.load(p, map_location='cpu', weights_only=False)
            newcfg = copy.deepcopy(oldcfg)
            newcfg.pairs = 8
            newcfg.layers = 4
            newcfg.future_block_index = oldcfg.layers - 1
            new = MirrorLM(newcfg)
            newer_opt = torch.optim.AdamW(new.parameters(), lr=1e-4)
            expand_architecture(new, newer_opt, saved, oldcfg)
            new.eval()
            with torch.no_grad():
                a, ah, _ = old(x[:, :-3])
                b, bh, _ = new(x[:, :-3])
                self.assertTrue(torch.equal(a, b))
                af = old.future_logits(ah)
                bf = new.future_logits(bh)
                for ar, br in zip(af, bf):
                    self.assertTrue(torch.equal(ar, br))
            oldmap = dict(old.named_parameters())
            newmap = dict(new.named_parameters())
            self.assertTrue(torch.equal(oldmap['embed.weight'], newmap['embed.weight']))
            self.assertGreater(len(newmap), len(oldmap))
            om = opt.state[oldmap['blocks.1.ff.in_anti']]['exp_avg']
            nm = newer_opt.state[newmap['blocks.1.ff.in_anti']]['exp_avg']
            self.assertTrue(torch.equal(om, nm[:oldcfg.pairs]))
            self.assertTrue(torch.equal(nm[oldcfg.pairs:], torch.zeros_like(nm[oldcfg.pairs:])))
            self.assertEqual(newer_opt.param_groups[0]['lr'], 3e-4)
            # One update must be possible with added layers and expanded moments.
            new.train()
            newer_opt.zero_grad(set_to_none=True)
            new.loss_with_future(x)['loss'].backward()
            newer_opt.step()
            save_checkpoint(p, new, newer_opt, 18, 1100, {'training':'test'},
                            np.random.default_rng(11), newcfg)
            test_restore = MirrorLM(newcfg)
            opt_restore = torch.optim.AdamW(test_restore.parameters())
            loaded = restore_checkpoint(p, test_restore, opt_restore)
            self.assertEqual(loaded['step'], 18)
            self.assertEqual([q.name for q in Path(d).iterdir()], ['latest.pt'])

    def test_v25_checkpoint_load_without_new_buffers(self):
        from poincare_frontier import load_v25_compatible
        cfg = Config(vocab_size=40, dim=32, layers=2, heads=4,
                     pairs=4, topk_pairs=2, expert_hidden=24, context=8)
        model = MirrorLM(cfg)
        v25 = {k: v for k,v in model.state_dict().items() if not k.endswith('pair_bias')}
        restored = MirrorLM(cfg)
        load_v25_compatible(restored, v25)
        for k,v in v25.items():
            self.assertTrue(torch.equal(v, restored.state_dict()[k]))


class TestWideningAndCorpusTransition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_width_growth_preserves_logits_and_new_channels_learn(self):
        import copy
        from poincare_frontier import expand_architecture, load_v25_compatible
        cfg = Config(vocab_size=64, dim=32, layers=2, heads=4,
                     pairs=4, topk_pairs=2, expert_hidden=24,
                     context=8, future_mode='rollout')
        old = MirrorLM(cfg)
        optim = torch.optim.AdamW(old.parameters())
        x = torch.randint(0, cfg.vocab_size, (2, 11))
        optim.zero_grad()
        old.loss_with_future(x)['loss'].backward()
        optim.step()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'latest.pt'
            save_checkpoint(path, old, optim, 27, 3800, {'set':'test'},
                            np.random.default_rng(5), cfg)
            saved = torch.load(path, map_location='cpu', weights_only=False)
            new_cfg = copy.deepcopy(cfg)
            new_cfg.dim *= 2
            new_cfg.heads *= 2
            new_cfg.layers += 2
            new_cfg.pairs *= 2
            new_cfg.expert_hidden *= 2
            new_cfg.norm_divisor = cfg.dim
            new_cfg.future_block_index = cfg.layers - 1
            new = MirrorLM(new_cfg)
            newopt = torch.optim.AdamW(new.parameters())
            expand_architecture(new, newopt, saved, cfg, widen_noise=0.)
            old.eval()
            new.eval()
            with torch.no_grad():
                q, qh, _ = old(x[:,:-3])
                r, rh, _ = new(x[:,:-3])
                self.assertTrue(torch.allclose(q, r, atol=8e-6))
                for a,b in zip(old.future_logits(qh), new.future_logits(rh)):
                    self.assertTrue(torch.allclose(a, b, atol=8e-6))
            self.assertEqual(len(newopt.state), len(optim.state))
            self.assertEqual(new_cfg.norm_divisor, cfg.dim)
            # Seed new coordinates and verify they receive an actual gradient.
            with torch.no_grad():
                new.embed.weight[:,cfg.dim:].normal_(0., 1e-3)
            new.train()
            newopt.zero_grad(set_to_none=True)
            new.loss_with_future(x)['loss'].backward()
            self.assertGreater(new.embed.weight.grad[:,cfg.dim:].abs().sum().item(), 0.)
            newopt.step()

    def test_explicit_corpus_adoption_guard(self):
        import subprocess
        # Two different document sets, identical BYTE vocabulary. A data
        # change without --adopt-data MUST fail, not silently reset training.
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            olddata = base/'old_data'
            newdata = base/'new_data'
            run = base/'run'
            script = str(Path(__file__).resolve().parents[1]/'poincare_frontier.py')
            for title, directory in [('old',olddata), ('new',newdata)]:
                directory.mkdir()
                arr = np.array(([3 if title=='old' else 4]*160)+[256], dtype='<u2')
                arr.tofile(directory/'train.bin')
                arr.tofile(directory/'valid.bin')
                meta = {'vocab_size':257, 'tokenizer_sha256':None, 'unit':'UTF8_bytes_plus_eos',
                        'token_dtype':'<u2', 'tokenizer_model':None,
                        'split_unit':'whole_book', 'train_doc_lengths':[161],
                        'valid_doc_lengths':[161], 'train_books':1,'valid_books':1,
                        'train_sha256':file_hash(directory/'train.bin'),
                        'valid_sha256':file_hash(directory/'valid.bin')}
                (directory/'meta.json').write_text(json.dumps(meta))
            cfg = Config(vocab_size=257,dim=32,layers=2,heads=4,pairs=4,
                         topk_pairs=2,expert_hidden=24,context=8)
            model = MirrorLM(cfg)
            opt = torch.optim.AdamW(model.parameters())
            save_checkpoint(run/'latest.pt', model,opt, 11, 999,
                            json.loads((olddata/'meta.json').read_text()),np.random.default_rng(1),cfg)
            cmd = [sys.executable, script, 'train','--data-dir',str(newdata),
                   '--run-dir',str(run),'--steps','1','--batch','1',
                   '--eval-every','1','--eval-batches','1','--save-every','1','--cpu']
            env = dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            rejected = subprocess.run(cmd,env=env,capture_output=True,text=True)
            self.assertNotEqual(rejected.returncode,0)
            self.assertIn('add --adopt-data',rejected.stderr)
            accepted = subprocess.run(cmd+['--adopt-data'],env=env,capture_output=True,text=True)
            self.assertEqual(accepted.returncode,0,accepted.stderr)
            cp = torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
            self.assertEqual(cp['step'],12)
            self.assertEqual(cp['growth_history'][-1]['kind'],'data_transition')
            self.assertEqual(len(list(run.iterdir())),1)


class TestDispatchAndSizing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_chunked_dispatch_matches_outputs_and_gradients(self):
        import copy
        torch.manual_seed(9)
        c1 = Config(vocab_size=48,dim=32,layers=2,heads=4,pairs=4,
                    topk_pairs=2,expert_hidden=24,context=8,future_mode='rollout')
        c2 = copy.deepcopy(c1)
        c2.dispatch_mode = 'gather'
        c2.gather_chunk = 3
        a = MirrorLM(c1)
        b = MirrorLM(c2)
        b.load_state_dict(a.state_dict())
        x = torch.randint(0,48,(2,11))
        la = a.loss_with_future(x)['loss']
        lb = b.loss_with_future(x)['loss']
        self.assertTrue(torch.allclose(la,lb,atol=1e-6))
        la.backward()
        lb.backward()
        for (k,p),(k2,q) in zip(a.named_parameters(),b.named_parameters()):
            self.assertEqual(k,k2)
            if p.grad is not None:
                self.assertTrue(torch.allclose(p.grad,q.grad,atol=1e-5), k)

    def test_size_formula_counts_tied_head_once(self):
        from poincare_frontier import count_parameters_exact
        cfg = Config(dim=48,heads=4,layers=2,pairs=8,topk_pairs=2,
                     expert_hidden=64,context=96)
        self.assertEqual(count_parameters_exact(cfg),
                         sum(p.numel() for p in MirrorLM(cfg).parameters()))
        big = Config(vocab_size=32000,dim=256,layers=8,heads=8,
                     pairs=32,topk_pairs=2,expert_hidden=256,context=2048)
        self.assertEqual(count_parameters_exact(big),47355776)
