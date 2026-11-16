"""FA2 R55: train-only trust, inherited Adam preservation and paired controls."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from poincare_frontier import Config, MirrorLM
from train_r55_pair_successor import (cache_books, memory_arm, score_books,
                                      train_only_gate, audit_inheritance,
                                      evaluate, ARMS)


class R55TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_same_model_and_optimizer_train_only_new_gate(self):
        torch.manual_seed(55)
        model = MirrorLM(Config(dim=16, layers=2, heads=2, pairs=4,
            topk_pairs=1, expert_hidden=16, context=16, moe_every=2,
            symbiotic_rank=8, micro_meta=True, simplex_transfer=True,
            dispatch_mode='gather', balance_weight=0.0))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        ids = torch.tensor([[65,32,66,65,32,66,65,32,66]])
        _, _, metrics = model(ids,ids)
        metrics['loss'].backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        parent = {'model':deepcopy(model.state_dict()),
                  'optimizer':deepcopy(optimizer.state_dict())}
        with torch.no_grad():
            before = model(ids)[0]
        model.enable_pair_successor(optimizer)
        for name, param in model.named_parameters():
            param.requires_grad_(name == 'pair_successor.weight')
        stream = np.array([32 if t % 4 == 0 else 65 + (t % 2)
                           for t in range(180)],dtype=np.uint16)
        with memory_arm(model.pair_successor,'neutral',None):
            records = cache_books(model,stream,[len(stream)],8,2,1,2026)
        weights_before = model.pair_successor.weight.detach().clone()
        learned = train_only_gate(model.pair_successor,optimizer,records,6,.03,56,
                                  nonspace_weight=3.0)
        self.assertEqual(learned['updates'],6)
        self.assertEqual(learned['nonspace_weight'],3.0)
        self.assertGreater(learned['tokens_processed'],0)
        self.assertFalse(torch.equal(weights_before,model.pair_successor.weight))
        lineage = audit_inheritance(parent,model,optimizer,True)
        self.assertTrue(lineage['pass'],lineage['issues'])
        self.assertEqual(lineage['inherited_adam_states'],
                         lineage['retained_adam_states'])
        self.assertTrue(lineage['gate_adam_state_present'])
        wrong_queries = torch.arange(32640).roll(137)
        rows = {arm:score_books(model.pair_successor,records,arm,wrong_queries)
                for arm in ARMS}
        self.assertEqual(rows['neutral'],score_books(model.pair_successor,
                          records,'neutral',wrong_queries))
        # No gate manipulation may change inherited model predictions.
        with torch.no_grad(), memory_arm(model.pair_successor,'neutral',None):
            self.assertTrue(torch.equal(before,model(ids)[0]))

    def test_control_context_restores_even_on_exception(self):
        from pair_successor_memory import PairSuccessorTrust
        trust = PairSuccessorTrust()
        with torch.no_grad():
            trust.weight.fill_(.5)
        old = trust.weight.detach().clone()
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with memory_arm(trust,'neutral',torch.arange(32640)):
                self.assertEqual(trust.weight.abs().sum().item(),0.)
                raise RuntimeError('injected')
        self.assertTrue(torch.equal(trust.weight,old))
        self.assertIsNone(trust.query_map)
        self.assertEqual(trust.value_control,'exact')


if __name__ == '__main__':
    unittest.main()
