"""R40 audit must detect optimizer resets and corpus substitutions."""
import copy
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_r40_lineage import audit


def sample_pair():
    weights = {'head.weight': torch.tensor([1., 2.])}
    state = {0: {'step': torch.tensor(9.), 'exp_avg': torch.tensor([.1, -.2]),
                 'exp_avg_sq': torch.tensor([.01, .04])}}
    optimizer = {'param_groups': [{'params': [0], 'lr': .01}], 'state': state}
    before = {'model': weights, 'optimizer': optimizer, 'step': 9,
              'config': {'dim': 2}, 'data': {'split_manifest_sha256': 'fixed'},
              'rng_numpy': {}, 'rng_torch': torch.ones(1), 'rng_python': (1,)}
    after = copy.deepcopy(before)
    after['step'] = 12
    after['model']['head.weight'][0] += .1
    after['optimizer']['state'][0]['step'] += 3
    return before, after


class LineageTests(unittest.TestCase):
    def test_accepts_unchanged_model_identity_with_continuous_adam(self):
        before, after = sample_pair()
        result = audit(before, after)
        self.assertTrue(result['lineage_pass'], result['issues'])
        self.assertEqual(result['inherited_adam_states_advanced'], 1)
        self.assertEqual(result['steps_added'], 3)

    def test_rejects_adam_counter_reset(self):
        before, after = sample_pair()
        after['optimizer']['state'][0]['step'] = torch.tensor(2.)
        self.assertFalse(audit(before, after)['lineage_pass'])

    def test_rejects_dropped_optimizer_moments(self):
        before, after = sample_pair()
        del after['optimizer']['state'][0]['exp_avg']
        self.assertFalse(audit(before, after)['lineage_pass'])

    def test_rejects_new_data_and_model_geometry(self):
        before, after = sample_pair()
        after['data']['split_manifest_sha256'] = 'different'
        after['model']['head.weight'] = torch.ones(3)
        self.assertFalse(audit(before, after)['lineage_pass'])


if __name__ == '__main__':
    unittest.main()
