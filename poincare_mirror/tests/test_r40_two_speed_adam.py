"""R40 geometry/backbone update tests; no alternative optimizer or model."""
import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poincare_frontier import (
    attenuate_inherited_adam_step,
    inherited_step_parameters,
    is_attached_geometry_parameter,
)


class TwoSpeedAdamTests(unittest.TestCase):
    def test_parameter_partition_matches_geometry_warmup(self):
        positive = ('reciprocal.r', 'hierarchy.a',
                    'blocks.1.ff.symbiotic.gain')
        negative = ('blocks.1.ff.router.weight', 'token_embedding.weight',
                    'forecast.weight', 'norm.weight')
        self.assertTrue(all(is_attached_geometry_parameter(n) for n in positive))
        self.assertTrue(all(not is_attached_geometry_parameter(n) for n in negative))

        class NamedModel:
            def named_parameters(self):
                return [(n, torch.nn.Parameter(torch.ones(1)))
                        for n in positive + negative]

        self.assertEqual(len(inherited_step_parameters(NamedModel())), len(negative))

    def test_adam_moments_are_identical_at_both_step_rates(self):
        def run(scale):
            trunk = torch.nn.Parameter(torch.tensor([2., 3.]))
            geometry = torch.nn.Parameter(torch.tensor([4., 5.]))
            optimizer = torch.optim.AdamW([trunk, geometry], lr=.01, weight_decay=.1)
            before_trunk = trunk.detach().clone()
            before_geometry = geometry.detach().clone()
            (trunk.square().sum() + 3. * geometry.square().sum()).backward()
            optimizer.step()
            attenuate_inherited_adam_step([trunk], [before_trunk], scale)
            return trunk, geometry, before_trunk, before_geometry, optimizer

        full = run(1.)
        slowed = run(.1)
        torch.testing.assert_close(slowed[0].detach(),
            full[2] + .1 * (full[0].detach() - full[2]), atol=2e-7, rtol=1e-6)
        torch.testing.assert_close(slowed[1].detach(), full[1].detach())
        for idx in (0, 1):
            a, b = full[4].state_dict()['state'][idx], slowed[4].state_dict()['state'][idx]
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                self.assertTrue(torch.equal(a[key], b[key]), (idx, key))
        self.assertEqual(len(slowed[4].param_groups), 1)

    def test_zero_scale_preserves_inherited_weight_but_not_its_adam_history(self):
        p = torch.nn.Parameter(torch.tensor([1.]))
        opt = torch.optim.AdamW([p], lr=.1)
        before = p.detach().clone()
        p.square().sum().backward()
        opt.step()
        attenuate_inherited_adam_step([p], [before], 0.)
        self.assertTrue(torch.equal(p.detach(), before))
        self.assertIn('exp_avg', opt.state[p])
        self.assertEqual(opt.state[p]['step'].item(), 1)

    def test_invalid_scales_and_snapshot_shapes_rejected(self):
        p = torch.nn.Parameter(torch.ones(2))
        for scale in (math.nan, math.inf, -.1, 1.1):
            with self.assertRaisesRegex(ValueError, 'scale'):
                attenuate_inherited_adam_step([p], [p.detach().clone()], scale)
        with self.assertRaisesRegex(ValueError, 'count'):
            attenuate_inherited_adam_step([p], [], .5)
        with self.assertRaisesRegex(ValueError, 'shape'):
            attenuate_inherited_adam_step([p], [torch.ones(3)], .5)


if __name__ == '__main__':
    unittest.main()
