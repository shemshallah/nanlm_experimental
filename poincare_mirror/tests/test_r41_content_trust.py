"""R41 reciprocal-content gate: inherited logits, Adam moments, causality."""
import copy
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poincare_frontier import Config, MirrorLM, ReciprocalScaleGeometry, expand_architecture


class ReciprocalContentTrustTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_trust_reads_only_causal_prefix(self):
        torch.manual_seed(41)
        geom = ReciprocalScaleGeometry(32, 8, 8, content_trust=True)
        with torch.no_grad():
            geom.context_trust.weight.zero_()
            geom.context_trust.weight[0,0] = 1.
        ids = torch.tensor([[65, 66, 32, 67, 68, 69, 70]])
        states = torch.randn(1, 7, 8)
        states_corrupt = states.clone()
        states_corrupt[:, 5:] += 20.
        ids_corrupt = ids.clone()
        ids_corrupt[:, 5:] = 255
        p = geom.feedback_trust(geom.causal_scales(states, ids))
        q = geom.feedback_trust(geom.causal_scales(states_corrupt, ids_corrupt))
        torch.testing.assert_close(p[:, :5], q[:, :5], atol=0, rtol=0)
        self.assertGreater((p[:, 5:] - q[:, 5:]).abs().max().item(), 0.)
        self.assertTrue(torch.all((p > 0.) & (p < 2.)))

    def test_real_micro_geometry_migrates_logits_and_optimizer_exactly(self):
        torch.manual_seed(40)
        previous = Config(vocab_size=257, dim=32, layers=2, heads=4,
            pairs=4, topk_pairs=2, expert_hidden=16, context=8,
            moe_every=2, symbiotic_rank=8, hierarchy_rank=8,
            micro_meta=True, simplex_transfer=True, reciprocal_scales=True)
        old = MirrorLM(previous)
        # The open gate is important: an inactive bridge could conceal drift.
        with torch.no_grad():
            old.reciprocal.gate.fill_(.06)
        oldopt = torch.optim.AdamW(old.parameters(), lr=1e-4)
        tokens = torch.tensor([[65, 66, 32, 67, 68, 69, 70, 71]])
        old.loss_with_future(tokens)['loss'].backward()
        oldopt.step()
        old.zero_grad(set_to_none=True)
        snapshot = {'model': copy.deepcopy(old.state_dict()),
                    'optimizer': copy.deepcopy(oldopt.state_dict())}
        grown_cfg = replace(previous, reciprocal_content_trust=True)
        grown = MirrorLM(grown_cfg)
        grownopt = torch.optim.AdamW(grown.parameters(), lr=1e-4)
        expand_architecture(grown, grownopt, snapshot, previous)

        self.assertTrue(torch.equal(grown.reciprocal.context_trust.weight,
                                    torch.zeros_like(grown.reciprocal.context_trust.weight)))
        self.assertEqual(len(grownopt.param_groups), 1)
        old_names = list(dict(old.named_parameters()))
        new_names = list(dict(grown.named_parameters()))
        old_ids = oldopt.state_dict()['param_groups'][0]['params']
        new_ids = grownopt.state_dict()['param_groups'][0]['params']
        mapped_ids = dict(zip(new_names, new_ids))
        old_states = oldopt.state_dict()['state']
        new_states = grownopt.state_dict()['state']
        for name, prior_id in zip(old_names, old_ids):
            if prior_id not in old_states:
                continue
            after_state = new_states[mapped_ids[name]]
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                self.assertTrue(torch.equal(old_states[prior_id][key],after_state[key]),
                                (name,key))

        old.eval()
        grown.eval()
        with torch.no_grad():
            original = old(tokens)[0]
            migrated = grown(tokens)[0]
        self.assertTrue(torch.equal(original, migrated),
                        f'max initial logit drift {(original-migrated).abs().max()}')
        # Two independent batch elements must use their own (B,T) trust map,
        # not the flattened (B*T,1,1) microtransformer addressing shape.
        doubled = torch.cat((tokens, tokens.roll(1, dims=1)), dim=0)
        with torch.no_grad():
            self.assertTrue(torch.equal(old(doubled)[0], grown(doubled)[0]))
        # With the migrated gate moved away from zero, the correct causal
        # prefix can modulate the preexisting reciprocal output.
        with torch.no_grad():
            grown.reciprocal.context_trust.weight.zero_()
            grown.reciprocal.context_trust.weight[0,0] = 1.
            changed = grown(tokens)[0]
        self.assertGreater((migrated-changed).abs().max().item(), 0.)


if __name__ == '__main__':
    unittest.main()
