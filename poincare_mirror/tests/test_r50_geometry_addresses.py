"""R50 tests: aligned address controls preserve the entire inherited state."""
import sys
from pathlib import Path
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from probe_r50_geometry_addresses import (MODES, address_ablation,
                                           chain_relabelled, fixed_permutations)
from poincare_frontier import SparseSimplexTransfer


class FakeGeometry(nn.Module):
    def __init__(self):
        super().__init__()
        self.simplex = SparseSimplexTransfer(8)
        self.triangle_table = nn.Embedding(8192, 8)


class FakeReciprocal(nn.Module):
    def __init__(self):
        super().__init__()
        self.enabled = True


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.reciprocal = FakeReciprocal()
        self.geometry = FakeGeometry()


class AddressControls(unittest.TestCase):
    def setUp(self):
        self.perms = fixed_permutations(2026)
        self.model = FakeModel()

    def test_seeded_bijections_are_reproducible_and_nontrivial(self):
        other = fixed_permutations(2026)
        for key, perm in self.perms.items():
            self.assertTrue(torch.equal(perm, other[key]))
            self.assertTrue(torch.equal(perm.sort().values, torch.arange(len(perm))))
            self.assertFalse(torch.equal(perm, torch.arange(len(perm))))

    def test_relabelled_boundary_is_a_valid_chain_complex(self):
        s = self.model.geometry.simplex
        d = chain_relabelled(s.d1, s.d2, s.d3, self.perms)
        self.assertTrue(torch.equal(d['d2'] @ d['d1'], torch.zeros(56, 8)))
        self.assertTrue(torch.equal(d['d3'] @ d['d2'], torch.zeros(70, 28)))
        for name in ('d1', 'd2', 'd3'):
            self.assertFalse(torch.equal(d[name], getattr(s, name)))
            self.assertTrue(torch.equal(d[name].abs().sum(-1),
                                        getattr(s, name).abs().sum(-1)))

    def test_every_mode_restores_exact_model_and_buffers(self):
        before = {name: x.clone() for name, x in self.model.state_dict().items()}
        s = self.model.geometry.simplex
        original_d = [getattr(s, name).clone() for name in ('d1', 'd2', 'd3')]
        for mode in MODES:
            with address_ablation(self.model, mode, self.perms):
                if mode == 'reciprocal_disabled':
                    self.assertFalse(self.model.reciprocal.enabled)
            self.assertTrue(self.model.reciprocal.enabled)
            for name, tensor in before.items():
                self.assertTrue(torch.equal(tensor, self.model.state_dict()[name]))
            for name, value in zip(('d1', 'd2', 'd3'), original_d):
                self.assertTrue(torch.equal(getattr(s, name), value))
            for emb in (s.pair_shared, s.pair_anti, self.model.geometry.triangle_table):
                self.assertFalse(emb._forward_pre_hooks)

    def test_pair_and_triangle_remapping_is_applied_without_weights_mutation(self):
        s = self.model.geometry.simplex
        edge_idx = torch.tensor([0, 1, 32639])
        face_idx = torch.tensor([1, 17, 8191])
        old_pairs = s.pair_shared(edge_idx).clone()
        old_faces = self.model.geometry.triangle_table(face_idx).clone()
        with address_ablation(self.model, 'pair_address_scrambled', self.perms):
            self.assertTrue(torch.equal(s.pair_shared(edge_idx),
                                        s.pair_shared.weight[self.perms['pair_addresses'][edge_idx]]))
            self.assertTrue(torch.equal(s.pair_anti(edge_idx),
                                        s.pair_anti.weight[self.perms['pair_addresses'][edge_idx]]))
        with address_ablation(self.model, 'triangle_address_scrambled', self.perms):
            self.assertTrue(torch.equal(self.model.geometry.triangle_table(face_idx),
                     self.model.geometry.triangle_table.weight[self.perms['triangle_hashes'][face_idx]]))
        self.assertTrue(torch.equal(s.pair_shared(edge_idx), old_pairs))
        self.assertTrue(torch.equal(self.model.geometry.triangle_table(face_idx), old_faces))

    def test_exception_restores_everything(self):
        s = self.model.geometry.simplex
        previous = s.d1.clone()
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            with address_ablation(self.model, 'all_scrambled', self.perms):
                self.assertFalse(torch.equal(s.d1, previous))
                raise RuntimeError('test failure')
        self.assertTrue(torch.equal(s.d1, previous))
        self.assertFalse(s.pair_shared._forward_pre_hooks)
        self.assertFalse(self.model.geometry.triangle_table._forward_pre_hooks)
        self.assertTrue(self.model.reciprocal.enabled)


if __name__ == '__main__':
    unittest.main()
