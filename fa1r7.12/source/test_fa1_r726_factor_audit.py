#!/usr/bin/env python3
"""R7.26 synthetic factor-mask, book-bootstrap and restoration regression."""
from types import SimpleNamespace
import math
import torch
from fa1_r723_diamond import CombinatorialDiamond
from fa1_r724_geometry import OrientedSimplicialGeometry
from fa1_r726_factor_audit import FACTORS, audit, mask


class SyntheticNLL:
    """A deterministic probe; NOT real model validation or training evidence."""
    def __init__(self, diamond):
        self.diamond = diamond

    def eval(self):
        self.diamond.eval()

    def joint_nll(self, data, **_kwargs):
        d, g = self.diamond, self.diamond.geometry
        vertices = data['vertex_ids'].long()
        tri = (257 * vertices[:, 0] + 3 * vertices[:, 1] + vertices[:, 2]) % 2763520
        even = g.face_plus(tri).squeeze(-1)
        odd = g.face_minus(tri).squeeze(-1)
        edges = torch.nn.functional.one_hot(vertices[:, 0] % 64, 64).float()
        pair_even = g.edge_plus(edges).mean(-1)
        pair_odd = g.edge_minus(edges).mean(-1)
        connection = g.edge_phase(edges).squeeze(-1)
        curvature = odd + connection
        volume = (vertices[:, 3].float() - vertices[:, 4].float()) / 256.
        tetra = g.tetra_net(torch.stack((curvature, curvature.sin(), curvature.cos(),
                                        volume, volume.square(), odd.abs(), odd.square()), -1)).mean(-1)
        geometry = (torch.tanh(g.edge_gain) * (pair_even + pair_odd + connection)
                    + torch.tanh(g.face_gain) * (even + odd)
                    + torch.tanh(g.tetra_gain) * tetra
                    + torch.tanh(g.meta_gate) * tetra * .5)
        value = 5. + .05 * d.triple_code(tri).mean(-1) + .03 * geometry
        return value, None, {'synthetic': True}


def test():
    torch.manual_seed(726)
    torch.set_num_threads(2)
    d = CombinatorialDiamond()
    d.geometry = OrientedSimplicialGeometry()
    g = d.geometry
    model = SimpleNamespace(diamond=d)
    tri = torch.tensor([10, 12345, 456789])
    edges = torch.randn(3, 64)
    samples = (
        ('r723_triangle_address', d.triple_code, tri),
        ('face_splus_address', g.face_plus, tri),
        ('face_sminus_address', g.face_minus, tri),
        ('edge_splus', g.edge_plus, edges),
        ('edge_sminus', g.edge_minus, edges),
        ('edge_connection', g.edge_phase, edges),
    )
    for factor, module, values in samples:
        original = module(values).detach().clone()
        with mask(model, factor):
            assert torch.equal(module(values), torch.zeros_like(original)), factor
        assert torch.equal(module(values), original), ('restoration', factor)
        assert not module._forward_hooks
    with mask(model, 'edge_odd_joint'):
        assert torch.count_nonzero(g.edge_minus(edges)) == 0
        assert torch.count_nonzero(g.edge_phase(edges)) == 0
    assert not g.edge_minus._forward_hooks and not g.edge_phase._forward_hooks

    features = torch.randn(2, 70, 7)
    for factor, removed in [('tetra_closure', (0, 1, 2)), ('tetra_volume', (3, 4))]:
        seen = []
        with mask(model, factor):
            handle = g.tetra_net.register_forward_pre_hook(
                lambda _module, args: seen.append(args[0].clone()))
            try:
                g.tetra_net(features)
            finally:
                handle.remove()
        changed = seen[0]
        for col in range(7):
            wanted = ((torch.ones_like(features[..., col]) if col == 2 else
                       torch.zeros_like(features[..., col])) if col in removed
                      else features[..., col])
            assert torch.equal(changed[..., col], wanted), (factor, col)
        assert not g.tetra_net._forward_pre_hooks

    # Edge curl alone is exact; independent oriented face code is not exact.
    phases = torch.randn(2, 28)
    ed = g.triangle_edges
    dA = phases[:, ed[:, 0]] + phases[:, ed[:, 2]] - phases[:, ed[:, 1]]
    assert (dA[:, g.tetra_faces] * g.face_signs).sum(-1).abs().max() < 1e-6
    face_code = g.face_minus(torch.arange(56)).squeeze(-1)
    assert ((dA + face_code)[:, g.tetra_faces] * g.face_signs).sum(-1).abs().max() > 1e-5

    with torch.no_grad():
        d.gate.fill_(.1)
        for j, name in enumerate(('edge_gain', 'face_gain', 'tetra_gain', 'meta_gate')):
            getattr(g, name).fill_(.01 + .01 * j)
    old_gates = [getattr(g, name).detach().clone() for name in
                 ('edge_gain', 'face_gain', 'tetra_gain', 'meta_gate')]
    n = 6
    data = {'book_ids': torch.tensor([0, 0, 1, 1, 2, 2]), 'target': torch.arange(n),
            'hidden': torch.zeros(n, 96), 'near_features': torch.zeros(n, 60),
            'far_features': torch.zeros(n, 32),
            'memory_modal_ids': torch.zeros(n, 12, dtype=torch.long),
            'memory_modal_mask': torch.zeros(n, 12, dtype=torch.bool),
            'vertex_ids': torch.stack([torch.randperm(256)[:8] for _ in range(n)]),
            'word_ids': torch.zeros(n, 32, dtype=torch.long),
            'byte_ids': torch.zeros(n, 128, dtype=torch.long),
            'sentence_ids': torch.zeros(n, 8, dtype=torch.long)}
    report = audit(SyntheticNLL(d), data)
    assert report['all_masks_restored'] and report['books'] == 3
    assert report['foreign_donors'] == n
    assert len(report['removal_penalties']) == len(FACTORS) + 4
    assert all(math.isfinite(x['mean_nats_per_vocab_word']) for x in report['removal_penalties'].values())
    assert all(torch.equal(getattr(g, name), value) for name, value in zip(
        ('edge_gain', 'face_gain', 'tetra_gain', 'meta_gate'), old_gates))
    assert report['geometry_gain']['book_cluster_ci']['books'] == 3
    print('PASS: nine independent factors; S+/S- and non-flat closure; exact hook/gate restoration')
    print('PASS: synthetic full factor audit, foreign-book donors, interaction and book CIs')


if __name__ == '__main__':
    test()
