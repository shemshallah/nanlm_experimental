#!/usr/bin/env python3
"""Tests for z3.moe_vectorized against a per-expert reference implementation."""

from __future__ import annotations

import pathlib
import sys
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from moe_vectorized import VectorizedSparseMoE  # noqa: E402


class PerExpertReference(nn.Module):
    """Literal per-expert SwiGLU reference used only by the tests."""

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        topk: int,
        shared_experts: int,
        expert_hidden: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.topk = topk
        self.shared_experts = shared_experts
        self.expert_hidden = expert_hidden
        self.dropout = dropout

        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.gate_w = nn.Parameter(torch.empty(n_experts, d_model, expert_hidden))
        self.up_w = nn.Parameter(torch.empty(n_experts, d_model, expert_hidden))
        self.down_w = nn.Parameter(torch.empty(n_experts, expert_hidden, d_model))
        if shared_experts > 0:
            self.shared_gate_w = nn.Parameter(
                torch.empty(shared_experts, d_model, expert_hidden)
            )
            self.shared_up_w = nn.Parameter(
                torch.empty(shared_experts, d_model, expert_hidden)
            )
            self.shared_down_w = nn.Parameter(
                torch.empty(shared_experts, expert_hidden, d_model)
            )
        else:
            self.shared_gate_w = None
            self.shared_up_w = None
            self.shared_down_w = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.router.weight)
        for e in range(self.n_experts):
            nn.init.xavier_uniform_(self.gate_w[e])
            nn.init.xavier_uniform_(self.up_w[e])
            nn.init.xavier_uniform_(self.down_w[e])
        if self.shared_gate_w is not None:
            for s in range(self.shared_experts):
                nn.init.xavier_uniform_(self.shared_gate_w[s])
                nn.init.xavier_uniform_(self.shared_up_w[s])
                nn.init.xavier_uniform_(self.shared_down_w[s])

    def forward(self, x):
        b, t, d = x.shape
        flat = x.reshape(-1, d)
        logits = self.router(flat)
        probs = F.softmax(logits, dim=-1)
        topv, topi = torch.topk(logits, k=self.topk, dim=-1)
        weights = F.softmax(topv, dim=-1)

        out = torch.zeros_like(flat)
        for e in range(self.n_experts):
            rows, slots = (topi == e).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            xe = flat[rows]
            hidden = F.silu(xe @ self.gate_w[e]) * (xe @ self.up_w[e])
            ye = F.dropout(hidden @ self.down_w[e], p=self.dropout, training=self.training)
            out = out.index_add(0, rows, weights[rows, slots].unsqueeze(-1) * ye)

        if self.shared_gate_w is not None:
            for s in range(self.shared_experts):
                hidden = F.silu(flat @ self.shared_gate_w[s]) * (
                    flat @ self.shared_up_w[s]
                )
                shared = F.dropout(
                    hidden @ self.shared_down_w[s], p=self.dropout, training=self.training
                )
                out = out + shared

        mask = torch.zeros_like(logits).scatter_(1, topi, 1.0)
        f = mask.mean(dim=0)
        p = probs.mean(dim=0)
        load_balance = self.n_experts * (f * p).sum()
        router_z = torch.logsumexp(logits, dim=-1).pow(2).mean()
        return out.view(b, t, d), {"load_balance": load_balance, "router_z": router_z}


class TestVectorizedSparseMoE(unittest.TestCase):
    D = 8
    E = 4
    K = 2
    S = 1
    H = 16
    B = 2
    T = 3

    def _model(self, **kwargs):
        cfg = dict(
            d_model=self.D,
            n_experts=self.E,
            topk=self.K,
            shared_experts=self.S,
            expert_hidden=self.H,
            dropout=0.0,
        )
        cfg.update(kwargs)
        return VectorizedSparseMoE(**cfg)

    def _pair(self, shared_experts=None):
        shared = self.S if shared_experts is None else shared_experts
        torch.manual_seed(0)
        vec = self._model(shared_experts=shared).double().eval()
        ref = PerExpertReference(
            d_model=self.D,
            n_experts=self.E,
            topk=self.K,
            shared_experts=shared,
            expert_hidden=self.H,
            dropout=0.0,
        ).double().eval()
        ref.load_state_dict(vec.state_dict())
        return vec, ref

    def test_forward_shapes_and_aux_keys(self):
        model = self._model()
        x = torch.randn(self.B, self.T, self.D)
        out, aux = model(x)

        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, x.dtype)
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(set(aux), {"load_balance", "router_z"})
        for name, value in aux.items():
            self.assertEqual(value.shape, (), f"{name} should be scalar")
            self.assertTrue(torch.isfinite(value), f"{name} should be finite")
        self.assertGreaterEqual(aux["load_balance"].item(), 0.0)
        self.assertGreaterEqual(aux["router_z"].item(), 0.0)

    def test_matches_per_expert_reference(self):
        vec, ref = self._pair()
        x = torch.randn(self.B, self.T, self.D, dtype=torch.float64)
        out_v, aux_v = vec(x)
        out_r, aux_r = ref(x)

        torch.testing.assert_close(out_v, out_r, atol=1e-5, rtol=1e-5)
        for key in ("load_balance", "router_z"):
            torch.testing.assert_close(aux_v[key], aux_r[key], atol=1e-5, rtol=1e-5)

    def test_half_precision_forward_and_grad(self):
        model = self._model().half()
        x = torch.randn(
            self.B, self.T, self.D, dtype=torch.float16, requires_grad=True
        )
        out, aux = model(x)
        self.assertEqual(out.dtype, torch.float16)
        self.assertTrue(torch.isfinite(out).all())
        out.float().sum().backward()
        self.assertIsNotNone(model.gate_w.grad)
        self.assertTrue(torch.isfinite(model.gate_w.grad).all())

    def test_group_and_row_chunk_dispatch_matches_reference(self):
        # Force several expert groups and several row chunks, including a
        # final partial chunk.  This is the dispatch path used to avoid the
        # 8.6 GB per-token weight gather at the T4 profile.
        d, e, k, h, s = 8, 5, 3, 16, 1
        torch.manual_seed(7)
        vec = VectorizedSparseMoE(
            d_model=d, n_experts=e, topk=k, shared_experts=s,
            expert_hidden=h, dropout=0.0, group_size=2, max_rows=7,
        ).double().eval()
        ref = PerExpertReference(
            d_model=d, n_experts=e, topk=k, shared_experts=s,
            expert_hidden=h, dropout=0.0,
        ).double().eval()
        ref.load_state_dict(vec.state_dict())
        x = torch.randn(3, 11, d, dtype=torch.float64)
        out_v, aux_v = vec(x)
        out_r, aux_r = ref(x)
        torch.testing.assert_close(out_v, out_r, atol=1e-5, rtol=1e-5)
        for key in ("load_balance", "router_z"):
            torch.testing.assert_close(aux_v[key], aux_r[key], atol=1e-5, rtol=1e-5)

    def test_gradients_match_per_expert_reference(self):
        vec, ref = self._pair()
        x_v = torch.randn(
            self.B, self.T, self.D, dtype=torch.float64, requires_grad=True
        )
        x_r = x_v.detach().clone().requires_grad_(True)

        out_v, aux_v = vec(x_v)
        out_r, aux_r = ref(x_r)

        torch.manual_seed(1234)
        gout = torch.randn_like(out_v)
        loss_v = (
            (out_v * gout).sum()
            + 0.1 * aux_v["load_balance"]
            + 0.01 * aux_v["router_z"]
        )
        loss_r = (
            (out_r * gout).sum()
            + 0.1 * aux_r["load_balance"]
            + 0.01 * aux_r["router_z"]
        )
        loss_v.backward()
        loss_r.backward()

        torch.testing.assert_close(x_v.grad, x_r.grad, atol=1e-5, rtol=1e-5)
        ref_params = dict(ref.named_parameters())
        vec_params = dict(vec.named_parameters())
        self.assertEqual(set(vec_params), set(ref_params))
        for name, param in vec_params.items():
            self.assertIsNotNone(param.grad, f"missing grad for {name}")
            torch.testing.assert_close(
                param.grad, ref_params[name].grad, atol=1e-5, rtol=1e-5
            )

    def test_route_topk(self):
        model = self._model().eval()
        flat = torch.randn(16, self.D)
        logits, probs, topv, topi, weights = model.route(flat)

        self.assertEqual(logits.shape, (16, self.E))
        self.assertEqual(topi.shape, (16, self.K))
        self.assertEqual(topv.shape, (16, self.K))
        self.assertEqual(weights.shape, (16, self.K))

        torch.testing.assert_close(probs, F.softmax(logits, dim=-1))
        torch.testing.assert_close(topv, logits.gather(1, topi))
        torch.testing.assert_close(weights, F.softmax(topv, dim=-1))
        torch.testing.assert_close(
            weights.sum(dim=-1), torch.ones(16), atol=1e-6, rtol=1e-6
        )
        self.assertTrue((weights >= 0).all())
        self.assertTrue((topi >= 0).all() and (topi < self.E).all())

        expected = torch.zeros_like(logits).scatter(
            1, logits.topk(self.K, dim=-1).indices, 1.0
        )
        actual = torch.zeros_like(logits).scatter_(1, topi, 1.0)
        self.assertTrue(torch.equal(expected, actual))

    def test_load_balance_and_router_z(self):
        model = self._model().eval()
        x = torch.randn(self.B, self.T, self.D)
        _, aux = model(x)

        flat = x.reshape(-1, self.D)
        logits, probs, _, topi, _ = model.route(flat)
        mask = torch.zeros_like(logits).scatter_(1, topi, 1.0)
        f = mask.mean(dim=0)
        p = probs.mean(dim=0)

        expected_lb = self.E * (f * p).sum()
        expected_z = torch.logsumexp(logits, dim=-1).pow(2).mean()
        torch.testing.assert_close(aux["load_balance"], expected_lb)
        torch.testing.assert_close(aux["router_z"], expected_z)

        torch.testing.assert_close(
            f.sum(), torch.tensor(float(self.K)), atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            p.sum(), torch.tensor(1.0), atol=1e-6, rtol=1e-6
        )

    def test_finite_grads(self):
        model = self._model()
        x = torch.randn(self.B, self.T, self.D, requires_grad=True)
        out, aux = model(x)
        loss = (
            out.pow(2).mean()
            + 0.05 * aux["load_balance"]
            + 0.01 * aux["router_z"]
        )
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0.0)

        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"missing grad for {name}")
            self.assertTrue(torch.isfinite(param.grad).all(), f"non-finite grad {name}")

        self.assertGreater(model.router.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.shared_gate_w.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.gate_w.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.up_w.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.down_w.grad.abs().sum().item(), 0.0)

    def test_shared_experts_disabled(self):
        vec, ref = self._pair(shared_experts=0)
        self.assertIsNone(vec.shared_gate_w)
        self.assertFalse(any(k.startswith("shared") for k in vec.state_dict()))

        x = torch.randn(self.B, self.T, self.D, dtype=torch.float64)
        out_v, aux_v = vec(x)
        out_r, aux_r = ref(x)
        torch.testing.assert_close(out_v, out_r, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            aux_v["load_balance"], aux_r["load_balance"], atol=1e-5, rtol=1e-5
        )

    def test_flattened_shape_invariance(self):
        model = self._model().eval()
        x = torch.randn(2, 3, self.D)
        out_bt, aux_bt = model(x)
        out_flat, aux_flat = model(x.reshape(6, 1, self.D))

        torch.testing.assert_close(out_bt.reshape(6, 1, self.D), out_flat)
        torch.testing.assert_close(aux_bt["load_balance"], aux_flat["load_balance"])
        torch.testing.assert_close(aux_bt["router_z"], aux_flat["router_z"])

    def test_config_validation_and_from_config(self):
        with self.assertRaises(ValueError):
            self._model(topk=0)
        with self.assertRaises(ValueError):
            self._model(topk=self.E + 1)
        with self.assertRaises(ValueError):
            self._model(shared_experts=-1)
        with self.assertRaises(ValueError):
            self._model(d_model=0)
        with self.assertRaises(ValueError):
            self._model(expert_hidden=0)
        with self.assertRaises(ValueError):
            model = self._model()
            model(torch.randn(2, 3, self.D + 1))

        cfg = SimpleNamespace(
            d_model=self.D,
            n_experts=self.E,
            topk=self.K,
            shared_experts=self.S,
            expert_hidden=self.H,
            dropout=0.0,
        )
        model = VectorizedSparseMoE.from_config(cfg)
        self.assertEqual(model.d_model, self.D)
        self.assertEqual(model.n_experts, self.E)
        self.assertEqual(model.topk, self.K)
        self.assertEqual(model.shared_experts, self.S)
        self.assertEqual(model.expert_hidden, self.H)
        self.assertEqual(model.dropout, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
