#!/usr/bin/env python3
"""Integration checks for the assembled v3 model.

These tests are intentionally small.  They verify that the long-context RoPE,
vectorized MoE, and optional memory path are actually wired into V3LM, not
just present as standalone modules.
"""

from __future__ import annotations

import unittest

import torch

try:
    from .moe_vectorized import VectorizedSparseMoE
    from .v3_model import Block, SparseMoE, V3Config, V3LM
except ImportError:  # script execution from z3/
    from moe_vectorized import VectorizedSparseMoE
    from v3_model import Block, SparseMoE, V3Config, V3LM


class V3IntegrationTest(unittest.TestCase):
    def test_default_parameter_count_is_unchanged(self):
        cfg = V3Config()
        model = V3LM(cfg)
        self.assertGreater(model.parameter_count(), 250_000_000)

    def test_auto_rope_selects_none_at_1k_and_yarn_at_8k(self):
        short = V3Config(
            vocab_size=64, ctx=1024, max_ctx=8192, d_model=32,
            n_layers=1, n_heads=4, swiglu=64, window=256,
            global_every=1, moe_layers=0, n_experts=4, topk=2,
            shared_experts=0, expert_hidden=64, mtp=1,
        )
        short_model = V3LM(short)
        self.assertEqual(short_model.blocks[0].attn.rope.scaling, "none")

        long = V3Config(
            vocab_size=64, ctx=8192, max_ctx=8192, d_model=32,
            n_layers=1, n_heads=4, swiglu=64, window=256,
            global_every=1, moe_layers=0, n_experts=4, topk=2,
            shared_experts=0, expert_hidden=64, mtp=1,
        )
        long_model = V3LM(long)
        rope = long_model.blocks[0].attn.rope
        self.assertEqual(rope.scaling, "yarn")
        self.assertEqual(rope.max_context, 8192)

        # Exercise the full 8k position range directly; a full 8k model
        # forward is much heavier and is covered by the Colab smoke.
        x = torch.randn(1, long.n_heads, 8192, long.d_model // long.n_heads)
        y = rope(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertTrue(torch.isfinite(y).all())

    def test_ntk_mode_constructs(self):
        cfg = V3Config(
            vocab_size=64, ctx=8192, max_ctx=8192, d_model=32,
            n_layers=1, n_heads=4, swiglu=64, window=256,
            global_every=1, moe_layers=0, n_experts=4, topk=2,
            shared_experts=0, expert_hidden=64, mtp=1,
            rope_scaling="ntk",
        )
        model = V3LM(cfg)
        self.assertEqual(model.blocks[0].attn.rope.scaling, "ntk")

    def test_factorized_tied_embedding_reduces_params_and_runs(self):
        base = dict(
            vocab_size=512, ctx=32, max_ctx=32, d_model=64,
            n_layers=4, n_heads=8, swiglu=128, window=16,
            global_every=2, moe_layers=2, n_experts=4, topk=2,
            shared_experts=1, expert_hidden=128, mtp=3,
            mtp_block=True, mtp_block_experts=4,
            mtp_block_hidden=64, mtp_aux_layers=(3,),
            dropout=0.0, use_memory=False,
        )
        full = V3LM(
            V3Config(
                **base,
                factorized_embedding=False,
                mtp_tie_heads=False,
            )
        )
        compact = V3LM(
            V3Config(
                **base,
                factorized_embedding=True,
                embedding_rank=16,
                mtp_tie_heads=True,
            )
        )
        self.assertLess(compact.parameter_count(), full.parameter_count())
        self.assertIsNotNone(compact.tok_up)
        self.assertEqual(compact.n_mtp_heads, 2)
        self.assertEqual(len(compact.mtp_heads), 2)
        self.assertEqual(compact.mtp_heads[0].out_features, 16)
        probe = torch.randn(2, 4, 64)
        torch.testing.assert_close(
            compact.mtp_head_logits(compact.mtp_heads[0], probe),
            compact.project_logits(probe),
        )

        tokens = torch.randint(0, 512, (2, 32))
        logits, mtp, aux = compact(tokens)
        self.assertEqual(tuple(logits.shape), (2, 32, 512))
        self.assertEqual(len(mtp), 2)
        self.assertEqual(tuple(mtp[0].shape), (2, 32, 512))
        self.assertIn("mtp_aux_logits", aux)
        self.assertIn("mtp_load_balance", aux)
        self.assertTrue(torch.isfinite(aux["mtp_load_balance"]))
        self.assertTrue(torch.isfinite(logits).all())

        loss = logits.float().log_softmax(-1).gather(
            -1, tokens[..., None]
        ).mean()
        for branch_logits in mtp:
            loss = loss + branch_logits.float().log_softmax(-1).gather(
                -1, tokens[..., None]
            ).mean()
        loss.backward()
        self.assertIsNotNone(compact.tok_up.weight.grad)
        self.assertTrue(torch.isfinite(compact.tok_up.weight.grad).all())

    def test_hierarchical_router_feedback_through_blocks(self):
        cfg = V3Config(
            vocab_size=64, ctx=32, max_ctx=32, d_model=32,
            n_layers=4, n_heads=4, swiglu=64, window=16,
            global_every=2, moe_layers=2, n_experts=8, topk=2,
            shared_experts=1, expert_hidden=64, mtp=1,
            moe_hierarchical=True, moe_branches=4,
            moe_context_router=True, moe_route_feedback=True,
            dropout=0.0, use_memory=False,
        )
        model = V3LM(cfg)
        moe = model.blocks[-1].mlp.inner
        self.assertEqual(moe.branches, 4)
        self.assertEqual(moe.experts_per_branch, 2)
        tokens = torch.randint(0, 64, (2, 32))
        logits, _mtp, _aux = model(tokens)
        self.assertTrue(torch.isfinite(logits).all())
        logits.float().logsumexp(dim=-1).mean().backward()
        self.assertIsNotNone(model.blocks[-1].route_to_state.weight.grad)
        self.assertTrue(
            torch.isfinite(model.blocks[-1].route_to_state.weight.grad).all()
        )

        x = model.embed(tokens)
        ctx = model.route_init.unsqueeze(0).expand(x.shape[0], -1)
        _x, _lb, _z, next_ctx, _aux = model.blocks[0](x, ctx)
        self.assertGreater(float(next_ctx.abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(next_ctx).all())

        # The router context must reach the last MoE block and the MTP path
        # must be able to condition on it.
        self.assertTrue(model.blocks[-1].route_feedback)
        self.assertEqual(
            model.blocks[-1].mlp.inner.expert_context.weight.shape[1],
            cfg.d_model,
        )

    def test_learned_router_mix_starts_at_identity_and_gets_gradient(self):
        common = dict(
            vocab_size=64, ctx=16, max_ctx=16, d_model=32,
            n_layers=2, n_heads=4, n_kv_heads=2, swiglu=64,
            window=8, global_every=2, moe_layers=1, n_experts=4,
            topk=2, shared_experts=1, expert_hidden=64,
            moe_hierarchical=True, moe_context_router=True,
            moe_hyperdimensional=True, moe_route_feedback=True,
            dropout=0.0, use_memory=False, mtp=1, mtp_block=False,
        )
        fixed = Block(V3Config(**common, learned_router_mix=False), 1)
        mixed = Block(V3Config(**common, learned_router_mix=True), 1)
        missing, unexpected = mixed.load_state_dict(
            fixed.state_dict(), strict=False
        )
        self.assertEqual(missing, ["router_mix"])
        self.assertEqual(unexpected, [])
        x = torch.randn(2, 5, 32)
        route = torch.randn(2, 32)
        loop = torch.randn(2, 32)
        token = torch.randn(2, 5, 32)
        fixed.eval()
        mixed.eval()
        with torch.no_grad():
            fixed_out = fixed(x, route, loop, token)
            mixed_out = mixed(x, route, loop, token)
        torch.testing.assert_close(fixed_out[0], mixed_out[0])
        torch.testing.assert_close(
            torch.nn.functional.softplus(mixed.router_mix),
            torch.ones(3),
        )
        mixed.train()
        out = mixed(x, route, loop, token)
        out[0].square().mean().backward()
        self.assertIsNotNone(mixed.router_mix.grad)
        self.assertTrue(torch.isfinite(mixed.router_mix.grad).all())

    def test_hard_object_id_embedding_is_addressable(self):
        cfg = V3Config(
            vocab_size=64, ctx=16, max_ctx=16, d_model=32,
            n_layers=1, n_heads=4, n_kv_heads=2, swiglu=64,
            window=8, global_every=1, moe_layers=1, n_experts=4,
            topk=2, shared_experts=1, expert_hidden=64,
            dropout=0.0, use_memory=False, mtp=1, mtp_block=False,
            use_concept_loop=True, concept_loop_dim=16,
            concept_top_pairs=4, concept_top_triples=2,
            object_id_embedding=True, object_id_dim=8,
        )
        model = V3LM(cfg)
        embedding = model.concept_loop.object_id_embedding
        self.assertEqual(embedding.num_embeddings, 32640)
        tokens = torch.randint(0, 64, (2, 16))
        logits, _mtp, _aux = model(tokens)
        self.assertTrue(torch.isfinite(logits).all())
        logits.float().logsumexp(-1).mean().backward()
        self.assertIsNotNone(embedding.weight.grad)
        self.assertIsNotNone(
            model.concept_loop.object_id_to_state.weight.grad
        )

    def test_full_byte_probe_and_cross_resolution_shapes(self):
        cfg = V3Config(
            vocab_size=64, ctx=16, max_ctx=8192, d_model=32,
            n_layers=1, n_heads=4, n_kv_heads=2, swiglu=64,
            window=8, global_every=1, moe_layers=0, n_experts=4,
            topk=2, shared_experts=0, expert_hidden=64,
            dropout=0.0, use_memory=False, mtp=2, mtp_block=True,
            mtp_block_experts=4, mtp_block_hidden=64,
            mtp_aux_layers=(1,), byte_probe=True, byte_probe_full=True,
            word_probe_size=16, probe_detach=True,
        )
        model = V3LM(cfg)
        tokens = torch.randint(0, 64, (2, 16))
        logits, _mtp, aux = model(tokens)
        self.assertEqual(tuple(logits.shape), (2, 16, 64))
        self.assertEqual(
            tuple(aux["byte_sequence_logits"].shape),
            (2, 16, 48, 256),
        )
        aux["byte_sequence_logits"].float().logsumexp(-1).mean().backward()
        self.assertIsNotNone(model.byte_sequence_probe.weight.grad)
        self.assertIsNone(model.tok.weight.grad)

    def test_speculative_mtp_generation(self):
        cfg = V3Config(
            vocab_size=64, ctx=16, max_ctx=64, d_model=32,
            n_layers=2, n_heads=4, n_kv_heads=2, swiglu=64,
            window=8, global_every=2, moe_layers=1, n_experts=4,
            topk=2, shared_experts=1, expert_hidden=64,
            dropout=0.0, use_memory=False, mtp=2, mtp_block=True,
            mtp_block_experts=4, mtp_block_hidden=64,
            mtp_aux_layers=(1,),
        )
        model = V3LM(cfg).eval()
        tokens = torch.randint(0, 64, (1, 8))
        ids, stats = model.generate_speculative(
            tokens, max_new_tokens=5, temperature=0.0, seed=3
        )
        self.assertEqual(ids.size(1) - tokens.size(1), 5)
        self.assertEqual(stats["drafted"], 2)
        self.assertGreaterEqual(stats["forwards"], 3)
        self.assertGreaterEqual(stats["acceptance"], 0.0)
        self.assertLessEqual(stats["acceptance"], 1.0)

    def test_long_context_forward_path(self):
        cfg = V3Config(
            vocab_size=64, ctx=512, max_ctx=8192, d_model=32,
            n_layers=1, n_heads=4, n_kv_heads=2, swiglu=64,
            window=256, global_every=2, moe_layers=0, n_experts=4,
            topk=2, shared_experts=0, expert_hidden=64,
            dropout=0.0, use_memory=False, mtp=1, use_concept_loop=False,
        )
        model = V3LM(cfg).eval()
        tokens = torch.randint(0, 64, (1, 512))
        with torch.no_grad():
            logits, _mtp, _aux = model(tokens)
        self.assertEqual(tuple(logits.shape), (1, 512, 64))
        self.assertTrue(torch.isfinite(logits).all())

    def test_detached_probes_do_not_backprop_to_trunk(self):
        cfg = V3Config(
            vocab_size=64, ctx=16, max_ctx=16, d_model=32,
            n_layers=1, n_heads=4, n_kv_heads=2, swiglu=64,
            window=8, global_every=1, moe_layers=0, n_experts=4,
            topk=2, shared_experts=0, expert_hidden=64,
            dropout=0.0, use_memory=False, mtp=1, mtp_block=False,
            byte_probe=True, word_probe_size=16, probe_detach=True,
        )
        model = V3LM(cfg)
        tokens = torch.randint(0, 64, (2, 16))
        _logits, _mtp, aux = model(tokens)
        aux["byte_probe_logits"].float().logsumexp(-1).mean().backward()
        self.assertIsNotNone(model.byte_probe.weight.grad)
        self.assertIsNone(model.tok.weight.grad)
        self.assertIsNone(model.blocks[0].attn.qkv.weight.grad)
        self.assertIsNone(model.tok_up.weight.grad)

    def test_sparse_moe_uses_vectorized_inner(self):
        cfg = V3Config(
            vocab_size=64, ctx=32, max_ctx=32, d_model=32,
            n_layers=2, n_heads=4, swiglu=64, window=16,
            global_every=2, moe_layers=1, n_experts=4, topk=2,
            shared_experts=1, expert_hidden=64, mtp=1,
        )
        model = V3LM(cfg)
        moe = model.blocks[-1].mlp
        self.assertIsInstance(moe, SparseMoE)
        self.assertIsInstance(moe.inner, VectorizedSparseMoE)
        self.assertFalse(hasattr(moe, "experts"))

        x = torch.randint(0, 64, (2, 32))
        logits, mtp, aux = model(x)
        self.assertEqual(tuple(logits.shape), (2, 32, 64))
        self.assertIn("load_balance", aux)
        self.assertIn("router_z", aux)

    def test_gqa_forward_and_kv_projection(self):
        base = dict(
            vocab_size=64, ctx=32, max_ctx=32, d_model=64,
            n_layers=1, n_heads=8, swiglu=128, window=16,
            global_every=1, moe_layers=0, n_experts=4, topk=2,
            shared_experts=0, expert_hidden=128, mtp=1, dropout=0.0,
        )
        mha = V3LM(V3Config(**base))
        gqa = V3LM(V3Config(**base, n_kv_heads=2))
        self.assertEqual(gqa.blocks[0].attn.qkv.out_features, 64 + 2 * (2 * 8))
        self.assertLess(gqa.parameter_count(), mha.parameter_count())
        x = torch.randint(0, 64, (2, 32))
        y, _, _ = gqa(x)
        self.assertEqual(tuple(y.shape), (2, 32, 64))
        self.assertTrue(torch.isfinite(y).all())
        manual = V3LM(
            V3Config(**base, n_kv_heads=2, force_manual_attention=True)
        ).eval()
        manual.load_state_dict(gqa.state_dict())
        with torch.no_grad():
            y_manual, _, _ = manual(x)
        torch.testing.assert_close(y_manual, y, atol=1e-5, rtol=1e-5)

    def test_attention_gate_forward(self):
        base = dict(
            vocab_size=64, ctx=32, max_ctx=32, d_model=64,
            n_layers=1, n_heads=8, swiglu=128, window=16,
            global_every=1, moe_layers=0, n_experts=4, topk=2,
            shared_experts=0, expert_hidden=128, mtp=1, dropout=0.0,
        )
        plain = V3LM(V3Config(**base, attn_gate=False))
        gated = V3LM(V3Config(**base, attn_gate=True))
        self.assertGreater(gated.parameter_count(), plain.parameter_count())
        x = torch.randint(0, 64, (2, 32))
        y, _, _ = gated(x)
        self.assertTrue(torch.isfinite(y).all())

    def test_manual_attention_matches_sdpa(self):
        base = dict(
            vocab_size=64, ctx=32, max_ctx=32, d_model=32,
            n_layers=1, n_heads=4, swiglu=64, window=16,
            global_every=1, moe_layers=0, n_experts=4, topk=2,
            shared_experts=0, expert_hidden=64, mtp=1, dropout=0.0,
        )
        sdpa = V3LM(V3Config(**base)).eval()
        manual = V3LM(V3Config(**base, force_manual_attention=True)).eval()
        manual.load_state_dict(sdpa.state_dict())
        x = torch.randint(0, 64, (2, 32))
        with torch.no_grad():
            y_sdpa, _, _ = sdpa(x)
            y_manual, _, _ = manual(x)
        torch.testing.assert_close(y_manual, y_sdpa, atol=1e-5, rtol=1e-5)

    def test_auxiliary_compute_switches(self):
        cfg = V3Config(
            vocab_size=64, ctx=16, max_ctx=32, d_model=32,
            n_layers=2, n_heads=4, n_kv_heads=2, swiglu=64,
            window=8, global_every=2, moe_layers=1, n_experts=4,
            topk=2, shared_experts=1, expert_hidden=64,
            moe_hierarchical=True, moe_branches=2,
            moe_context_router=True, moe_route_feedback=True,
            mtp=3, mtp_block=True, mtp_block_experts=4,
            mtp_block_hidden=64, mtp_aux_layers=(0,),
            concept_loop_vertices=8, concept_loop_dim=8,
            concept_top_vertices=4, concept_top_pairs=2,
            concept_top_triples=1, use_memory=True,
            memory_slots=8, memory_heads=2,
            byte_probe=True, byte_probe_full=True,
            force_manual_attention=True, dropout=0.0,
        )
        model = V3LM(cfg).eval()
        tokens = torch.randint(0, 64, (2, 16))
        with torch.no_grad():
            _, _, aux_off = model(
                tokens,
                compute_mtp_aux=False,
                compute_byte_probe=False,
                compute_byte_sequence=False,
            )
            self.assertNotIn("mtp_aux_logits", aux_off)
            self.assertNotIn("byte_probe_logits", aux_off)
            self.assertNotIn("byte_sequence_logits", aux_off)
            _, _, aux_on = model(
                tokens,
                compute_mtp_aux=True,
                compute_byte_probe=True,
                compute_byte_sequence=True,
            )
            self.assertIn("mtp_aux_logits", aux_on)
            self.assertIn("byte_probe_logits", aux_on)
            self.assertIn("byte_sequence_logits", aux_on)

    def test_feedback_bus_starts_closed_and_carries_gradient(self):
        cfg = V3Config(
            vocab_size=64, ctx=16, max_ctx=32, d_model=32,
            n_layers=2, n_heads=4, n_kv_heads=2, swiglu=64,
            window=8, global_every=2, moe_layers=1, n_experts=4,
            topk=2, shared_experts=1, expert_hidden=64,
            moe_hierarchical=True, moe_branches=2,
            moe_context_router=True, moe_route_feedback=True,
            mtp=3, mtp_block=True, mtp_block_experts=4,
            mtp_block_hidden=64, mtp_aux_layers=(0,),
            concept_loop_vertices=8, concept_loop_dim=8,
            concept_top_vertices=4, concept_top_pairs=2,
            concept_top_triples=1, use_memory=True,
            memory_slots=8, memory_heads=2,
            force_manual_attention=True, dropout=0.0,
        )
        model = V3LM(cfg)
        tokens = torch.randint(0, 64, (2, 16))
        logits_a, mtp_a, aux = model(tokens)
        feedback = aux.get("feedback")
        self.assertIsInstance(feedback, dict)
        for key in (
            "feedback_token_gate",
            "feedback_route_hidden_gate",
            "feedback_route_concept_gate",
            "memory_feedback_gate",
            "mtp_carry_gate",
            "object_depth_carry_gate",
        ):
            self.assertIn(key, feedback)
            self.assertAlmostEqual(float(feedback[key]), 0.0, places=6)
        self.assertGreater(float(feedback["route_state_norm"]), 0.0)
        self.assertAlmostEqual(float(feedback["progress"]), 0.0, places=6)

        model.set_feedback_progress(1.0)
        opened = []
        for name, parameter in model.named_parameters():
            if name.endswith(
                (
                    "token_feedback_gate",
                    "route_hidden_gate",
                    "route_concept_gate",
                    "memory_feedback_gate",
                    "mtp_carry_gate",
                    "depth_carry_gate",
                )
            ):
                parameter.data.fill_(1.0)
                opened.append(name)
        self.assertGreaterEqual(len(opened), 6)
        logits_b, mtp_b, aux_b = model(tokens)
        self.assertGreater(float((logits_a - logits_b).abs().max()), 0.0)
        self.assertGreater(
            float(aux_b["feedback"]["feedback_token_gate"]), 0.5
        )

        loss = logits_b.float().logsumexp(dim=-1).mean()
        for head in mtp_b:
            loss = loss + head.float().logsumexp(dim=-1).mean()
        loss.backward()
        for name, parameter in model.named_parameters():
            if name in opened:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_memory_path_gradients(self):
        cfg = V3Config(
            vocab_size=32, ctx=32, max_ctx=32, d_model=32,
            n_layers=1, n_heads=4, swiglu=64, window=16,
            global_every=1, moe_layers=0, n_experts=4, topk=2,
            shared_experts=0, expert_hidden=64, mtp=1, dropout=0.0,
            use_memory=True, use_ngram_memory=True,
            memory_slots=8, memory_heads=4, memory_size=1024,
            memory_order=4, memory_min_count=1,
        )
        model = V3LM(cfg)
        base = torch.arange(32).repeat(64)
        model.build_memory(base)

        tokens = torch.arange(32).repeat(1).reshape(1, 32)
        logits, mtp, aux = model(tokens)
        self.assertIn("memory_entropy", aux)
        self.assertGreater(len(model.memory_path.logit_bias(tokens)), 0)

        loss = logits.float().log_softmax(-1).gather(-1, tokens[..., None]).mean()
        for head in mtp:
            loss = loss + head.float().log_softmax(-1).gather(
                -1, tokens[..., None]
            ).mean()
        loss.backward()
        alpha = model.memory_path.logit_bias.alpha
        self.assertIsNotNone(alpha.grad)
        self.assertTrue(torch.isfinite(alpha.grad).all())


if __name__ == "__main__":
    unittest.main()
