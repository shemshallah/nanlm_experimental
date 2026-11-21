"""CPU contract tests for exact v103-training loop and RSI online adaptation."""
from __future__ import annotations
import hashlib
import tempfile
import struct
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from terminal_rsi.unified import PRESETS, Preset
from terminal_rsi.verbatim_v103 import VerbatimV103RSI, build_v103_optimizer, save_combined, resume_combined
from terminal_rsi.verbatim_metrics import byte_metrics_wrapped
from terminal_rsi.train_verbatim import (
    score_byte_observations, train_one_batch, finish_rsi_round, verify_parent_source,
    build_causal_contexts, build_book_disjoint_probes,
)
from terminal_rsi.legacy_v103_reference import PARITY_SCHEMA


class VerbatimTrainingCPU(unittest.TestCase):
    def setUp(self):
        self.cfg = Preset("cpu", 32, 1, 4, 64, 1, 2, 1e-3, 1)
        self.patch = patch.dict(PRESETS, {"cpu": self.cfg})
        self.patch.start()
        torch.manual_seed(1103)
        self.model = VerbatimV103RSI("cpu")
        self.optimizer = build_v103_optimizer(self.model)
        self.context = torch.randint(0, 256, (2, 32), dtype=torch.long)
        self.target = torch.randint(0, 256, (2,), dtype=torch.long)
        self.aux = self.model.original._default_aux(self.context).to(torch.int32)

    def tearDown(self):
        self.patch.stop()

    def test_bitwise_training_adapts_original_rsi_and_additive_residual(self):
        old, bits = self.model.original(self.context, self.target, self.aux)
        combined, _ = self.model(self.context, self.target, self.aux)
        self.assertTrue(torch.equal(old, combined))
        self.assertEqual(self.model.residual_gate.item(), 0.0)
        before = self.model.original.v103_slot_counts.clone()
        report = train_one_batch(self.model, self.optimizer, self.context,
                                 self.target, self.aux)
        self.assertTrue(0 <= report["bit_accuracy"] <= 1)
        self.assertGreater(report["bpb"], 0)
        self.assertFalse(torch.equal(before, self.model.original.v103_slot_counts))
        self.assertNotEqual(self.model.residual_gate.item(), 0.0)
        self.assertGreater(len(self.optimizer.state), 0)
        self.assertEqual(len(self.optimizer.param_groups), 4)

    def test_verified_rsi_feedback_controls_the_same_neural_upgrade_path(self):
        """A bounded RSI trust change alters learned residuals, never the gate-zero parent."""
        historic, _ = self.model.original(self.context, self.target, self.aux)
        inherited, _ = self.model(self.context, self.target, self.aux)
        self.assertTrue(torch.equal(historic, inherited))
        with torch.no_grad():
            self.model.residual_gate.fill_(0.1)
            self.model.original.feedback_state.zero_()
        early_historic, _ = self.model.original(self.context, self.target, self.aux)
        early, _ = self.model(self.context, self.target, self.aux)
        residual = early - early_historic
        self.assertGreater(float(residual.abs().sum()), 0)
        with torch.no_grad():
            self.model.original.feedback_state.fill_(4.0)
        late_historic, _ = self.model.original(self.context, self.target, self.aux)
        late, _ = self.model(self.context, self.target, self.aux)
        factor = 1.0 + 0.25 * torch.tanh(torch.tensor(4.0))
        torch.testing.assert_close(late - late_historic, residual * factor,
                                   atol=3e-5, rtol=3e-4)
        with torch.no_grad():
            self.model.residual_gate.zero_()
        restored, _ = self.model(self.context, self.target, self.aux)
        self.assertTrue(torch.equal(restored, late_historic))

    def test_original_terminal_rsi_is_active_even_when_byte_training_is_zero(self):
        """Original always performs two verifier-driven actor/critic steps."""
        previous = self.model.original.actor_logits.detach().clone()
        initial_events = int(self.model.original.verified_events)
        report = finish_rsi_round(self.model, self.optimizer)
        self.assertEqual(len(report), 2)
        self.assertEqual([event["verified_success"] for event in report],
                         [False, True])
        self.assertFalse(torch.equal(previous, self.model.original.actor_logits))
        self.assertEqual(int(self.model.original.verified_events),
                         initial_events + 2)
        self.assertGreater(len(self.optimizer.state), 0)
        self.assertEqual(len(self.optimizer.param_groups), 4)

    def test_exact_256_way_dev_scorer_passes_aux_and_local_every_time(self):
        class Spy:
            def __init__(self):
                self.calls = []
            def eval(self):
                pass
            def __call__(self, history, target, aux, local):
                assert aux is not None and local is not None
                assert aux.shape == (len(target), 6)
                assert local.shape == (len(target), 4)
                self.calls.append(len(target))
                bits = ((target[:, None] >> torch.arange(7, -1, -1)) & 1).float()
                # Uniform normalized byte distribution yields BPB=8 exactly.
                return torch.zeros_like(bits), bits

        spy = Spy()
        examples = torch.randint(0, 256, (2, 33))
        aux = torch.randint(0, 255, (2, 6), dtype=torch.int32)
        local = torch.randint(0, 5, (2, 4), dtype=torch.int32)
        scored = byte_metrics_wrapped(spy, examples, n=2, topk_n=2,
                                      auxiliary=aux, local=local)
        self.assertEqual(spy.calls, [2, 512])
        self.assertAlmostEqual(scored["bpb"], 8.0, places=5)
        self.assertAlmostEqual(scored["maximum_log_probability_mass_error"],
                               0.0, places=5)

    def test_longer_context_never_changes_inherited_v103_bits(self):
        early = torch.randint(0, 256, (2, 32))
        long_history = torch.cat((early, self.context), dim=-1)
        old, bits_a = self.model.original(self.context, self.target, self.aux)
        new, bits_b = self.model(long_history, self.target, self.aux)
        self.assertTrue(torch.equal(old, new))
        self.assertTrue(torch.equal(bits_a, bits_b))
        # First scored historical byte at offset32 has 32 left padding bytes.
        raw = torch.arange(96, dtype=torch.uint8)
        windows = build_causal_contexts(raw, torch.tensor([32, 64]), 64)
        self.assertEqual(windows.shape, (2, 65))
        self.assertTrue(torch.equal(windows[0, :32], torch.zeros(32, dtype=torch.long)))
        self.assertTrue(torch.equal(windows[0, 32:-1], raw[:32].long()))
        self.assertEqual(windows[0, -1].item(), 32)

    def test_long_context_dev_is_book_sealed_and_last32_exact(self):
        first = bytes(range(35))
        second = bytes(range(100, 135))
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "win_dev.bin"
            with file.open("wb") as fh:
                fh.write(struct.pack("<II", 2, 32))
                for book in (first, second):
                    fh.write(struct.pack("<I", len(book)))
                    fh.write(book)
            # Original independent DEV evaluator windows target positions 32..34.
            full = torch.stack([torch.tensor(
                [list(book[i-32:i+1]) for i in range(32,35)],
                dtype=torch.long) for book in (first,second)])
            aux = torch.zeros(2,3,6,dtype=torch.int32)
            local = torch.zeros(2,3,4,dtype=torch.int32)
            probes, features, book_matches, digest = build_book_disjoint_probes(
                file, 6, 64, full, aux, local, torch.device("cpu"))
            self.assertEqual(probes.shape, (6,65))
            self.assertEqual(features.shape, (6,6))
            self.assertEqual(book_matches.shape, (6,4))
            self.assertTrue(torch.equal(probes[:, -33:],
                                        full.reshape(-1,33)))
            # Second book must NOT inherit the first book's final bytes.
            self.assertTrue(torch.equal(probes[3,:32],torch.zeros(32,dtype=torch.long)))
            self.assertEqual(probes[3,-1].item(),132)

    def test_lookahead_and_critic_losses_are_wired(self):
        report = train_one_batch(
            self.model, self.optimizer, self.context, self.target, self.aux,
            upgrade_loss_weight=1.0, microbatch=0,
            lookahead_loss_weight=0.5, critic_loss_weight=0.1)
        self.assertIn("lookahead_bpb", report)
        self.assertIn("critic_mse", report)
        self.assertGreater(report["lookahead_bpb"], 0.0)
        self.assertGreater(report["critic_mse"], 0.0)
        for head in (self.model.upgrades.output[1],
                     self.model.upgrades.output[2]):
            self.assertIsNotNone(head.weight.grad)
            self.assertGreater(float(head.weight.grad.abs().max()), 0.0)
        critic = self.model.upgrades.critic
        for parameter in critic.parameters():
            if parameter.grad is not None:
                self.assertGreater(float(parameter.grad.abs().max()), 0.0)
                break
        else:
            self.fail("critic received no gradient")

    def test_no_train_on_fake_missing_causal_features(self):
        with self.assertRaisesRegex(ValueError, "six causal features"):
            train_one_batch(self.model, self.optimizer, self.context,
                            self.target, None)

    def test_combined_checkpoint_exact_restart_all_rsi_and_adam_tensors(self):
        train_one_batch(self.model, self.optimizer, self.context,
                        self.target, self.aux)
        provenance = {"source_sha256": "orig-sha", "feature_manifest": {"sha": "causal"},
                      "dev_sha256": "dev-sha",
                      "sampler": {"cursor": 128, "stride": 1, "train_positions": 64}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "unified.pt"
            save_combined(path, self.model, self.optimizer, 42, provenance,
                          feedback_events=[{"verified": True}])
            recovered = VerbatimV103RSI("cpu")
            recovered_optimizer = build_v103_optimizer(recovered)
            report = resume_combined(path, recovered, recovered_optimizer,
                                     source_sha256="orig-sha",
                                     feature_manifest={"sha": "causal"},
                                     dev_sha256="dev-sha")
            self.assertEqual(report["step"], 42)
            self.assertEqual(report["sampler"]["cursor"], 128)
            self.assertEqual(len(recovered_optimizer.state),
                             len(self.optimizer.state))
            for name, prior in self.model.state_dict().items():
                self.assertTrue(torch.equal(prior, recovered.state_dict()[name]), name)
            for first, second in zip(self.optimizer.param_groups,
                                     recovered_optimizer.param_groups):
                self.assertEqual(first["lr"], second["lr"])
            with self.assertRaisesRegex(ValueError, "source|identity"):
                resume_combined(path, recovered, recovered_optimizer,
                                source_sha256="wrong", feature_manifest={"sha":"causal"},
                                dev_sha256="dev-sha")

    def test_original_sampler_and_feature_fingerprints_cannot_drift(self):
        raw = bytes(range(70))
        digest = hashlib.sha256(raw).hexdigest()
        manifest = {"schema": "v103-causal-feature-bridge-v1",
                    "feature_sha256": "fixture", "bpe_merge_sha256": "x"}
        src = {"schema": PARITY_SCHEMA, "source_sha256": digest,
               "v103_feature_manifest": manifest,
               "holdout_digests": {"dev": "pinned-dev"},
               "sampler": {"schema": "full-cycle-affine-v1", "cursor": 271,
                           "stride": 1, "train_positions": len(raw) - 32}}
        got = verify_parent_source(src, raw, None, manifest, "pinned-dev")
        self.assertEqual(got["cursor"], 271)
        with self.assertRaisesRegex(ValueError, "stream SHA"):
            verify_parent_source(src, raw+b"x", None, manifest, "pinned-dev")
        with self.assertRaisesRegex(ValueError, "features differ"):
            verify_parent_source(src, raw, None, {"other": "feature"}, "pinned-dev")
        with self.assertRaisesRegex(ValueError, "DEV digest"):
            verify_parent_source(src, raw, None, manifest, "other-dev")


if __name__ == "__main__":
    unittest.main(verbosity=2)
