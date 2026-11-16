"""CPU contract tests for exact v103-training loop and RSI online adaptation."""
from __future__ import annotations
import hashlib
import unittest
from unittest.mock import patch
import torch
from terminal_rsi.unified import PRESETS, Preset
from terminal_rsi.verbatim_v103 import VerbatimV103RSI, build_v103_optimizer
from terminal_rsi.verbatim_metrics import byte_metrics_wrapped
from terminal_rsi.train_verbatim import (
    score_byte_observations, train_one_batch, verify_parent_source,
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
        self.assertEqual(len(self.optimizer.param_groups), 3)

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

    def test_no_train_on_fake_missing_causal_features(self):
        with self.assertRaisesRegex(ValueError, "six causal features"):
            train_one_batch(self.model, self.optimizer, self.context,
                            self.target, None)

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
