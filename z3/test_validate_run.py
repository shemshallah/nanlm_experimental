#!/usr/bin/env python3
"""Unit tests for the static z3 run validator."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validate_run import EXPECTED_115M, validate_payload


def _run(**overrides):
    payload = {
        "status": "completed",
        "nonfinite_skips": 0,
        "feedback_gate_warmup_steps": 500,
        "feedback_progress_final": 1.0,
        "config": {
            "preset": "115m",
            "resolved": {
                "params": EXPECTED_115M,
                "moe_layout": [
                    {
                        "layer": 7,
                        "branches": 4,
                        "experts_per_branch": 3,
                        "topk": 2,
                        "degenerate": False,
                    }
                ],
            },
        },
    }
    payload.update(overrides)
    return payload


def _test(**overrides):
    payload = {
        "test": {
            "sidecar": True,
            "document_framing": True,
            "feedback_progress": 1.0,
        },
        "aggregate": {
            "feedback": {
                "feedback_token_gate": 0.1,
                "feedback_route_concept_gate": 0.1,
                "progress": 1.0,
            },
            "object_memory": {"object_gate": 0.1},
        },
        "macro_bits_per_byte": 2.0,
        "corpora": {
            "code": {"bits_per_byte": 1.8},
            "prose": {"bits_per_byte": 2.2},
        },
    }
    payload.update(overrides)
    return payload


class ValidateRunTests(unittest.TestCase):
    def test_valid_payload_passes(self):
        result = validate_payload(_run(), _test())
        self.assertTrue(result["ok"], result["errors"])

    def test_parameter_mismatch_fails(self):
        result = validate_payload(
            _run(config={"preset": "115m", "resolved": {"params": 1}}),
            _test(),
        )
        self.assertFalse(result["ok"])
        self.assertTrue(
            any(item["name"] == "parameter_count" for item in result["errors"])
        )

    def test_baseline_regression_fails(self):
        baseline = _test()
        baseline["macro_bits_per_byte"] = 1.0
        baseline["corpora"]["prose"]["bits_per_byte"] = 1.0
        result = validate_payload(_run(), _test(), baseline)
        self.assertFalse(result["ok"])
        names = {item["name"] for item in result["errors"]}
        self.assertIn("macro_vs_baseline", names)
        self.assertIn("domain_regressions", names)

    def test_kaggle10m_preset_uses_115m_parameter_count(self):
        run = _run()
        run["config"]["preset"] = "kaggle10m"
        result = validate_payload(run, _test())
        self.assertTrue(result["ok"], result["errors"])

    def test_feedback_progress_must_reach_one(self):
        result = validate_payload(
            _run(feedback_progress_final=0.5), _test()
        )
        self.assertFalse(result["ok"])
        self.assertTrue(
            any(
                item["name"] == "feedback_final_progress"
                for item in result["errors"]
            )
        )

    def test_required_feedback_open_is_an_error(self):
        test = _test()
        test["aggregate"]["feedback"]["feedback_token_gate"] = 0.0
        test["aggregate"]["feedback"]["feedback_route_concept_gate"] = 0.0
        result = validate_payload(_run(), test, require_feedback_open=True)
        self.assertFalse(result["ok"])
        self.assertTrue(
            any(
                item["name"] == "feedback_path_opened"
                for item in result["errors"]
            )
        )

    def test_required_feedback_open_accepts_an_open_gate(self):
        result = validate_payload(_run(), _test(), require_feedback_open=True)
        self.assertTrue(result["ok"], result["errors"])

    def test_required_object_gate_is_an_error(self):
        test = _test()
        test["aggregate"]["object_memory"]["object_gate"] = 0.01
        result = validate_payload(_run(), test, require_object_gate=True)
        self.assertFalse(result["ok"])
        self.assertTrue(
            any(
                item["name"] == "object_gate_open"
                for item in result["errors"]
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
