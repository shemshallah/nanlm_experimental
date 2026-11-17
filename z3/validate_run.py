#!/usr/bin/env python3
"""Validate a completed z3 train/test run against the promotion gates.

This is a static checker: it reads ``config.json`` / ``result.json`` from the
run directory and a test artifact from ``eval_domains.py`` or ``eval_v3.py``.
It does not load a model or a GPU.

Example::

    python z3/validate_run.py \
      --run runs/z3_115m \
      --test runs/z3_115m/test_domains.json \
      --baseline runs/z3_115m_no_feedback/test_domains.json \
      --out runs/z3_115m/validation.json

Exit code is 0 only when all error-severity checks pass.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

EXPECTED_115M = 115_238_237
EXPECTED_KAGGLE10M_MTP64 = 111_097_437


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _feedback_dict(test: dict[str, Any]) -> dict[str, Any]:
    aggregate = test.get("aggregate")
    if isinstance(aggregate, dict) and isinstance(aggregate.get("feedback"), dict):
        return aggregate["feedback"]
    if isinstance(test.get("feedback"), dict):
        return test["feedback"]
    return {}


def _object_dict(test: dict[str, Any]) -> dict[str, Any]:
    aggregate = test.get("aggregate")
    if isinstance(aggregate, dict) and isinstance(
        aggregate.get("object_memory"), dict
    ):
        return aggregate["object_memory"]
    if isinstance(test.get("object_memory"), dict):
        return test["object_memory"]
    return {}


def validate_payload(
    run: dict[str, Any],
    test: dict[str, Any],
    baseline: dict[str, Any] | None = None,
    *,
    expected_params: int | None = None,
    min_feedback_gate: float = 1e-4,
    max_macro_regression: float = 0.03,
    max_domain_regression: float = 0.05,
    min_object_gate: float = 0.05,
    require_feedback_open: bool = False,
    require_object_gate: bool = False,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(
        name: str,
        ok: bool,
        detail: Any,
        severity: str = "error",
    ) -> None:
        checks.append(
            {
                "name": name,
                "ok": bool(ok),
                "severity": severity,
                "detail": detail,
            }
        )

    config = run.get("config") or {}
    resolved = config.get("resolved") or {}
    preset = config.get("preset") or run.get("preset")
    params = resolved.get("params")
    if params is None:
        params = (test.get("model_config") or {}).get("params")
    target = expected_params
    if target is None and preset in (
        "115m", "kaggle10m", "kaggle10m_obj", "kaggle10m_route",
        "kaggle10m_lr3e4", "kaggle10m_lr3e4_decay", "kaggle10m_lr4e4",
        "kaggle10m_nomtp", "kaggle10m_noconcept", "kaggle10m_trunk",
    ):
        target = EXPECTED_115M
    elif target is None and preset == "kaggle10m_mtp64":
        target = EXPECTED_KAGGLE10M_MTP64
    if target is not None:
        check(
            "parameter_count",
            params == target,
            {"actual": params, "expected": target},
        )

    status = run.get("status")
    check(
        "status",
        status in ("completed", "max_seconds"),
        status,
    )
    check(
        "nonfinite_skips",
        int(run.get("nonfinite_skips") or 0) == 0,
        run.get("nonfinite_skips"),
    )

    warmup = run.get("feedback_gate_warmup_steps")
    if warmup is None:
        warmup = config.get("feedback_gate_warmup_steps")
    check("feedback_warmup", int(warmup or 0) > 0, warmup)
    final_progress = run.get("feedback_progress_final")
    check(
        "feedback_final_progress",
        final_progress is not None
        and abs(float(final_progress) - 1.0) < 1e-6,
        final_progress,
    )

    test_info = test.get("test") or {}
    check(
        "test_sidecar",
        bool(test_info.get("sidecar")) or bool(test.get("sidecar")),
        test_info.get("sidecar", test.get("sidecar")),
    )
    check(
        "test_document_framing",
        bool(test_info.get("document_framing"))
        or bool(test.get("document_framing")),
        test_info.get(
            "document_framing", test.get("document_framing")
        ),
    )
    test_progress = test_info.get(
        "feedback_progress", test.get("feedback_progress")
    )
    check(
        "test_feedback_progress",
        test_progress is not None
        and abs(float(test_progress) - 1.0) < 1e-6,
        test_progress,
    )

    feedback = _feedback_dict(test)
    gate_values = {
        key: value
        for key, value in feedback.items()
        if key.endswith("gate")
    }
    max_gate = max(
        (abs(float(value)) for value in gate_values.values()),
        default=0.0,
    )
    check(
        "feedback_path_opened",
        max_gate > min_feedback_gate,
        {"max_abs_gate": max_gate, "threshold": min_feedback_gate},
        severity="error" if require_feedback_open else "warning",
    )

    macro = test.get("macro_bits_per_byte")
    if macro is None:
        macro = test.get("bits_per_byte")
    check("macro_bpb_finite", _finite(macro), macro)

    object_memory = _object_dict(test)
    object_gate = object_memory.get("object_gate")
    if object_gate is not None:
        check(
            "object_gate_open",
            float(object_gate) > min_object_gate,
            {"object_gate": object_gate, "threshold": min_object_gate},
            severity="error" if require_object_gate else "warning",
        )

    layout = resolved.get("moe_layout") or []
    degenerate = [
        item for item in layout if bool(item.get("degenerate"))
    ]
    check(
        "moe_layout_non_degenerate",
        not degenerate,
        {"degenerate_layers": degenerate},
        severity="warning",
    )

    if baseline is not None and _finite(macro):
        baseline_macro = baseline.get("macro_bits_per_byte")
        if baseline_macro is None:
            baseline_macro = baseline.get("bits_per_byte")
        if _finite(baseline_macro):
            delta = float(macro) - float(baseline_macro)
            check(
                "macro_vs_baseline",
                delta <= max_macro_regression,
                {
                    "macro": float(macro),
                    "baseline": float(baseline_macro),
                    "delta": delta,
                    "max_regression": max_macro_regression,
                },
            )
        domains = test.get("corpora") or {}
        baseline_domains = baseline.get("corpora") or {}
        regressions = {}
        for name, metrics in domains.items():
            baseline_metrics = baseline_domains.get(name)
            if not baseline_metrics:
                continue
            current = metrics.get("bits_per_byte")
            prior = baseline_metrics.get("bits_per_byte")
            if _finite(current) and _finite(prior):
                delta = float(current) - float(prior)
                if delta > max_domain_regression:
                    regressions[name] = {
                        "current": float(current),
                        "baseline": float(prior),
                        "delta": delta,
                    }
        check(
            "domain_regressions",
            not regressions,
            {
                "max_regression": max_domain_regression,
                "regressions": regressions,
            },
        )

    errors = [
        item for item in checks
        if not item["ok"] and item["severity"] == "error"
    ]
    warnings = [
        item for item in checks
        if not item["ok"] and item["severity"] == "warning"
    ]
    return {
        "ok": not errors,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "preset": preset,
        "params": params,
        "macro_bits_per_byte": macro,
    }


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="training run directory")
    parser.add_argument("--test", required=True, help="test JSON artifact")
    parser.add_argument("--baseline", default="", help="optional baseline test JSON")
    parser.add_argument("--out", default="", help="validation JSON output")
    parser.add_argument("--expected-params", type=int, default=0)
    parser.add_argument("--min-feedback-gate", type=float, default=1e-4)
    parser.add_argument("--max-macro-regression", type=float, default=0.03)
    parser.add_argument("--max-domain-regression", type=float, default=0.05)
    parser.add_argument("--min-object-gate", type=float, default=0.05)
    parser.add_argument(
        "--require-feedback-open",
        action="store_true",
        help="treat a closed feedback bus as an error (for stability gates)",
    )
    parser.add_argument(
        "--require-object-gate",
        action="store_true",
        help="treat a closed object gate as an error",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run).expanduser()
    config_path = run_dir / "config.json"
    result_path = run_dir / "result.json"
    if not result_path.is_file():
        raise SystemExit(f"result.json not found in {run_dir}")
    result = _load(result_path)
    config = _load(config_path) if config_path.is_file() else {}
    run_payload = dict(result)
    run_payload["config"] = config or result.get("config") or {}
    test_payload = _load(Path(args.test).expanduser())
    baseline_payload = (
        _load(Path(args.baseline).expanduser())
        if args.baseline else None
    )
    validation = validate_payload(
        run_payload,
        test_payload,
        baseline_payload,
        expected_params=args.expected_params or None,
        min_feedback_gate=args.min_feedback_gate,
        max_macro_regression=args.max_macro_regression,
        max_domain_regression=args.max_domain_regression,
        min_object_gate=args.min_object_gate,
        require_feedback_open=args.require_feedback_open,
        require_object_gate=args.require_object_gate,
    )
    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    for item in validation["checks"]:
        mark = "PASS" if item["ok"] else (
            "WARN" if item["severity"] == "warning" else "FAIL"
        )
        print(f"[{mark}] {item['name']}: {item['detail']}")
    print(
        f"[validate] ok={validation['ok']} "
        f"errors={len(validation['errors'])} "
        f"warnings={len(validation['warnings'])}"
    )
    return 0 if validation["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
