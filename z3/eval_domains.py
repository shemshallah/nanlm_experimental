#!/usr/bin/env python3
"""Evaluate a z3 checkpoint over a directory of per-domain held-out text.

This is the test-side companion to ``build_domain_bins.py``.  Each
``test/<domain>.txt`` file is encoded with document framing, scored with the
concept sidecar, and aggregated into:

* per-domain BPB / bits-per-token / top-1 / top-5,
* token-weighted and macro BPB,
* a domain bootstrap 95% interval,
* feedback-bus and router diagnostics,
* one JSON artifact for the run.

Example::

    python z3/eval_domains.py \
      --checkpoint runs/z3_115m/best.pt \
      --test-dir data/z3_domains_115m/test \
      --tokenizer layered --document-framing --sidecar \
      --ctx 512 --batch-size 8 --device cuda \
      --out runs/z3_115m/test_domains.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval_v3  # noqa: E402


def _bootstrap_mean(
    values: list[float],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    mean = sum(values) / len(values)
    if resamples <= 0 or len(values) == 1:
        return mean, mean, mean
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        draws.append(sum(sample) / len(sample))
    draws.sort()
    lower = draws[int(0.025 * (len(draws) - 1))]
    upper = draws[int(0.975 * (len(draws) - 1))]
    return mean, lower, upper


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest(
    path: str,
    *,
    cfg: Any,
    layered_model: str,
    corpora: list[tuple[str, Path]],
) -> dict[str, Any]:
    manifest_path = Path(path).expanduser()
    if not manifest_path.is_file():
        raise SystemExit(f"data manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "z3-document-framed-domain-bins":
        raise SystemExit(
            f"unexpected data manifest kind: {manifest.get('kind')!r}"
        )
    if int(manifest.get("vocab_size", -1)) != int(cfg.vocab_size):
        raise SystemExit(
            f"manifest vocab {manifest.get('vocab_size')} != model "
            f"vocab {cfg.vocab_size}"
        )
    if not (manifest.get("document_framing") or {}).get("enabled"):
        raise SystemExit("manifest does not declare document framing")
    expected_layered = manifest.get("layered_model_sha256")
    layered_path = Path(layered_model).expanduser()
    if expected_layered and layered_path.is_file():
        actual = _sha256_file(layered_path)
        if actual != expected_layered:
            raise SystemExit(
                "test tokenizer does not match data manifest: "
                f"{layered_path} sha256={actual} expected={expected_layered}"
            )
    domains = manifest.get("domains") or {}
    for name, corpus_path in corpora:
        entry = domains.get(name)
        if entry is None:
            raise SystemExit(
                f"test corpus {name!r} is not in data manifest"
            )
        expected = entry.get("test_text_sha256")
        if not expected:
            raise SystemExit(
                f"data manifest has no test hash for {name!r}"
            )
        actual = _sha256_file(corpus_path)
        if actual != expected:
            raise SystemExit(
                f"test corpus {name!r} does not match manifest: "
                f"{corpus_path} sha256={actual} expected={expected}"
            )
    return manifest


def _parse_corpora(
    test_dir: str | None,
    specs: list[str],
    max_domains: int,
) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    if test_dir:
        root = Path(test_dir).expanduser()
        if not root.is_dir():
            raise SystemExit(f"test directory not found: {root}")
        for path in sorted(root.glob("*.txt")):
            result.append((path.stem, path))
    for spec in specs:
        if "=" in spec:
            name, raw_path = spec.split("=", 1)
        else:
            raw_path = spec
            name = Path(spec).stem
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise SystemExit(f"corpus not found: {path}")
        result.append((name, path))
    if max_domains > 0:
        result = result[:max_domains]
    if not result:
        raise SystemExit("no test corpora; pass --test-dir or --corpus")
    names = [name for name, _ in result]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate corpus names in test set")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--test-dir", default="")
    parser.add_argument(
        "--data-manifest",
        default="",
        help="manifest.json from build_domain_bins.py; verifies test hashes",
    )
    parser.add_argument(
        "--corpus",
        action="append",
        default=[],
        metavar="[NAME=]PATH",
        help="extra corpus outside --test-dir",
    )
    parser.add_argument("--out", default="eval_domains.json")
    parser.add_argument(
        "--tokenizer",
        choices=("auto", "v4", "layered", "bpe", "bytes"),
        default="layered",
    )
    parser.add_argument("--bpe-model", default="")
    parser.add_argument(
        "--layered-model",
        default=str(eval_v3.LAYERED_TOKENIZER_PATH),
    )
    parser.add_argument(
        "--sidecar",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-sidecar",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--document-framing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--ctx", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-bytes", type=int, default=0)
    parser.add_argument("--max-domains", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args(argv)

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.max_bytes < 0:
        raise SystemExit("--max-bytes must be non-negative")

    device = eval_v3._resolve_device(args.device)
    started = time.time()
    model, cfg, metadata = eval_v3.load_checkpoint(
        args.checkpoint, config_json=args.config or None
    )
    model = model.to(device)
    if hasattr(model, "set_feedback_progress"):
        model.set_feedback_progress(1.0)
    ctx = int(args.ctx) if args.ctx > 0 else int(cfg.ctx)
    tokenizer = eval_v3.load_tokenizer(
        args.tokenizer,
        args.bpe_model or None,
        args.layered_model or None,
    )
    sidecar_builder = (
        eval_v3._make_sidecar_builder(args.layered_model)
        if args.sidecar
        and getattr(cfg, "use_concept_loop", False)
        and getattr(tokenizer, "kind", None) == "layered"
        else None
    )
    if (
        args.sidecar
        and getattr(cfg, "use_concept_loop", False)
        and sidecar_builder is None
    ):
        message = "concept model test has no sidecar"
        if args.require_sidecar:
            raise SystemExit(message)
        print(f"[warn] {message}", flush=True)

    corpora_specs = _parse_corpora(
        args.test_dir or None, args.corpus, args.max_domains
    )
    data_manifest = (
        _verify_manifest(
            args.data_manifest,
            cfg=cfg,
            layered_model=args.layered_model,
            corpora=corpora_specs,
        )
        if args.data_manifest else None
    )
    router_accumulator = eval_v3.RouterAccumulator()
    corpora: dict[str, dict[str, Any]] = {}
    for name, path in corpora_specs:
        ids = eval_v3.encode_file(
            path,
            tokenizer,
            args.max_bytes,
            document_framing=bool(args.document_framing),
        )
        if not ids:
            raise SystemExit(f"{name}: corpus produced no tokens")
        if max(ids) >= cfg.vocab_size:
            raise SystemExit(
                f"{name}: tokenizer/model vocab mismatch "
                f"({max(ids)} >= {cfg.vocab_size})"
            )
        byte_lengths = eval_v3._token_byte_lengths(tokenizer, ids)
        metrics = eval_v3.evaluate_ids_full(
            model,
            ids,
            cfg,
            ctx=ctx,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
            name=name,
            router_accumulator=router_accumulator,
            token_byte_lengths=byte_lengths,
            sidecar_builder=sidecar_builder,
        )
        raw_bytes = path.stat().st_size
        if args.max_bytes > 0:
            raw_bytes = min(raw_bytes, args.max_bytes)
        metrics["path"] = str(path)
        metrics["raw_bytes"] = raw_bytes
        metrics["encoded_tokens"] = len(ids)
        metrics["bytes_per_token"] = raw_bytes / max(1, len(ids))
        metrics["bits_per_byte"] = (
            metrics.get("eval_bits_per_byte")
            or metrics["bits_per_token"] / max(1e-9, metrics["bytes_per_token"])
        )
        corpora[name] = metrics
        print(
            f"[test] {name:24s} tokens={metrics['tokens']:8d} "
            f"bpb={metrics['bits_per_byte']:.4f} "
            f"top1={metrics['top1'] * 100:5.2f}%",
            flush=True,
        )

    aggregate = eval_v3.aggregate_metrics(corpora, router_accumulator)
    total_tokens = sum(int(m["tokens"]) for m in corpora.values())
    total_bytes = sum(
        int(m.get("eval_bytes") or 0) for m in corpora.values()
    )
    total_loss = sum(
        float(m["loss"]) * int(m["tokens"]) for m in corpora.values()
    )
    weighted_bpb = (
        total_loss / math.log(2) / total_bytes
        if total_bytes > 0 else None
    )
    bpb_values = [
        float(m["bits_per_byte"])
        for m in corpora.values()
        if m.get("bits_per_byte") is not None
    ]
    macro, lower, upper = _bootstrap_mean(
        bpb_values,
        resamples=args.bootstrap,
        seed=args.seed,
    )
    result = {
        "tool": "z3/eval_domains",
        "version": 1,
        "checkpoint": str(Path(args.checkpoint).expanduser()),
        "checkpoint_step": metadata.get("step"),
        "model_config": {
            "params": int(model.parameter_count()),
            "ctx": int(cfg.ctx),
            "vocab_size": int(cfg.vocab_size),
        },
        "test": {
            "corpora": [name for name, _ in corpora_specs],
            "ctx": ctx,
            "batch_size": int(args.batch_size),
            "document_framing": bool(args.document_framing),
            "sidecar": sidecar_builder is not None,
            "feedback_progress": (
                float(model.feedback_progress.detach().cpu())
                if hasattr(model, "feedback_progress") else None
            ),
            "data_manifest": (
                str(Path(args.data_manifest).expanduser().resolve())
                if data_manifest is not None else None
            ),
        },
        "tokens": total_tokens,
        "eval_bytes": total_bytes,
        "weighted_bits_per_byte": weighted_bpb,
        "macro_bits_per_byte": macro,
        "macro_bits_per_byte_ci95": [lower, upper],
        "aggregate": aggregate,
        "corpora": corpora,
        "wall_seconds": round(time.time() - started, 3),
    }
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[done] macro_bpb={macro:.4f} "
        f"ci95=[{lower:.4f},{upper:.4f}] "
        f"weighted_bpb={weighted_bpb if weighted_bpb is None else round(weighted_bpb, 4)} "
        f"test={'/'.join(name for name, _ in corpora_specs)}",
        flush=True,
    )
    print(f"[done] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
