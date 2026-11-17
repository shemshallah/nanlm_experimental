"""10-minute Blackwell v4 smoke for the unified 6.5M Terminal-RSI model.

Builds a fresh English-only, book-disjoint pilot corpus, probes the
95 GiB Blackwell GPU, then runs real FP32 v103 + BF16 upgrade-branch
forward/backward/AdamW steps through ``train_verbatim --smoke``.  CPU, T4
and Blackwell use the exact same 6,495,162-parameter graph; Blackwell is
an execution option, not a larger model.  The smoke reports peak VRAM,
throughput and the v4 ledger, but deliberately does NOT save a checkpoint
or a champion.

Example::

    python -m terminal_rsi.run_blackwell_smoke \
      --output-dir /marimo/terminal_rsi_blackwell_smoke \
      --seconds 600 --context 512 --batch 1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .prepare_cpu_pilot import build_cpu_pilot
from .run_molab_cpu_1000 import source_dir


def _run_tee(command: list[str], cwd: Path, console: Path) -> int:
    print("Executing Blackwell smoke:", " ".join(command), flush=True)
    with console.open("w", encoding="utf-8") as log:
        with subprocess.Popen(
            command, cwd=cwd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1,
        ) as child:
            assert child.stdout is not None
            for line in child.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return child.wait()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--corpus-dir", type=Path,
                   help="Previously extracted official NaNROUTER batch 0004")
    p.add_argument("--seconds", type=float, default=600.0,
                   help="Wall-clock smoke budget in the training loop")
    p.add_argument("--context", type=int, default=256,
                   help="Observed causal bytes; the unified model caps at 256")
    p.add_argument("--batch", type=int, default=8192,
                   help="Effective Blackwell batch; activation memory is split by --microbatch")
    p.add_argument("--microbatch", type=int, default=512,
                   help="Microbatch for gradient accumulation; effective batch remains --batch")
    p.add_argument("--upgrade-loss-weight", type=float, default=1.0,
                   help="Direct byte CE on the additive upgrade branch")
    p.add_argument("--train-bytes", type=int, default=2_000_000)
    p.add_argument("--dev-books", type=int, default=64)
    p.add_argument("--max-dev-positions", type=int, default=8)
    p.add_argument("--topk-probes", type=int, default=1)
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--rsi-version", choices=("off", "v3", "v4"), default="v4",
                   help="v4 is the unified path; v3/off are parity baselines")
    p.add_argument("--rsi-v4-cycle-updates", type=int, default=500)
    p.add_argument("--rsi-v4-horizon", type=int, default=5)
    p.add_argument("--rsi-v4-candidates", type=int, default=2)
    p.add_argument("--rsi-v4-shards", type=int, default=4)
    p.add_argument("--rsi-v4-shard-positions", type=int, default=128)
    p.add_argument("--rsi-v4-eval-chunk", type=int, default=8)
    p.add_argument("--rsi-v4-min-gain", type=float, default=0.01)
    p.add_argument("--rsi-v4-z", type=float, default=2.0)
    p.add_argument("--rsi-v4-critic-steps", type=int, default=16)
    p.add_argument("--rsi-v4-policy-steps", type=int, default=8)
    p.add_argument("--hierarchical-eval-books", type=int, default=4,
                   help="Contiguous DEV books for byte/BPE/word/sentence metrics; 0=off")
    p.add_argument("--hierarchical-eval-max-positions", type=int, default=512,
                   help="Positions per book for hierarchical eval")
    p.add_argument("--hierarchical-eval-batch", type=int, default=8)
    args = p.parse_args(argv)
    if (args.seconds <= 0 or args.batch < 1 or args.microbatch < 0
            or not args.upgrade_loss_weight >= 0
            or not 32 <= args.context <= 256):
        p.error("Positive seconds/batch and nonnegative microbatch/upgrade loss; context in [32, 256]")
    if (args.hierarchical_eval_books < 0
            or args.hierarchical_eval_max_positions < 0
            or args.hierarchical_eval_batch < 1):
        p.error("Hierarchical eval needs nonnegative books/positions and positive batch")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics = output / "blackwell_smoke_metrics.json"
    console = output / "blackwell_smoke_console.log"
    candidate = output / "blackwell_smoke_candidate.pt"
    if metrics.exists() or console.exists():
        p.error(f"Smoke receipt or console already exists under {output}; choose a NEW --output-dir")

    corpus = source_dir(output, args.corpus_dir)
    corpus_report = build_cpu_pilot(
        corpus_dir=corpus, output_dir=output,
        train_bytes=args.train_bytes, dev_books=args.dev_books,
        dev_positions=1024, seed=103,
    )
    print("Fresh English-only TRAIN/DEV:", json.dumps(corpus_report, indent=2), flush=True)

    preflight = subprocess.run(
        [sys.executable, "-m", "terminal_rsi.plan_blackwell", "--probe-gpu"],
        capture_output=True, text=True,
    )
    if preflight.returncode:
        print(preflight.stdout, preflight.stderr, file=sys.stderr)
        return preflight.returncode
    plan = json.loads(preflight.stdout)
    print("Blackwell preflight:", json.dumps(plan, sort_keys=True, indent=2), flush=True)

    command = [
        sys.executable, "-m", "terminal_rsi.train_verbatim",
        "--mode", "blackwell", "--fresh-start",
        "--seconds", str(args.seconds), "--batch", str(args.batch),
        "--microbatch", str(args.microbatch),
        "--upgrade-loss-weight", str(args.upgrade_loss_weight),
        "--upgrade-context", str(args.context),
        "--train-stream", str(output / "train_cpu_pilot.bin"),
        "--bpe-merges", "terminal_rsi/reference/nancm_v103_bpe.txt",
        "--dev-bin", str(output / "win_dev.bin"),
        "--dev-manifest", str(output / "win_dev.manifest.json"),
        "--output", str(candidate),
        "--metrics-output", str(metrics),
        "--max-dev-positions", str(args.max_dev_positions),
        "--topk-probes", str(args.topk_probes),
        "--rsi-version", args.rsi_version,
        "--progress-every", str(args.progress_every),
        "--smoke",
    ]
    if args.rsi_version == "v4":
        command += [
            "--rsi-holdout-stream", str(output / "rsi_holdout.bin"),
            "--rsi-v4-cycle-updates", str(args.rsi_v4_cycle_updates),
            "--rsi-v4-horizon", str(args.rsi_v4_horizon),
            "--rsi-v4-candidates", str(args.rsi_v4_candidates),
            "--rsi-v4-shards", str(args.rsi_v4_shards),
            "--rsi-v4-shard-positions", str(args.rsi_v4_shard_positions),
            "--rsi-v4-eval-chunk", str(args.rsi_v4_eval_chunk),
            "--rsi-v4-min-gain", str(args.rsi_v4_min_gain),
            "--rsi-v4-z", str(args.rsi_v4_z),
            "--rsi-v4-critic-steps", str(args.rsi_v4_critic_steps),
            "--rsi-v4-policy-steps", str(args.rsi_v4_policy_steps),
        ]
    if args.hierarchical_eval_books:
        command += [
            "--hierarchical-eval-books", str(args.hierarchical_eval_books),
            "--hierarchical-eval-corpus-dir", str(corpus),
            "--hierarchical-eval-max-positions",
            str(args.hierarchical_eval_max_positions),
            "--hierarchical-eval-batch", str(args.hierarchical_eval_batch),
        ]
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    status = _run_tee(command, Path.cwd(), console)
    receipt = json.loads(metrics.read_text()) if metrics.is_file() else None
    summary = {
        "schema": "terminal-rsi-blackwell-smoke-v1",
        "status": "completed" if status == 0 else f"failed:{status}",
        "command": command,
        "preflight": plan,
        "corpus": corpus_report,
        "receipt": receipt,
        "console": str(console),
        "note": (
            "Smoke only: the 6.5M unified model is used; no tens-of-GiB "
            "optimizer checkpoint or champion is written. Peak VRAM and "
            "throughput are in receipt.gpu; v4 ledger is in receipt.rsi_v4."
        ),
    }
    (output / "blackwell_smoke_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Terminal-RSI Blackwell smoke",
        "",
        f"- Status: `{summary['status']}`",
        f"- Budget: {args.seconds:.0f}s",
        f"- Context / batch: {args.context} / {args.batch}",
        f"- GPU: `{plan['hardware']['device_name']}`",
        f"- Free VRAM at preflight: {plan['hardware']['free_vram_gib']:.1f} GiB",
        f"- Planned FP32 AdamW state: {plan['fp32_weights_grad_adam_gib_approx']:.1f} GiB",
        f"- Planned activation floor: {plan['bf16_checkpoint_activation_floor_gib']:.2f} GiB",
    ]
    if receipt:
        gpu = receipt.get("gpu") or {}
        lines += [
            f"- Updates: {receipt['actual_byte_updates_this_round']}",
            f"- Training loop: {receipt['training_duration_seconds']:.1f}s",
            f"- Updates/s: {receipt['updates_per_second']:.3f}",
            f"- Peak allocated: {gpu.get('peak_allocated_gib')} GiB",
            f"- Peak reserved: {gpu.get('peak_reserved_gib')} GiB",
            f"- DEV before/after BPB: {(receipt.get('before') or {}).get('bpb')} -> {(receipt.get('after') or {}).get('bpb')}",
        ]
        hierarchical = receipt.get("hierarchical") or {}
        for level in ("byte", "subword", "token", "word", "sentence"):
            metric = hierarchical.get(level) or {}
            if not metric.get("units"):
                continue
            lines.append(
                f"- {level}: top1={metric.get('top1'):.4f} "
                f"top5={metric.get('top5'):.4f} "
                f"units={metric.get('units')} "
                f"bits/unit={metric.get('bits_per_unit'):.4f} "
                f"loss/unit={metric.get('loss_per_unit'):.4f}")
        byte_metric = hierarchical.get("byte") or {}
        if byte_metric:
            lines.append(
                f"- byte BPB/BPC/loss: {byte_metric.get('bpb'):.4f} / "
                f"{byte_metric.get('bpc'):.4f} / {byte_metric.get('loss'):.4f}")
    lines += ["", f"Console: `{console}`", f"Metrics: `{metrics}`", ""]
    (output / "RUN_SUMMARY.md").write_text("\n".join(lines))
    print("SMOKE SUMMARY:", json.dumps(summary, indent=2), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
