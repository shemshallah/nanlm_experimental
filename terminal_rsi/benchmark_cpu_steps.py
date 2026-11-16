"""Measured CPU step times on a synthetic Terminal-RSI v103 + upgrades.

This measures the ACTUAL three-mode-shared forward/BCE/backward/Adam/RSI
implementation, but not the privately held trained v103 checkpoint or real DEV.
It is a throughput diagnostic, never a trained-weight BPB result.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch

from terminal_rsi.train_verbatim import train_one_batch
from terminal_rsi.verbatim_v103 import VerbatimV103RSI, build_v103_optimizer, exact_parent_parity


def measure(history: int = 256, *, batch: int = 2, steps: int = 3,
            warmup: int = 1, threads: int = 2) -> dict:
    if history < 32 or history > 256 or batch < 1 or steps < 1 or warmup < 0 or threads < 1:
        raise ValueError("CPU benchmark needs 32..256 past bytes and positive step/batch/thread")
    torch.manual_seed(103)
    torch.set_num_threads(threads)
    model = VerbatimV103RSI("cpu")
    optimizer = build_v103_optimizer(model)
    observed = torch.randint(0, 256, (batch, history), dtype=torch.long)
    target = torch.randint(0, 256, (batch,), dtype=torch.long)
    aux = torch.randint(0, 256, (batch, 6), dtype=torch.int32)
    initial = exact_parent_parity(model, observed, target, aux)
    durations = []
    losses = []
    for i in range(warmup + steps):
        started = time.perf_counter()
        result = train_one_batch(model, optimizer, observed, target, aux)
        elapsed = time.perf_counter() - started
        if i >= warmup:
            durations.append(elapsed)
            losses.append(result["bpb"])
    # The terminal RSI actor/critic phase is the once-per-round original
    # policy update, separate from the per-byte optimizer step above.
    durations.sort()
    return {
        "protocol": "synthetic-v103-plus-upgrades-full-CPU-training-step",
        "true_trained_checkpoint_loaded": False,
        "true_book_disjoint_dev_scored": False,
        "device": "cpu",
        "threads": threads,
        "observed_history": history,
        "batch_size": batch,
        "warmup_steps": warmup,
        "measured_steps": steps,
        "parameter_count": model.count_parameters(),
        "original_parameters": sum(p.numel() for p in model.original.parameters()),
        "upgrade_parameters": sum(p.numel() for p in model.upgrades.parameters()) + 1,
        "preserved_three_adam_groups": len(optimizer.param_groups) == 3,
        "gate_zero_exact_original_parity": initial["exact_parity"],
        "step_seconds_min": durations[0],
        "step_seconds_median": statistics.median(durations),
        "step_seconds_max": durations[-1],
        "steps_per_second_median": 1.0 / statistics.median(durations),
        "last_synthetic_train_bpb": losses[-1],
        "synthetic_train_bpb_is_not_dev": True,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history", type=int, default=256)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args(argv)
    print("TERMINAL_RSI_CPU_STEP_BENCHMARK", json.dumps(
        measure(args.history, batch=args.batch, steps=args.steps,
                warmup=args.warmup, threads=args.threads), sort_keys=True),
          flush=True)


if __name__ == "__main__":
    main()
