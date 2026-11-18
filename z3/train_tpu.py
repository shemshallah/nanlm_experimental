#!/usr/bin/env python3
"""Experimental 8-core TPU wrapper around ``train_v3.main``.

This is the multi-core companion to the maintained single-device path:

    python z3/train_v3.py --device xla ...

Use it when a TPU VM exposes several XLA devices (Kaggle TPU v3-8/v5e-8):

    python z3/train_tpu.py --nprocs 8 --out runs/z3_115m_tpu \
      --preset kaggle10m_nomtp \
      --corpus data/wikitext_bins/train.bin \
      --val-corpus data/wikitext_bins/valid.bin \
      --data-manifest data/wikitext_bins/manifest.json \
      --batch-size 4

Each rank runs the same trainer with a different seed and a rank-local output
directory.  ``torch_xla.core.xla_model.optimizer_step`` reduces gradients
across replicas, so the master's checkpoint is the synchronized model.  The
non-master checkpoint directories are deleted after the workers exit.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _worker(index: int, argv: list[str], out: str, seed: int) -> None:
    import torch_xla.core.xla_model as xm

    rank_out = out if xm.is_master_ordinal() else f"/tmp/z3_tpu_rank{index}"
    child_argv = list(argv) + [
        "--device", "xla",
        "--seed", str(seed + index),
        "--out", rank_out,
    ]
    from train_v3 import main

    raise SystemExit(main(child_argv))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run train_v3.py on every TPU XLA replica.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--nprocs", type=int, default=8)
    args, passthrough = parser.parse_known_args(argv)
    if args.nprocs < 1:
        raise SystemExit("--nprocs must be positive")
    try:
        import torch_xla.distributed.xla_multiprocessing as xmp
    except ImportError as exc:
        raise SystemExit(
            "train_tpu.py requires torch_xla; install the TPU wheel first"
        ) from exc

    print(
        f"[tpu] spawning {args.nprocs} replicas; master out={args.out}",
        flush=True,
    )
    xmp.spawn(
        _worker,
        args=(passthrough, args.out, args.seed),
        nprocs=args.nprocs,
        start_method="fork",
    )
    for index in range(args.nprocs):
        shutil.rmtree(f"/tmp/z3_tpu_rank{index}", ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
