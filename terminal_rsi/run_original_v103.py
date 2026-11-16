"""Run ORIGINAL archived v103 CPU training code without replacing its implementation.

This subprocess invokes legacy_v103_reference.main verbatim, with CUDA
hidden before torch import. All feature preprocessing, optimizer groups,
affine sampler, online counts, bitwise loss and terminal verifier are owned
by the ORIGINAL function. It writes a separate candidate, never its parent.

Requires exact original 10M train stream, original BPE merges and sealed DEV.
Even --seconds=0 executes the original terminal-feedback stage and saves a
NEW candidate: use 'python -m terminal_rsi.verbatim_v103' for read-only parity.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys


def build_original_v103_command(*, checkpoint: Path, train: Path,
                                dev: Path, bpe: Path, output: Path,
                                seconds: float = 0,
                                causal_copy: bool = False,
                                reference_dev: Path | None = None,
                                reference_full256: Path | None = None):
    """One deterministic and inspectable invocation of the untouched function."""
    if output.resolve() == checkpoint.resolve():
        raise ValueError("Refusing overwrite of historical trained checkpoint")
    if output.exists():
        raise FileExistsError("Refusing existing original-reference candidate")
    required = {"checkpoint": checkpoint, "train": train,
                "dev": dev, "bpe": bpe}
    for label, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"Original {label} missing: {path}")
    if seconds < 0:
        raise ValueError("Negative CPU training budget")
    script = ("import sys; from terminal_rsi.legacy_v103_reference "
              "import main; main(sys.argv[1:])")
    command = [sys.executable, "-c", script,
               "--corpus", str(train), "--resume", str(checkpoint),
               "--output", str(output), "--seconds", str(seconds),
               "--batch", "4", "--checkpoint-every", "0",
               "--holdout-dev", str(dev),
               "--bpe-merges", str(bpe),
               "--allow-cpu-test", "--v103-pathways"]
    if causal_copy:
        command.append("--causal-copy")
    if reference_dev is not None:
        if not reference_dev.is_file():
            raise FileNotFoundError(f"Original frozen C DEV missing: {reference_dev}")
        command.extend(["--reference-dev", str(reference_dev)])
    if reference_full256 is not None:
        if not reference_full256.is_file():
            raise FileNotFoundError(f"Original frozen C full-256 DEV missing: {reference_full256}")
        command.extend(["--reference-full256", str(reference_full256)])
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Protected terminal_rsi_gpu_v103_full.pt")
    parser.add_argument("--train-stream", type=Path, required=True)
    parser.add_argument("--dev-bin", type=Path, required=True)
    parser.add_argument("--bpe-merges", type=Path,
                        default=Path("terminal_rsi/reference/nancm_v103_bpe.txt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=0)
    parser.add_argument("--reference-dev", type=Path, default=None,
                        help="Frozen original-C DEV Rec stream, if checkpoint requires it")
    parser.add_argument("--reference-full256", type=Path, default=None,
                        help="Frozen original-C full 256-probability reference, if recorded")
    args = parser.parse_args(argv)
    if args.checkpoint.resolve() == args.output.resolve():
        parser.error("Protected original checkpoint must never be overwritten")
    # Read only a small metadata subset of the original checkpoint. The
    # actual tensor loader and exact optimizer migration run in the verbatim
    # worker process, not in this launcher.
    import torch
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = checkpoint.get("configuration") or {}
    if checkpoint.get("schema") != "terminal-rsi-integrated-cuda-v103-pathways-v2":
        raise ValueError("Refusing an unrelated model as original-v103 donor")
    historical = checkpoint.get("holdout_digests") or {}
    supplied = {"c_dev_reference": args.reference_dev,
                "c_full256_reference": args.reference_full256}
    for field, path in supplied.items():
        wanted = historical.get(field)
        if wanted and path is None:
            raise ValueError(f"Original checkpoint requires --{field.replace('_', '-')} "
                             f"(SHA-256 {wanted}); original main() cannot replay without it")
        if path is not None:
            if not path.is_file():
                raise FileNotFoundError(f"Missing original frozen reference: {path}")
            got = hashlib.sha256(path.read_bytes()).hexdigest()
            if wanted and got != wanted:
                raise ValueError(f"Original frozen {field} SHA mismatch: "
                                 f"wanted {wanted}, got {got}")
    run = build_original_v103_command(
        checkpoint=args.checkpoint, train=args.train_stream, dev=args.dev_bin,
        bpe=args.bpe_merges, output=args.output, seconds=args.seconds,
        causal_copy=bool(cfg.get("causal_copy_enabled", False)),
        reference_dev=args.reference_dev,
        reference_full256=args.reference_full256)
    print("Original v103 checkpoint:", args.checkpoint)
    print("Original v103 SHA256:", hashlib.sha256(args.checkpoint.read_bytes()).hexdigest())
    print("Same archived original main() function; CPU subprocess with no CUDA.")
    if args.seconds == 0:
        print("Original main() still performs terminal actor/critic verification at 0 seconds.")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(run, check=True, env=env)
    return args.output


if __name__ == "__main__":
    main()
