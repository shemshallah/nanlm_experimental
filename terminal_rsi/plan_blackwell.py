"""Blackwell Terminal-RSI *preflight*, without checkpoint load or training.

Use --probe-gpu only inside the eventual 96 GiB Blackwell container.
The mathematical model remains the SAME VerbatimV103RSI and three Adam groups.
"""
from __future__ import annotations

import argparse
import json
import torch

from terminal_rsi.unified import blackwell_memory_plan, device_for


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--physical-gib", type=float, default=96.0,
                   help="Planning-only physical VRAM; overridden by --probe-gpu")
    p.add_argument("--usable-fraction", type=float, default=0.92)
    p.add_argument("--probe-gpu", action="store_true",
                   help="Validate CUDA Blackwell hardware and available free VRAM")
    args = p.parse_args(argv)
    physical_gib = args.physical_gib
    hardware = {}
    if args.probe_gpu:
        device = device_for("blackwell")
        free, total = torch.cuda.mem_get_info(device)
        physical_gib = total / (1024 ** 3)
        hardware = {
            "device_name": torch.cuda.get_device_name(device),
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "free_vram_gib": free / (1024 ** 3),
        }
        if not hardware["bf16_supported"]:
            raise RuntimeError("Cannot enable Blackwell BF16 additive specialists")
    plan = blackwell_memory_plan(
        physical_vram_gib=physical_gib,
        usable_fraction=args.usable_fraction)
    plan["hardware"] = hardware
    if args.probe_gpu and hardware["free_vram_gib"] < plan["fp32_weights_grad_adam_gib_approx"]:
        raise RuntimeError("Insufficient FREE VRAM for planned FP32 AdamW state")
    print(json.dumps(plan, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
