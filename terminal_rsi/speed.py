"""Optional torch.compile path for the 100M upgrade branch.

The repository contains a top-level ``z3/`` notebook directory that shadows
torch-dynamo's ``z3`` dependency when the repo root is on ``sys.path``.  The
compile helper removes the repo root during the first trace and restores it
afterwards.  Failures are non-fatal: the eager graph remains the source of
truth.
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch


def compile_upgrade(model, *, mode: str = "default"):
    """Compile the upgrade branch without changing checkpoint parameter names."""
    if not hasattr(torch, "compile"):
        print("torch.compile unavailable; keeping eager upgrade branch", flush=True)
        return None
    repo_root = Path(__file__).resolve().parent.parent
    saved = sys.path[:]
    try:
        sys.path[:] = [
            item for item in sys.path
            if Path(item or ".").resolve() != repo_root
        ]
        compiled = torch.compile(model.upgrades, mode=mode)
        device = next(model.parameters()).device
        dummy = torch.zeros(1, 33, dtype=torch.long, device=device)
        with torch.no_grad():
            compiled(dummy)
        print(f"torch.compile upgrade branch enabled (mode={mode})", flush=True)
        return compiled
    except Exception as exc:  # pragma: no cover - hardware/runtime dependent
        print(f"torch.compile failed, keeping eager path: {type(exc).__name__}: {exc}",
              flush=True)
        return None
    finally:
        sys.path[:] = saved
