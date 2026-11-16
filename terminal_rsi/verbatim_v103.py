"""Verbatim v103 bit-path first; learned CPU/T4/Blackwell RSI upgrades second.

The v103 model is the exact archived implementation from the last training
notebook (cell 14), NOT a rewritten approximation. At residual_gate=0, output
bit logits are bit-for-bit identical on the same device to inherited v103.
The original trained v103 checkpoint can be attached read-only and its full
online-state tensors and Adam moments migrated to ONE unified optimizer.

GPU vs CPU floating-point equality is tolerance-checked, not promised bitwise.
NaNCM original C *numerical* parity still needs its separate trained C oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.nn import functional as F

from .legacy_v103_reference import V103ParityGPUFullSystem, PARITY_SCHEMA
from .unified import PRESETS, TerminalRSI, device_for


class VerbatimV103RSI(nn.Module):
    def __init__(self, mode="cpu"):
        super().__init__()
        if mode not in PRESETS:
            raise ValueError("Unknown CPU/T4/Blackwell mode")
        self.cfg = PRESETS[mode]
        self.original = V103ParityGPUFullSystem()
        self.upgrades = TerminalRSI(self.cfg)
        self.residual_gate = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _byte_to_conditional_bit_logits(byte_logits: torch.Tensor,
                                        target: torch.Tensor):
        """Exact differentiable 256-way factorization into causal bit logits."""
        if byte_logits.ndim != 2 or byte_logits.shape[-1] != 256:
            raise ValueError("Byte logits must be [batch,256]")
        target = target.long()
        choices = torch.arange(256, device=byte_logits.device)
        log_probs = F.log_softmax(byte_logits.float(), dim=-1)
        out = []
        for depth in range(8):
            # True target contributes only already-observed more-significant bits.
            prefix = target >> (8 - depth)
            same = (choices[None, :] >> (8 - depth)) == prefix[:, None]
            bit_is_one = ((choices >> (7 - depth)) & 1).bool()[None, :]
            mass_one = torch.logsumexp(log_probs.masked_fill(~(same & bit_is_one), -torch.inf), -1)
            mass_zero = torch.logsumexp(log_probs.masked_fill(~(same & ~bit_is_one), -torch.inf), -1)
            out.append(mass_one - mass_zero)
        return torch.stack(out, dim=1)

    def forward(self, previous: torch.Tensor, target: torch.Tensor,
                aux: torch.Tensor | None = None,
                local: torch.Tensor | None = None):
        if previous.shape[1] != 32:
            raise ValueError("Exact v103 consumes 32 previous observed bytes")
        old_logits, bits = self.original(previous, target, aux, local)
        # When untrained, this extra model receives context but cannot perturb
        # the inherited v103 prediction because residual_gate is exact zero.
        current = self.upgrades(previous.long())["logits"][0, :, -1, :]
        proposed = self._byte_to_conditional_bit_logits(current, target)
        gate = torch.tanh(self.residual_gate)
        return old_logits + gate * (proposed - old_logits), bits

    @torch.no_grad()
    def observe_verified_byte(self, bpb: float, bit_accuracy: float):
        """Reuses the original v103 RSI critic; AFTER scoring only."""
        self.original.observe_byte_loss(bpb, bit_accuracy)

    @torch.no_grad()
    def observe_verified_terminal(self, success: bool):
        self.original.observe_terminal(success)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


def restore_v103_parent(original_path: Path, model: VerbatimV103RSI, optimizer):
    """Strict full checkpoint migration; retain every inherited Adam state."""
    path = Path(original_path)
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    source = torch.load(path, map_location="cpu", weights_only=True)
    if source.get("schema") != PARITY_SCHEMA:
        raise ValueError(f"Expected {PARITY_SCHEMA}; got {source.get('schema')}")
    if not source.get("optimizer", {}).get("state"):
        raise ValueError("Original v103 optimizer moments absent: refuse non-continuation")
    model.original.load_state_dict(source["model"], strict=True)
    model.original.causal_copy_enabled = bool(
        source.get("configuration", {}).get("causal_copy_enabled", False))
    old_ids = source["optimizer"]["param_groups"][0]["params"]
    old_names = [name for name, _ in model.original.named_parameters()]
    if len(old_ids) != len(old_names):
        raise ValueError("Original model/Adam parameter count mismatch")
    current = optimizer.state_dict()
    new_ids = current["param_groups"][0]["params"]
    names = [name for name, _ in model.named_parameters()]
    if len(new_ids) != len(names):
        raise ValueError("Unified optimizer does not own every trainable tensor")
    name_to_id = dict(zip(names, new_ids))
    original_adam = source["optimizer"]["state"]
    # Named mapping is vital: top-level residual_gate precedes all submodules
    # in torch.nn.Module.parameters(), despite being assigned afterward.
    current["state"] = {
        name_to_id["original." + name]: original_adam[old_id]
        for name, old_id in zip(old_names, old_ids)
        if old_id in original_adam
    }
    if len(current["state"]) != len(original_adam):
        raise ValueError("Failed to map all original Adam moments by name")
    for key, value in source["optimizer"]["param_groups"][0].items():
        if key != "params":
            current["param_groups"][0][key] = value
    optimizer.load_state_dict(current)
    return {
        "legacy_schema": PARITY_SCHEMA,
        "legacy_step": source.get("steps"),
        "legacy_sha256": source_sha,
        "legacy_adam_states": len(original_adam),
        "inherited_model_tensors": len(model.original.state_dict()),
        "frozen_copy_config": model.original.causal_copy_enabled,
        "parity_gate_initial": float(model.residual_gate.detach()),
    }


def exact_parent_parity(model: VerbatimV103RSI, previous: torch.Tensor,
                        target: torch.Tensor, aux: torch.Tensor | None = None):
    """Read-only exact same-device parity, including saved online buffers."""
    model.eval()
    with torch.no_grad():
        baseline, bits_a = model.original(previous, target, aux)
        migrated, bits_b = model(previous, target, aux)
    if not torch.equal(bits_a, bits_b) or not torch.equal(baseline, migrated):
        raise AssertionError("VERBATIM V103 PARITY FAILED. Refuse further training.")
    return {"exact_parity": True, "samples": int(len(target)),
            "max_abs_logit_error": float((baseline - migrated).abs().max().item())}


def save_combined(path: Path, model: VerbatimV103RSI, optimizer, step: int,
                  provenance: dict, feedback_events=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"schema": "terminal-rsi-verbatim-v103-plus-upgrades-v1",
              "mode": model.cfg.name, "config": vars(model.cfg),
              "model": model.state_dict(), "optimizer": optimizer.state_dict(),
              "steps": step, "legacy_provenance": provenance,
              "feedback_events": list(feedback_events or [])}
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".incomplete", delete=False) as f:
        tmp = Path(f.name)
    try:
        torch.save(bundle, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def main():
    parser = argparse.ArgumentParser(description="Read-only v103 checkpoint parity before any CPU/GPU training")
    parser.add_argument("--mode", choices=PRESETS, default="cpu")
    parser.add_argument("--legacy-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    dev = device_for(args.mode)
    model = VerbatimV103RSI(args.mode).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=PRESETS[args.mode].lr)
    audit = restore_v103_parent(args.legacy_checkpoint, model, opt)
    # Real checkpoint state, deterministic observed byte context; no target-future input.
    g = torch.Generator().manual_seed(103)
    p = torch.randint(0, 256, (4, 32), generator=g).to(dev)
    t = torch.randint(0, 256, (4,), generator=g).to(dev)
    report = exact_parent_parity(model, p, t)
    print(json.dumps({"checkpoint": audit, "inherited_model_parameter_count":
                      model.original.count_parameters() if hasattr(model.original, "count_parameters")
                      else sum(v.numel() for v in model.original.parameters()),
                      "unified_parameter_count": model.count_parameters(),
                      "parity": report}, indent=2))


if __name__ == "__main__":
    main()
