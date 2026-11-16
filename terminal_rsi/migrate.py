"""Audited Terminal-RSI capacity migration: CPU -> T4 -> 96GB Blackwell.

Same model architecture with larger dimensions. This is NOT mathematically
function-preserving; matched downstream evaluation is mandatory before promotion.
Existing parameter slices and their Adam moments are retained; new slices start
from the destination initializer with zero optimizer moments.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import torch

try:
    from .unified import PRESETS, Preset, TerminalRSI
except ImportError:
    from unified import PRESETS, Preset, TerminalRSI


def _prefix_copy(dst: torch.Tensor, src: torch.Tensor):
    if dst.ndim != src.ndim:
        raise ValueError("Cannot migrate tensor with different rank")
    if dst.ndim == 0:
        dst.copy_(src.to(dst.device, dtype=dst.dtype))
    else:
        sl = tuple(slice(0, min(a, b)) for a, b in zip(dst.shape, src.shape))
        dst[sl].copy_(src[sl].to(dst.device, dtype=dst.dtype))


def _structured_copy(name: str, dst: torch.Tensor, src: torch.Tensor,
                     old_cfg: Preset, new_cfg: Preset):
    """Preserve Q/K/V, SwiGLU gate/value and per-head attention identities."""
    if name.endswith(".qkv.weight"):
        old = src.view(3, old_cfg.heads, old_cfg.width // old_cfg.heads, old_cfg.width)
        new = dst.view(3, new_cfg.heads, new_cfg.width // new_cfg.heads, new_cfg.width)
        heads = min(old.shape[1], new.shape[1])
        features = min(old.shape[2], new.shape[2])
        new[:, :heads, :features, :old_cfg.width].copy_(old[:, :heads, :features])
    elif name.endswith(".project.weight") and name.startswith("backbone."):
        old = src.view(old_cfg.width, old_cfg.heads, old_cfg.width // old_cfg.heads)
        new = dst.view(new_cfg.width, new_cfg.heads, new_cfg.width // new_cfg.heads)
        heads = min(old.shape[1], new.shape[1])
        features = min(old.shape[2], new.shape[2])
        new[:old_cfg.width, :heads, :features].copy_(old[:, :heads, :features])
    elif name.endswith(".up.weight") and name.startswith("backbone."):
        old = src.view(2, 4 * old_cfg.width, old_cfg.width)
        new = dst.view(2, 4 * new_cfg.width, new_cfg.width)
        new[:, :4 * old_cfg.width, :old_cfg.width].copy_(old)
    else:
        _prefix_copy(dst, src)

def migration_summary(source: dict, new_model: TerminalRSI, new_optimizer):
    source_cfg = Preset(**source["config"])
    dst_cfg = new_model.cfg
    if dst_cfg.width < source_cfg.width or dst_cfg.layers < source_cfg.layers:
        raise ValueError("Migration is expansion only; refuse lossy shrink")
    if dst_cfg.width == source_cfg.width and dst_cfg.layers == source_cfg.layers:
        raise ValueError("For unchanged architecture, use exact --resume")
    donor = source["model"] if "model" in source else source["neural"]
    opt_donor = source["optimizer"] if "optimizer" in source else source["neural_optimizer"]
    old_module = TerminalRSI(source_cfg)
    source_names = [name for name, _ in old_module.named_parameters()]
    target_names = [name for name, _ in new_model.named_parameters()]
    current = new_optimizer.state_dict()
    old_id_list = opt_donor["param_groups"][0]["params"]
    new_id_list = current["param_groups"][0]["params"]
    if len(old_id_list) < len(source_names) or len(new_id_list) < len(target_names):
        raise ValueError("Optimizer parameter ordering does not match checkpoint model")
    name_to_old_id = dict(zip(source_names, old_id_list))
    name_to_new_id = dict(zip(target_names, new_id_list))
    old_states = opt_donor["state"]
    migrated, exact, moments = [], [], 0
    with torch.no_grad():
        for name, target in new_model.named_parameters():
            if name not in donor:
                # Additive extra blocks start as identity, preserving old
                # representations until their output projections are trained.
                parts = name.split(".")
                if len(parts) > 2 and parts[0] == "backbone" and int(parts[1]) >= source_cfg.layers:
                    if name.endswith("project.weight") or name.endswith("down.weight"):
                        target.zero_()
                continue
            prior = donor[name]
            if prior.ndim != target.ndim:
                raise ValueError("Incompatible tensor rank " + name)
            _structured_copy(name, target, prior.to(target.device), source_cfg, dst_cfg)
            migrated.append(name)
            if target.shape == prior.shape:
                exact.append(name)
            old_id = name_to_old_id.get(name)
            new_id = name_to_new_id.get(name)
            if old_id not in old_states or new_id is None:
                continue
            new_state = {}
            for key, val in old_states[old_id].items():
                if torch.is_tensor(val) and val.ndim > 0:
                    if val.shape != prior.shape:
                        raise ValueError("Optimizer moment shape mismatch " + name)
                    next_val = torch.zeros_like(target, dtype=val.dtype, device="cpu")
                    _structured_copy(name, next_val, val.cpu(), source_cfg, dst_cfg)
                    new_state[key] = next_val
                elif torch.is_tensor(val):
                    new_state[key] = val.clone().cpu()
                else:
                    new_state[key] = val
            current["state"][new_id] = new_state
            moments += 1
    # Retain Adam hyperparameters, including inherited LR, without remapping
    # nonexistent specialist parameters into other tensors.
    if len(opt_donor["param_groups"]) != 1 or len(current["param_groups"]) != 1:
        raise ValueError("Only one inherited unified AdamW group is supported")
    for key, value in opt_donor["param_groups"][0].items():
        if key != "params":
            current["param_groups"][0][key] = value
    new_optimizer.load_state_dict(current)
    return {
        "source": asdict(source_cfg),
        "destination": asdict(dst_cfg),
        "weights_migrated": len(migrated),
        "weights_exact_shape": len(exact),
        "adam_states_migrated": moments,
        "function_preserving": False,
        "requires_matched_dev_gate": True,
    }


def adopt_from_file(path: Path, new_model: TerminalRSI, new_optimizer):
    source = torch.load(path, map_location="cpu", weights_only=True)
    if source.get("format") not in ("terminal-rsi-unified-v1", "terminal-rsi-unified-agent-v1"):
        raise ValueError("Only audited Terminal-RSI unified checkpoints can be expanded")
    return migration_summary(source, new_model, new_optimizer)
