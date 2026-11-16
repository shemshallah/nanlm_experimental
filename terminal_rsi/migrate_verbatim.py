"""Explicit VERBATIM RSI capacity migration CPU -> T4 -> Blackwell.

The historical v103 core, 32-byte states, online counts, verified terminal
actor/critic, its two AdamW groups and every inherited moment remain EXACT.
Only the additive neural upgrade is width/depth-expanded with structured
QKV/SwiGLU/Adam tensor mapping. Such expansion is not function preserving
once its residual gate is nonzero, so DEV acceptance must precede promotion.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

from .unified import PRESETS, Preset
from .migrate import migration_summary
from .verbatim_v103 import VerbatimV103RSI, build_v103_optimizer, save_combined

SCHEMA = "terminal-rsi-verbatim-v103-plus-upgrades-v1"


def migrate_checkpoint(source: dict, new_mode: str):
    if source.get("schema") != SCHEMA:
        raise ValueError("Cannot migrate non-verbatim Terminal-RSI model")
    if new_mode not in PRESETS:
        raise ValueError("Unknown target mode")
    old_cfg = Preset(**source["config"])
    if new_mode == old_cfg.name:
        raise ValueError("Same-mode resume should be exact, not a migration")
    target_cfg = PRESETS[new_mode]
    if target_cfg.width < old_cfg.width or target_cfg.layers < old_cfg.layers:
        raise ValueError("Capacity migration must not shrink inherited tensors")
    trained = source["model"]
    original_state = {name[len("original."):]: tensor
                      for name, tensor in trained.items()
                      if name.startswith("original.")}
    source_upgrade = {name[len("upgrades."):]: tensor
                      for name, tensor in trained.items()
                      if name.startswith("upgrades.")}
    if not original_state or not source_upgrade or "residual_gate" not in trained:
        raise ValueError("Original v103, upgrade or RSI trust gate absent")
    old_optimizer = source["optimizer"]
    old_groups = old_optimizer["param_groups"]
    if len(old_groups) != 3:
        raise ValueError("Two historical + one upgrade AdamW groups required")
    model = VerbatimV103RSI(new_mode)
    model.original.load_state_dict(original_state, strict=True)
    with torch.no_grad():
        model.residual_gate.copy_(trained["residual_gate"])
    optimizer = build_v103_optimizer(model)
    # The expansion routine understands QKV head positions, SwiGLU halves,
    # identity-initialized new blocks and corresponding second moments.
    # Feed it ONLY the additive upgrade parameters; gate is handled separately.
    upgrade_ids = old_groups[2]["params"]
    old_upgrade_params = list(source_upgrade)
    if len(upgrade_ids) != 1 + len(old_upgrade_params):
        raise ValueError("Saved upgrade Adam mapping changed since last run")
    upgrade_old_optimizer = {
        "param_groups": [{**{k: v for k, v in old_groups[2].items()
                            if k != "params"}, "params": upgrade_ids[1:]}],
        "state": {pid: old_optimizer["state"][pid]
                  for pid in upgrade_ids[1:]
                  if pid in old_optimizer["state"]},
    }
    isolated = torch.optim.AdamW(model.upgrades.parameters(), lr=target_cfg.lr)
    report = migration_summary({
        "config": source["config"],
        "model": source_upgrade,
        "optimizer": upgrade_old_optimizer,
    }, model.upgrades, isolated)
    current = optimizer.state_dict()
    old_state = old_optimizer["state"]
    moment_copies = 0
    for idx in (0, 1):
        older = old_groups[idx]["params"]
        newer = current["param_groups"][idx]["params"]
        if len(older) != len(newer):
            raise ValueError("Original Adam parameter group width changed")
        for old_id, new_id in zip(older, newer):
            if old_id in old_state:
                current["state"][new_id] = copy.deepcopy(old_state[old_id])
                moment_copies += 1
    new_upgrade_ids = current["param_groups"][2]["params"]
    if upgrade_ids[0] in old_state:
        current["state"][new_upgrade_ids[0]] = copy.deepcopy(old_state[upgrade_ids[0]])
        moment_copies += 1
    isolated_state = isolated.state_dict()
    isolated_ids = isolated_state["param_groups"][0]["params"]
    if len(isolated_ids) != len(new_upgrade_ids) - 1:
        raise AssertionError("New upgrade Adam group did not align by name")
    for old_id, new_id in zip(isolated_ids, new_upgrade_ids[1:]):
        if old_id in isolated_state["state"]:
            current["state"][new_id] = isolated_state["state"][old_id]
    for idx in range(3):
        for key, value in old_groups[idx].items():
            if key != "params":
                current["param_groups"][idx][key] = copy.deepcopy(value)
    optimizer.load_state_dict(current)
    # Absolute parity of every old parameter AND persistent online buffer.
    for name, previous in original_state.items():
        if not torch.equal(previous, model.original.state_dict()[name]):
            raise AssertionError("VERBATIM historical v103 state lost: " + name)
    report.update({
        "old_mode": old_cfg.name, "new_mode": target_cfg.name,
        "original_v103_tensors_exact": len(original_state),
        "original_adam_moments_exact": moment_copies,
        "original_optimizer_group_lrs": [g["lr"] for g in optimizer.param_groups[:2]],
        "residual_gate_exact": bool(torch.equal(trained["residual_gate"],
                                                model.residual_gate.detach())),
        "parent_parity_at_zero_gate": float(model.residual_gate.detach()) == 0.0,
        "nonzero_gate_capacity_expansion_requires_real_book_dev": True,
        "no_checkpoint_promotion": True,
    })
    return model, optimizer, report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--mode", choices=("t4", "blackwell"), required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.source.resolve() == args.output.resolve() or args.output.exists():
        p.error("Migration must write a new candidate; preserve original checkpoint")
    source = torch.load(args.source, map_location="cpu", weights_only=True)
    model, optimizer, report = migrate_checkpoint(source, args.mode)
    provenance = dict(source["legacy_provenance"])
    provenance["capacity_migration"] = report
    save_combined(args.output, model, optimizer, int(source["steps"]), provenance,
                  source.get("feedback_events"), source.get("training_state"))
    print(report)
    print("Candidate only. Run exact same-source book-disjoint DEV before any training or promotion.")
    return report


if __name__ == "__main__":
    main()
