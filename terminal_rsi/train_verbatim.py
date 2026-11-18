"""Exact original v103 Terminal-RSI training loop + additive 3-preset upgrades.

CPU parity must pass before training, T4 or Blackwell. The inherited model,
online slot counts, two original AdamW groups, full-cycle sampler, and verified
terminal feedback all retain the original v103 semantics. The upgrade branch
starts at a strictly zero residual gate and learns by the SAME bitwise loss.
The uploaded .pt parent and independently sealed TEST are never overwritten.

This entry point requires ORIGINAL raw train bytes, BPE merge ranks and an
independent-book NaNCM-format DEV. No synthetic fallback / unmatched DEV BPB.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import copy
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.nn import functional as F
from torch import nn

from .build_fresh_dev import validate_fresh_dev, _write_if_unchanged_or_new
from .legacy_v103_reference import (
    PARITY_CONTEXT, PARITY_SCHEMA, build_train_features, feature_manifest,
    read_bpe_ranks, heldout_windows, heldout_auxiliary, make_windows,
    coverage_indices, co_prime_stride, terminal_feedback,
)
from .hierarchical_bus import (
    boundary_loss, boundary_targets, bit_logits_to_byte_log_probs,
)
from .unit_heads import build_unit_labels, unit_head_loss
from .fast_rsi import build_fast_cache, run_fast_adapter_rsi
from .unified import PRESETS, device_for
from .rsi_self_builder import RSISelfBuilder
from .rsi_v4 import (
    Action, BranchResult, BranchSnapshot, FastStateSnapshot, ProbeShards,
    RSIv4Config, RSIv4Controller, apply_action, probe_noise_audit,
)
from .champion_guard import consider_candidate
from .verbatim_metrics import byte_metrics_wrapped
from .legacy_v103_local_copy import book_local_features
from .verbatim_c256 import compare_c256_wrapped
from .verbatim_v103 import (
    VerbatimV103RSI, build_v103_optimizer, restore_v103_parent,
    exact_parent_parity, save_combined, resume_combined,
)

COMBINED_SCHEMA = "terminal-rsi-verbatim-v103-plus-upgrades-v1"


def score_byte_observations(logits: torch.Tensor, bits: torch.Tensor):
    """The original v103 BCE over all eight bits; no surrogate byte CE."""
    loss = F.binary_cross_entropy_with_logits(logits, bits)
    bpb = float((8 * loss.detach() / math.log(2)).item())
    accuracy = float(((logits.detach() > 0) == (bits > 0.5)).float().mean().item())
    return loss, bpb, accuracy


def _full_parent_log_probs(model: VerbatimV103RSI, history: torch.Tensor,
                            aux: torch.Tensor, samples: int):
    """Exact full 256-way teacher distribution from the inherited v103 core."""
    n = min(int(samples), len(history))
    if n <= 0:
        return None
    candidates = torch.arange(256, device=history.device).repeat(n)
    previous = history[:n].repeat_interleave(256, dim=0)
    repeated_aux = aux[:n].repeat_interleave(256, dim=0)
    teacher_bits, _ = model.original(
        previous[:, -32:], candidates, repeated_aux)
    logp = bit_logits_to_byte_log_probs(teacher_bits)
    diagonal = logp.reshape(n, 256, 256).diagonal(dim1=1, dim2=2)
    return F.softmax(diagonal, dim=-1).clamp_min(1e-12).log()


def _accumulate_stats(total: dict, stats: dict, weight: int) -> None:
    for key, value in stats.items():
        if key.endswith("_units"):
            total[key] = total.get(key, 0) + int(value)
        elif isinstance(value, (int, float)):
            total[key] = total.get(key, 0.0) + float(value) * weight


def train_one_batch(model: VerbatimV103RSI, optimizer, history: torch.Tensor,
                    target: torch.Tensor, aux: torch.Tensor,
                    grad_clip: float = 1.0, microbatch: int = 0,
                    upgrade_loss_weight: float = 0.0,
                    boundary_loss_weight: float = 0.0,
                    unit_targets: dict | None = None,
                    unit_loss_weight: float = 0.0,
                    distill_weight: float = 0.0,
                    distill_samples: int = 8,
                    moe_loss_weight: float = 0.01,
                    unit_byte_loss_weight: float = 0.0,
                    unit_byte_distill_weight: float = 0.0):
    """One v103 observe, 100M upgrade, hierarchy, unit heads and distillation."""
    if aux is None or aux.shape != (len(target), 6):
        raise ValueError("Strict v103 training requires the original six causal features")
    if not math.isfinite(grad_clip) or grad_clip <= 0:
        raise ValueError("Gradient clip must be finite and positive")
    for name, value in (("upgrade", upgrade_loss_weight),
                        ("boundary", boundary_loss_weight),
                        ("unit", unit_loss_weight),
                        ("distill", distill_weight),
                        ("moe", moe_loss_weight),
                        ("unit_byte", unit_byte_loss_weight),
                        ("unit_byte_distill", unit_byte_distill_weight)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} loss weight must be finite and nonnegative")
    model.train()
    n = len(target)
    use_aux = upgrade_loss_weight > 0
    use_hierarchy = use_aux and (
        boundary_loss_weight > 0 or unit_loss_weight > 0
        or distill_weight > 0 or moe_loss_weight > 0)
    if microbatch and microbatch < n:
        optimizer.zero_grad(set_to_none=True)
        total_loss = None
        aux_total = None
        boundary_total = 0.0
        distill_total = 0.0
        distill_count = 0
        moe_total = 0.0
        moe_count = 0
        unit_byte_total = 0.0
        unit_byte_count = 0
        unit_total: dict = {}
        unit_weight_total = 0
        total_hits = 0
        for start in range(0, n, microbatch):
            end = min(start + microbatch, n)
            part_target = target[start:end]
            part_history = history[start:end]
            if use_aux:
                if use_hierarchy:
                    logits, bits, current, upgrade_out = model(
                        part_history, part_target, aux[start:end],
                        return_aux=True, return_hierarchy=True)
                else:
                    logits, bits, current = model(
                        part_history, part_target, aux[start:end],
                        return_aux=True)
                    upgrade_out = None
                aux_sum = F.cross_entropy(current, part_target,
                                          reduction="sum")
                aux_total = (aux_sum.detach() if aux_total is None
                             else aux_total + aux_sum.detach())
            else:
                logits, bits = model(part_history, part_target,
                                     aux[start:end])
                aux_sum = None
                upgrade_out = None
            loss_sum = F.binary_cross_entropy_with_logits(
                logits, bits, reduction="sum")
            objective = loss_sum
            if aux_sum is not None:
                objective = objective + upgrade_loss_weight * aux_sum
            if upgrade_out is not None and boundary_loss_weight > 0:
                b_loss = boundary_loss(
                    upgrade_out["boundary_logits"],
                    boundary_targets(part_history).to(
                        upgrade_out["boundary_logits"].device))
                boundary_total += float(b_loss.detach()) * (end - start)
                objective = objective + boundary_loss_weight * b_loss * (end - start)
            if (upgrade_out is not None and unit_targets is not None
                    and unit_loss_weight > 0 and upgrade_out.get("unit_logits")):
                part_units = {key: value[start:end]
                              for key, value in unit_targets.items()}
                u_loss, u_stats = unit_head_loss(
                    upgrade_out["unit_logits"], part_units)
                objective = objective + unit_loss_weight * u_loss * (end - start)
                _accumulate_stats(unit_total, u_stats, end - start)
                unit_weight_total += end - start
            if upgrade_out is not None and distill_weight > 0:
                teacher = _full_parent_log_probs(
                    model, part_history, aux[start:end], distill_samples)
                if teacher is not None:
                    k = teacher.shape[0]
                    student = F.log_softmax(current[:k].float(), dim=-1)
                    kl = F.kl_div(student, teacher, log_target=True,
                                  reduction="sum")
                    objective = objective + distill_weight * kl * (n / k)
                    distill_total += float(kl.detach())
                    distill_count += k
            if upgrade_out is not None and moe_loss_weight > 0:
                moe_value = upgrade_out.get("moe_load_balance")
                if moe_value is not None:
                    objective = objective + moe_loss_weight * float(
                        moe_value.detach()) * (end - start)
                    moe_total += float(moe_value.detach()) * (end - start)
                    moe_count += end - start
            if (upgrade_out is not None and unit_byte_loss_weight > 0
                    and upgrade_out.get("unit_byte_raw") is not None):
                u_byte = F.cross_entropy(
                    upgrade_out["unit_byte_raw"].float(), part_target)
                objective = objective + unit_byte_loss_weight * u_byte * (end - start)
                unit_byte_total += float(u_byte.detach()) * (end - start)
                unit_byte_count += end - start
            if (upgrade_out is not None and unit_byte_distill_weight > 0
                    and upgrade_out.get("unit_byte_raw") is not None):
                teacher = _full_parent_log_probs(
                    model, part_history, aux[start:end], distill_samples)
                if teacher is not None:
                    k = teacher.shape[0]
                    student = F.log_softmax(
                        upgrade_out["unit_byte_raw"][:k].float(), dim=-1)
                    kl = F.kl_div(student, teacher, log_target=True,
                                  reduction="sum")
                    objective = objective + (
                        unit_byte_distill_weight * kl * (n / k))
            (objective / n).backward()
            total_loss = (loss_sum.detach() if total_loss is None
                          else total_loss + loss_sum.detach())
            total_hits += int(((logits.detach() > 0) == (bits > 0.5)).sum())
        grad_norm = float(nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip))
        optimizer.step()
        loss = float(total_loss) / n
        bpb = float(8 * loss / math.log(2))
        accuracy = total_hits / (n * 8)
        model.original.observe_online_bits(history[:, -32:], target, aux)
        model.observe_verified_byte(bpb, accuracy)
        report = {"bpb": bpb, "bit_accuracy": accuracy,
                  "grad_norm": grad_norm, "grad_clip": grad_clip}
        if use_aux:
            report["upgrade_bpb"] = float(
                (float(aux_total) / n) / math.log(2))
        if boundary_loss_weight > 0:
            report["boundary_bce"] = boundary_total / n
        for key, value in unit_total.items():
            report[key] = (float(value) / max(unit_weight_total, 1)
                           if not key.endswith("_units") else int(value))
        if distill_count:
            report["distill_kl"] = distill_total / distill_count
        if moe_count:
            report["moe_load_balance"] = moe_total / moe_count
        if unit_byte_count:
            report["unit_byte_bpb"] = float(
                (unit_byte_total / unit_byte_count) / math.log(2))
        return report
    if use_aux:
        if use_hierarchy:
            logits, bits, current, upgrade_out = model(
                history, target, aux, return_aux=True,
                return_hierarchy=True)
        else:
            logits, bits, current = model(history, target, aux, return_aux=True)
            upgrade_out = None
        loss, bpb, accuracy = score_byte_observations(logits, bits)
        aux_loss = F.cross_entropy(current, target)
        objective = loss + upgrade_loss_weight * aux_loss
        b_loss = None
        if upgrade_out is not None and boundary_loss_weight > 0:
            b_loss = boundary_loss(
                upgrade_out["boundary_logits"],
                boundary_targets(history).to(
                    upgrade_out["boundary_logits"].device))
            objective = objective + boundary_loss_weight * b_loss
        if (upgrade_out is not None and unit_targets is not None
                and unit_loss_weight > 0 and upgrade_out.get("unit_logits")):
            u_loss, u_stats = unit_head_loss(
                upgrade_out["unit_logits"], unit_targets)
            objective = objective + unit_loss_weight * u_loss
        else:
            u_stats = {}
        if upgrade_out is not None and distill_weight > 0:
            teacher = _full_parent_log_probs(
                model, history, aux, distill_samples)
            if teacher is not None:
                k = teacher.shape[0]
                student = F.log_softmax(current[:k].float(), dim=-1)
                objective = objective + distill_weight * F.kl_div(
                    student, teacher, log_target=True, reduction="batchmean")
        moe_value = (upgrade_out.get("moe_load_balance")
                     if upgrade_out is not None else None)
        if moe_value is not None and moe_loss_weight > 0:
            objective = objective + moe_loss_weight * moe_value.float()
        u_byte_value = None
        if (upgrade_out is not None and unit_byte_loss_weight > 0
                and upgrade_out.get("unit_byte_raw") is not None):
            u_byte_value = F.cross_entropy(
                upgrade_out["unit_byte_raw"].float(), target)
            objective = objective + unit_byte_loss_weight * u_byte_value
        if (upgrade_out is not None and unit_byte_distill_weight > 0
                and upgrade_out.get("unit_byte_raw") is not None):
            teacher = _full_parent_log_probs(
                model, history, aux, distill_samples)
            if teacher is not None:
                k = teacher.shape[0]
                objective = objective + unit_byte_distill_weight * F.kl_div(
                    F.log_softmax(upgrade_out["unit_byte_raw"][:k].float(),
                                  dim=-1),
                    teacher, log_target=True, reduction="batchmean")
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        upgrade_bpb = float(aux_loss.detach() / math.log(2))
    else:
        logits, bits = model(history, target, aux)
        loss, bpb, accuracy = score_byte_observations(logits, bits)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        upgrade_bpb = None
        b_loss = None
        u_stats = {}
        moe_value = None
        u_byte_value = None
        upgrade_out = None
    grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
    optimizer.step()
    model.original.observe_online_bits(history[:, -32:], target, aux)
    model.observe_verified_byte(bpb, accuracy)
    report = {"bpb": bpb, "bit_accuracy": accuracy,
              "grad_norm": grad_norm, "grad_clip": grad_clip}
    if upgrade_bpb is not None:
        report["upgrade_bpb"] = upgrade_bpb
    if b_loss is not None:
        report["boundary_bce"] = float(b_loss.detach())
    report.update(u_stats)
    if moe_value is not None:
        report["moe_load_balance"] = float(moe_value.detach())
    if u_byte_value is not None:
        report["unit_byte_bpb"] = float(
            u_byte_value.detach() / math.log(2))
    if upgrade_out is not None and distill_weight > 0:
        teacher = _full_parent_log_probs(
            model, history, aux, distill_samples)
        if teacher is not None:
            # Reported value for the full-batch path; the loss above already
            # used the same teacher distribution.
            k = teacher.shape[0]
            report["distill_kl"] = float(F.kl_div(
                F.log_softmax(current[:k].float(), dim=-1), teacher,
                log_target=True, reduction="batchmean").detach())
    return report


def finish_rsi_round(model: VerbatimV103RSI, optimizer):
    """Replay the archived terminal self-modification after EVERY training round.

    Two real commands and verifier-derived actor/critic Adam updates. This
    happens with zero byte-training steps too, exactly as in original v103.
    """
    return terminal_feedback(model.original, optimizer)


def build_causal_contexts(raw_bytes: torch.Tensor, target_positions: torch.Tensor,
                          history: int):
    """Longer CPU/T4/Blackwell context, WITHOUT changing the original 32 bytes.

    Book start pads with zeros. The current target is appended only in the
    separate final column and never appears in the model's context.
    """
    positions = target_positions.to("cpu", torch.long).reshape(-1)
    if history < 32 or positions.numel() == 0:
        raise ValueError("Need at least the original 32 observed byte positions")
    if int(positions.min()) < 32 or int(positions.max()) >= len(raw_bytes):
        raise ValueError("Invalid original v103 training target positions")
    deltas = torch.arange(history, 0, -1, dtype=torch.long)
    indices = positions[:, None] - deltas[None, :]
    causal = indices >= 0
    past = raw_bytes[indices.clamp_min(0)].long()
    past = torch.where(causal, past, torch.zeros_like(past))
    target = raw_bytes[positions].long()
    return torch.cat((past, target[:, None]), dim=-1)


def build_window_cache(raw_bytes: torch.Tensor, history: int,
                       device: torch.device):
    """Zero-padded sliding context view for GPU-side batch assembly.

    Returns ``(windows, targets)`` where ``windows[p]`` is exactly the
    ``history`` observed bytes before target position ``p`` and ``targets[p]``
    is byte ``p``.  The view aliases a single padded copy of the stream; a
    batch gather allocates only the batch, not the full N x history matrix.
    """
    if raw_bytes.ndim != 1 or raw_bytes.dtype != torch.uint8:
        raise ValueError("Window cache requires a 1-D uint8 byte stream")
    raw = raw_bytes.to(device)
    padded = torch.cat((
        torch.zeros(history, dtype=torch.uint8, device=device), raw))
    windows = padded.unfold(0, history, 1)[:raw.numel()]
    return windows, raw


def build_book_disjoint_probes(book_path: Path, n: int, history: int,
                              full_dev: torch.Tensor,
                              full_aux: torch.Tensor,
                              full_local: torch.Tensor | None,
                              device: torch.device):
    """Matched positions across books; no future/foreign-book context."""
    from .legacy_v103_local_copy import read_books
    books, skip, digest = read_books(book_path, expected_sha256=None)
    total_books, per, columns = full_dev.shape
    if len(books) != total_books or columns != 33:
        raise AssertionError("Original heldout shape differs from expected v103")
    chosen = torch.linspace(0, total_books * per - 1,
                            min(n, total_books * per)).long()
    windows = []
    for k in chosen.tolist():
        book_idx, pos_idx = divmod(k, per)
        target_at = skip + pos_idx
        book = books[book_idx]
        if target_at >= len(book):
            raise AssertionError("Independent DEV target exceeded its own book")
        fragment = book[max(0, target_at - history): target_at]
        # Byte zero is padding, never a token from a different document.
        pad = [0] * (history - len(fragment))
        windows.append(pad + list(fragment) + [book[target_at]])
    probe = torch.tensor(windows, dtype=torch.long, device=device)
    # Verify exact last-32 history/next-byte pair of EVERY sampled probe.
    oracle = full_dev.reshape(-1, 33)[chosen.to(full_dev.device)]
    if not torch.equal(probe[:, -33:], oracle.long()):
        raise AssertionError("Long-context DEV changed historical RSI 32-byte inputs")
    flat_aux = full_aux.reshape(-1, 6)[chosen.to(full_aux.device)]
    flat_local = (full_local.reshape(-1, 4)[chosen.to(full_local.device)]
                  if full_local is not None else None)
    return probe, flat_aux, flat_local, digest


def verify_parent_source(source: dict, raw: bytes, ranks_path: Path,
                         feature_info: dict, dev_digest: str,
                         *, explicit_new_dev: bool = False):
    actual_sha = hashlib.sha256(raw).hexdigest()
    if source.get("schema") != PARITY_SCHEMA:
        raise ValueError("Not the actual v103 full-system checkpoint")
    if source.get("source_sha256") != actual_sha:
        raise ValueError("Original training stream SHA-256 mismatch; NEVER reset sampler silently")
    previous = source.get("v103_feature_manifest")
    if previous is None:
        raise ValueError("Missing historical v103 feature manifest")
    if previous != feature_info:
        raise ValueError("BPE/word/match v103 features differ; refuse false verbatim parity")
    orig_dev = source.get("holdout_digests", {}).get("dev")
    if orig_dev and orig_dev != dev_digest and not explicit_new_dev:
        raise ValueError("Independent DEV digest differs from original checkpoint")
    sampler = dict(source.get("sampler") or {})
    window_count = len(raw) - PARITY_CONTEXT
    if (sampler.get("schema") != "full-cycle-affine-v1"
            or sampler.get("train_positions") != window_count
            or sampler.get("cursor", -1) < 0
            or math.gcd(sampler.get("stride", 0), window_count) != 1):
        raise ValueError("Cannot faithfully restore original full-cycle-affine sampler")
    return sampler


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=PRESETS, default="cpu")
    p.add_argument("--legacy-checkpoint", type=Path)
    p.add_argument("--allow-parent-mismatch", action="store_true",
                   help="Experimental: load verified parent weights even when the "
                        "train/DEV stream digests differ from the parent provenance")
    p.add_argument("--legacy-core-lr-scale", type=float, default=0.1,
                   help="LR multiplier for the inherited v103 groups on warm start")
    p.add_argument("--fresh-start", action="store_true",
                   help="Explicit disposable CPU bootstrap from original v103 architecture, no inherited weights")
    p.add_argument("--seed", type=int, default=2026,
                   help="Deterministic from-scratch initialization; ignored by resumed checkpoints")
    p.add_argument("--resume", type=Path, help="Resume exact-mode unified candidate with all Adam/RSI state")
    p.add_argument("--train-stream", type=Path, required=True)
    p.add_argument("--bpe-merges", type=Path, required=True)
    p.add_argument("--dev-bin", type=Path, required=True,
                   help="Book-bounded NaNCM win_dev.bin, historic or new English DEV")
    p.add_argument("--dev-manifest", type=Path,
                   help="Required for NEW DEV; checks book list and TRAIN/DEV SHA without opening TEST")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metrics-output", type=Path,
                   help="Optional stable JSON receipt with matched DEV metrics and 1000-update timing")
    p.add_argument("--champion-dir", type=Path,
                   help="Protected best-checkpoint directory; compare only matching DEV, preserve all prior winners")
    p.add_argument("--seconds", type=float, default=0.0,
                   help="Optional wall-clock budget; cannot combine with --updates")
    p.add_argument("--updates", type=int,
                   help="Exact number of optimizer updates, e.g. 1000 on CPU; independent of speed")
    p.add_argument("--progress-every", type=int, default=10,
                   help="Emit compact step + RSI self-modification stats every N updates")
    p.add_argument("--save-every", type=int, default=0,
                   help="Write a rolling recoverable checkpoint every N kept updates; 0=final only")
    p.add_argument("--smoke", action="store_true",
                   help="Blackwell memory/throughput smoke: skip full checkpoint/champion writes")
    p.add_argument("--no-gradient-checkpointing", action="store_true",
                   help="Trade VRAM for compute: keep full upgrade activations instead of recomputing")
    p.add_argument("--no-gpu-window-cache", action="store_true",
                   help="Disable the Blackwell GPU sliding-window cache (CPU/T4 unaffected)")
    p.add_argument("--cpu-threads", type=int, choices=(1, 2, 4, 8, 16), default=2,
                   help="CPU worker math threads: 1, 2, 4, 8 or 16; for GitHub/Molab")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--microbatch", type=int, default=0,
                   help="Split the effective batch for activation memory; 0=full batch")
    p.add_argument("--upgrade-loss-weight", type=float, default=0.0,
                   help="Direct byte CE on the additive upgrade branch (v4 training)")
    p.add_argument("--boundary-loss-weight", type=float, default=0.02,
                   help="Auxiliary causal subword/word/sentence boundary BCE")
    p.add_argument("--unit-loss-weight", type=float, default=0.1,
                   help="Auxiliary token/word/sentence head loss")
    p.add_argument("--distill-weight", type=float, default=0.5,
                   help="Full 256-way KL from the inherited parent into the 100M branch")
    p.add_argument("--distill-samples", type=int, default=8,
                   help="Samples per microbatch used for the 256-way teacher distribution")
    p.add_argument("--moe-loss-weight", type=float, default=0.01,
                   help="Load-balancing regularizer for the 256-expert MoE")
    p.add_argument("--unit-byte-loss-weight", type=float, default=0.1,
                   help="Direct CE on the current-unit -> byte decoder")
    p.add_argument("--unit-byte-distill-weight", type=float, default=0.5,
                   help="256-way parent KL into the unit -> byte decoder")
    p.add_argument("--ablation", choices=(
        "full", "no_loops", "no_bottom_up", "no_top_down", "no_lateral",
        "no_pair", "no_units", "no_boundary_pooling"), default="full",
                   help="Hierarchy ablation variant for the current run")
    p.add_argument("--compile-upgrade", action="store_true",
                   help="Try torch.compile on the 100M upgrade branch; eager fallback")
    p.add_argument("--rsi-cycle-updates", type=int, default=50,
                   help="RSI self-construction proposal/verification every N byte steps; 0 disables for ablation")
    p.add_argument("--rsi-version", choices=("v3", "v4", "off"), default="v3",
                   help="RSI controller: legacy v3 gate bandit, v4 counterfactual branch controller, or off")
    p.add_argument("--rsi-v4-cycle-updates", type=int, default=200,
                   help="Run a v4 counterfactual branching cycle every N kept updates")
    p.add_argument("--rsi-v4-horizon", type=int, default=10,
                   help="Real training updates per v4 branch")
    p.add_argument("--rsi-v4-candidates", type=int, default=2,
                   help="Number of proposed v4 branches per cycle (plus no-op)")
    p.add_argument("--rsi-v4-shards", type=int, default=4,
                   help="Disjoint internal TRAIN probe shards per v4 decision")
    p.add_argument("--rsi-v4-shard-positions", type=int, default=256,
                   help="Target bytes per v4 probe shard")
    p.add_argument("--rsi-v4-eval-chunk", type=int, default=8,
                   help="Mini-batch size for CPU probe evaluation; small is faster")
    p.add_argument("--rsi-holdout-stream", type=Path,
                   help="Book-disjoint raw bytes reserved only for RSI probes")
    p.add_argument("--rsi-v4-min-gain", type=float, default=0.001,
                   help="Minimum paired BPB gain for a v4 branch to be accepted")
    p.add_argument("--rsi-v4-z", type=float, default=2.0,
                   help="Paired standard-error multiplier for v4 acceptance")
    p.add_argument("--rsi-fast-adapters", type=int, default=0,
                   help="Cached rank-r adapter proposals per fast RSI round")
    p.add_argument("--rsi-fast-rounds", type=int, default=4)
    p.add_argument("--rsi-fast-steps", type=int, default=3)
    p.add_argument("--rsi-fast-lr", type=float, default=0.02)
    p.add_argument("--rsi-fast-rank", type=int, default=8)
    p.add_argument("--rsi-fast-cycle-updates", type=int, default=4)
    p.add_argument("--rsi-fast-min-gain", type=float, default=0.0005)
    p.add_argument("--rsi-fast-z", type=float, default=1.5)
    p.add_argument("--rsi-fast-shards", type=int, default=8)
    p.add_argument("--rsi-v4-meta", type=Path,
                   help="Optional v4 meta/replay JSON from a previous run")
    p.add_argument("--rsi-v4-critic-steps", type=int, default=32)
    p.add_argument("--rsi-v4-policy-steps", type=int, default=16)
    p.add_argument("--rsi-v4-branch-batch", type=int, default=64,
                   help="Small fast batch for branch train/eval; main batch remains --batch")
    p.add_argument("--rsi-v4-cycles-per-tick", type=int, default=16,
                   help="Mutations evaluated in one RSI tick")
    p.add_argument("--rsi-v4-parallel-replicas", type=int, default=0,
                   help="Run candidate branches on K cloned models in parallel CUDA streams")
    p.add_argument("--rsi-v4-entropy", type=float, default=0.01)
    p.add_argument("--legacy-terminal-demo", action="store_true",
                   help="Diagnostic ONLY: replay v103 fixed shell reward (+1/-1); never use to optimize low BPB")
    p.add_argument("--rsi-probe-positions", type=int, default=8,
                   help="Causal internal TRAIN probes for RSI verification; DEV/TEST never used")
    p.add_argument("--rsi-min-bpb-gain", type=float, default=0.002,
                   help="Minimum mean BPB gain on TWO disjoint internal TRAIN probes")
    p.add_argument("--max-dev-positions", type=int, default=16)
    p.add_argument("--topk-probes", type=int, default=8,
                   help="Bound exact 256-way byte top1/top5 probes to avoid 256x CPU evaluations")
    p.add_argument("--hierarchical-eval-books", type=int, default=0,
                   help="Evaluate contiguous DEV excerpts and report byte/BPE/word/sentence metrics")
    p.add_argument("--hierarchical-eval-corpus-dir", type=Path,
                   help="Extracted NaNROUTER corpus for hierarchical DEV excerpts")
    p.add_argument("--hierarchical-eval-max-positions", type=int, default=0,
                   help="Positions per book for hierarchical eval; 0=all after context")
    p.add_argument("--hierarchical-eval-batch", type=int, default=8,
                   help="Windows per 256-way byte-probability batch")
    p.add_argument("--generation-steps", type=int, default=0,
                   help="Byte-driven generation probe after training; 0=off")
    p.add_argument("--generation-prompt", default="The ")
    p.add_argument("--generation-temperature", type=float, default=0.9)
    p.add_argument("--generation-top-k", type=int, default=0)
    p.add_argument("--generation-word-onset-weight", type=float, default=0.3)
    p.add_argument("--fast-generation", action="store_true",
                   help="One-pass KV-cache generation")
    p.add_argument("--upgrade-context", type=int, default=None,
                   help="Observed causal bytes for additive model, up to preset context")
    p.add_argument("--reference-full256", type=Path,
                   help="Optional original C 256-probability oracle (matched SHA)")
    args = p.parse_args(argv)
    started = time.monotonic()
    if args.smoke:
        args.save_every = 0
    if (args.seconds < 0 or args.batch < 1 or args.microbatch < 0
            or not math.isfinite(args.upgrade_loss_weight)
            or args.upgrade_loss_weight < 0
            or args.max_dev_positions < 1
            or args.topk_probes < 1 or args.progress_every < 1
            or args.save_every < 0
            or (args.updates is not None and args.updates < 0)):
        p.error("Nonnegative seconds/updates/microbatch/upgrade-loss and positive batch/dev/topk/progress sample required")
    if (args.hierarchical_eval_books < 0
            or args.hierarchical_eval_batch < 1
            or args.hierarchical_eval_max_positions < 0):
        p.error("Hierarchical eval needs nonnegative books/positions and positive batch")
    if (args.generation_steps < 0 or args.generation_temperature <= 0
            or args.generation_top_k < 0):
        p.error("Generation needs nonnegative steps/top-k and positive temperature")
    if args.legacy_core_lr_scale <= 0:
        p.error("Parent warm-start LR scale must be positive")
    for name, value in (("unit", args.unit_loss_weight),
                        ("distill", args.distill_weight),
                        ("moe", args.moe_loss_weight),
                        ("unit_byte", args.unit_byte_loss_weight),
                        ("unit_byte_distill", args.unit_byte_distill_weight)):
        if value < 0:
            p.error(f"{name} loss weight must be nonnegative")
    if args.distill_samples < 0:
        p.error("Distill samples must be nonnegative")
    if args.hierarchical_eval_books and not args.hierarchical_eval_corpus_dir:
        p.error("--hierarchical-eval-corpus-dir is required for hierarchical DEV evaluation")
    if (args.rsi_cycle_updates < 0 or args.rsi_probe_positions < 1
            or not math.isfinite(args.rsi_min_bpb_gain) or args.rsi_min_bpb_gain <= 0):
        p.error("RSI cycle >=0, probe count positive, and minimum BPB gain finite and positive required")
    if args.updates is not None and args.seconds > 0:
        p.error("Choose --updates OR --seconds, not both")
    if args.rsi_version == "v4":
        if (args.rsi_v4_cycle_updates < 1 or args.rsi_v4_horizon < 1
                or args.rsi_v4_candidates < 0 or args.rsi_v4_shards < 1
                or args.rsi_v4_shard_positions < 1
                or not math.isfinite(args.rsi_v4_min_gain)
                or args.rsi_v4_min_gain < 0
                or not math.isfinite(args.rsi_v4_z) or args.rsi_v4_z < 0
                or args.rsi_v4_critic_steps < 0 or args.rsi_v4_policy_steps < 0
                or args.rsi_v4_branch_batch < 1
                or args.rsi_v4_cycles_per_tick < 1
                or args.rsi_v4_parallel_replicas < 0
                or not math.isfinite(args.rsi_v4_entropy) or args.rsi_v4_entropy < 0):
            p.error("v4 needs positive cycle/horizon/shards/positions and finite nonnegative thresholds")
        if args.rsi_v4_eval_chunk < 1:
            p.error("v4 probe evaluation chunk must be positive")
        if args.rsi_holdout_stream and not args.rsi_holdout_stream.is_file():
            p.error("RSI holdout stream does not exist")
    if args.rsi_version != "v3":
        # v4 owns the loop directly; v3's fixed gate builder must not also run.
        args.rsi_cycle_updates = 0
    if args.mode == "cpu":
        torch.set_num_threads(args.cpu_threads)
    if args.fresh_start == (args.legacy_checkpoint is not None):
        p.error("Choose exactly one: protected --legacy-checkpoint OR disposable --fresh-start")
    if args.legacy_checkpoint and args.output.resolve() == args.legacy_checkpoint.resolve():
        p.error("Never overwrite the original trained v103 checkpoint")
    if args.output.exists() and (args.resume is None or args.output.resolve() != args.resume.resolve()):
        p.error("Destination exists; use --resume of that exact unified candidate")
    device = device_for(args.mode)
    upgrade_context = args.upgrade_context or PRESETS[args.mode].context
    if not 32 <= upgrade_context <= PRESETS[args.mode].context:
        p.error("Upgrade context must be >=32 and <= selected preset capacity")
    source = (torch.load(args.legacy_checkpoint, map_location="cpu", weights_only=True)
              if args.legacy_checkpoint else None)
    if args.fresh_start and not args.resume:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    raw = args.train_stream.read_bytes()
    ranks = read_bpe_ranks(args.bpe_merges)
    if not ranks:
        raise RuntimeError("Original v103 BPE ranks required; no placeholder features")
    # Entire historical 10M stream: original v103 features and causal historic
    # index exactly as in archived cell 14. No cross-DEV book contamination.
    train_features, memory = build_train_features(raw, ranks)
    manifest = feature_manifest(raw, ranks, train_features)
    manifest["bpe_merge_sha256"] = hashlib.sha256(args.bpe_merges.read_bytes()).hexdigest()
    heldout = heldout_windows(str(args.dev_bin), device, history=PARITY_CONTEXT)
    if heldout is None:
        raise RuntimeError("Book-disjoint DEV cannot be omitted")
    dev, dev_digest = heldout
    historical_dev_digest = ((source.get("holdout_digests") or {}).get("dev")
                             if source is not None else None)
    using_new_dev = historical_dev_digest != dev_digest
    fresh_dev_metadata = None
    if args.dev_manifest:
        fresh_dev_metadata = validate_fresh_dev(
            args.dev_bin, args.dev_manifest,
            train_sha256=hashlib.sha256(raw).hexdigest())
        if not using_new_dev:
            raise ValueError("NEW DEV manifest supplied for historical original DEV")
    elif using_new_dev:
        raise ValueError("Different DEV requires explicit --dev-manifest with verified contents")
    if using_new_dev and args.reference_full256:
        raise ValueError("Historic frozen C full256 positions cannot score a NEW DEV")
    warm_parent = source is not None and args.allow_parent_mismatch
    sampler = (
        {"schema": "full-cycle-affine-v1", "cursor": 0,
         "stride": co_prime_stride(len(raw) - PARITY_CONTEXT, args.seed),
         "train_positions": len(raw) - PARITY_CONTEXT}
        if warm_parent else
        verify_parent_source(
            source, raw, args.bpe_merges, manifest, dev_digest,
            explicit_new_dev=fresh_dev_metadata is not None)
        if source is not None else
        {"schema": "full-cycle-affine-v1", "cursor": 0,
         "stride": co_prime_stride(len(raw) - PARITY_CONTEXT, args.seed),
         "train_positions": len(raw) - PARITY_CONTEXT})
    # CPU-resident features avoid filling 96GB GPU memory with 10M × 6 fields.
    train_data, _ = make_windows(raw, torch.device("cpu"), train_all=True,
                                 history=PARITY_CONTEXT)
    aux_data = torch.from_numpy(train_features[PARITY_CONTEXT:].copy()).to(dtype=torch.int32)
    if len(train_data) != len(aux_data):
        raise AssertionError("Original feature-target offset mismatch")
    unit_labels = None
    if args.unit_loss_weight > 0:
        unit_labels = build_unit_labels(raw, args.bpe_merges)
        print(
            "Unit labels:",
            f"tokens={len(unit_labels.token_vocab)}",
            f"words={len(unit_labels.word_vocab)}",
            f"sentences={len(unit_labels.sentence_vocab)}",
            flush=True,
        )
    dev_aux = heldout_auxiliary(str(args.dev_bin), device, ranks, memory)
    raw_tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    model = VerbatimV103RSI(args.mode).to(device)
    ablation_flags = {
        "bottom_up": True, "top_down": True, "lateral": True,
        "pair": True, "units": True, "boundary_pooling": True,
    }
    if args.ablation == "no_loops":
        ablation_flags.update(
            {"bottom_up": False, "top_down": False, "lateral": False,
             "pair": False})
    elif args.ablation == "no_bottom_up":
        ablation_flags["bottom_up"] = False
    elif args.ablation == "no_top_down":
        ablation_flags["top_down"] = False
    elif args.ablation == "no_lateral":
        ablation_flags["lateral"] = False
    elif args.ablation == "no_pair":
        ablation_flags["pair"] = False
    elif args.ablation == "no_units":
        ablation_flags["units"] = False
    elif args.ablation == "no_boundary_pooling":
        ablation_flags["boundary_pooling"] = False
    model.upgrades.hierarchy.ablation.update(ablation_flags)
    print("Hierarchy ablation:", ablation_flags, flush=True)
    compiled_upgrade = None
    if args.compile_upgrade:
        from .speed import compile_upgrade
        compiled_upgrade = compile_upgrade(model)
        if compiled_upgrade is not None:
            model._compiled_upgrades = compiled_upgrade
    if args.no_gradient_checkpointing:
        model.upgrades.gradient_checkpointing = False
    optimizer = build_v103_optimizer(
        model, freeze_residual_gate=(args.rsi_version == "v4"))
    provenance = (restore_v103_parent(args.legacy_checkpoint, model, optimizer)
                  if source is not None else
                  {"legacy_step": 0, "legacy_checkpoint_sha256": None,
                   "fresh_start_untrained_v103": True,
                   "original_optimizer_histories_inherited": False})
    if source is not None:
        # Parent warm start: fine-tune the inherited v103 core conservatively.
        # The 100M upgrade branch keeps its full LR and learns the hierarchy.
        for group in optimizer.param_groups[:2]:
            group["lr"] = float(group["lr"]) * args.legacy_core_lr_scale
    dev_local = None
    if model.original.causal_copy_enabled:
        # The original read-only local-copy specialist must receive true
        # target-causal DEV book matches. Zero local features change BPB.
        local_np, local_digest = book_local_features(
            args.dev_bin, expected_sha256=dev_digest)
        if local_digest != dev_digest:
            raise AssertionError("Original local-copy DEV digest mismatch")
        dev_local = torch.from_numpy(local_np).to(device=device, dtype=torch.int32)
        if dev_local.shape != (*dev.shape[:2], 4):
            raise AssertionError("Causal local specialist misaligned with DEV targets")
    # Select genuinely book-disjoint long windows; check the LAST 32 bytes
    # against the original heldout 33-column rows position for position.
    probes, probe_aux, probe_local, probe_digest = build_book_disjoint_probes(
        args.dev_bin, args.max_dev_positions, upgrade_context,
        dev, dev_aux, dev_local, device)
    if probe_digest != dev_digest:
        raise AssertionError("Original DEV identity changed during window expansion")
    actual = build_causal_contexts(
        raw_tensor, torch.arange(args.batch) + PARITY_CONTEXT,
        upgrade_context).to(device)
    actual_aux = aux_data[:args.batch].to(device=device)
    proof = exact_parent_parity(model, actual[:, :-1], actual[:, -1], actual_aux)
    print("VERBATIM v103 original checkpoint, gate=0:", json.dumps(proof))
    provenance.update({"source_sha256": hashlib.sha256(raw).hexdigest(),
                       "feature_manifest": manifest, "dev_sha256": dev_digest,
                       "upgrade_context": upgrade_context,
                       "original_two_adam_groups": [g["lr"] for g in optimizer.param_groups[:2]],
                       "fresh_dev_manifest_sha256": (hashlib.sha256(args.dev_manifest.read_bytes()).hexdigest()
                                                     if args.dev_manifest else None)})
    step = int(source["steps"]) if source is not None else 0
    events = []
    if args.resume:
        resumed = resume_combined(args.resume, model, optimizer,
                                  source_sha256=provenance["source_sha256"],
                                  feature_manifest=manifest, dev_sha256=dev_digest)
        step = int(resumed["step"])
        sampler = resumed["sampler"]
        events = resumed["events"]
        provenance = resumed["legacy_provenance"]
        if provenance.get("upgrade_context") != upgrade_context:
            raise ValueError("Changed causal upgrade context across RSI continuation")
        if len(train_data) != sampler["train_positions"]:
            raise ValueError("Changed coverage length in unified RSI resume")
        state = resumed["rng"]
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        print("Exact unified RSI checkpoint resumed at", step)
    elif source is not None:
        if "torch_rng" in source:
            torch.set_rng_state(source["torch_rng"])
        if device.type == "cuda" and source.get("cuda_rng"):
            torch.cuda.set_rng_state_all(source["cuda_rng"])
        if "python_rng" in source:
            random.setstate(source["python_rng"])
    # RSI owns the training loop. Its action policy proposes real changes to
    # the specialist router, feedback gain and additive-capacity gate; only
    # verified internal TRAIN-probe improvements survive a mutation cycle.
    # Public book-disjoint DEV is never consulted for self-construction.
    rsi_builder = None
    if args.rsi_cycle_updates:
        # Two DISJOINT causal partitions of TRAIN only. No held-out DEV/TEST
        # samples can influence RSI actor/critic rewards or mutation acceptance.
        meta_count = min(args.rsi_probe_positions, len(train_data) // 2,
                         2 if device.type == "cuda" else args.rsi_probe_positions)
        if meta_count < 1:
            raise ValueError("Need >=2 distinct training positions for dual RSI probes")
        paired_indices = torch.linspace(0, len(train_data) - 1, 2 * meta_count).long()
        verify_indices, audit_indices = paired_indices[::2], paired_indices[1::2]
        if torch.isin(verify_indices, audit_indices).any():
            raise AssertionError("RSI verifier/audit causal targets overlap")
        verify_window = build_causal_contexts(
            raw_tensor, verify_indices + PARITY_CONTEXT, upgrade_context).to(device)
        audit_window = build_causal_contexts(
            raw_tensor, audit_indices + PARITY_CONTEXT, upgrade_context).to(device)
        rsi_builder = RSISelfBuilder(
            model, optimizer, verify_window[:, :-1], verify_window[:, -1],
            aux_data[verify_indices].to(device),
            interval=args.rsi_cycle_updates,
            audit_history=audit_window[:, :-1], audit_targets=audit_window[:, -1],
            audit_auxiliary=aux_data[audit_indices].to(device),
            min_gain_bpb=args.rsi_min_bpb_gain)
        if args.resume and resumed["rng"].get("rsi_self_builder"):
            rsi_builder.load_state_dict(resumed["rng"]["rsi_self_builder"])
        elif args.resume and resumed["rng"].get("rsi_self_builder") is None:
            # Old checkpoints had no RSI topology builder. Preserve ALL
            # inherited model/Adam/sampler state and begin a new builder
            # ledger at update zero; this is an additive migration only.
            print("Upgrading pre-builder checkpoint: existing model/Adam state preserved", flush=True)
        print("RSI self-construction CORE enabled:", rsi_builder.state_dict(), flush=True)
    elif args.resume and resumed["rng"].get("rsi_self_builder"):
        raise ValueError("Cannot disable inherited RSI core without explicit migration")
    v4_controller = None
    if args.rsi_version == "v4":
        probe_raw = raw_tensor
        probe_aux = aux_data
        probe_source = "full TRAIN stream"
        if args.rsi_holdout_stream:
            holdout_raw = args.rsi_holdout_stream.read_bytes()
            if len(holdout_raw) < (args.rsi_v4_shards * args.rsi_v4_shard_positions
                                   + upgrade_context + 2):
                raise ValueError("RSI holdout stream is too short for the requested probes")
            holdout_features, _ = build_train_features(holdout_raw, ranks)
            probe_raw = torch.frombuffer(bytearray(holdout_raw), dtype=torch.uint8)
            probe_aux = torch.from_numpy(
                holdout_features[PARITY_CONTEXT:].copy()).to(dtype=torch.int32)
            probe_source = str(args.rsi_holdout_stream)
            print(
                "RSI-v4 probe source: book-disjoint holdout",
                probe_source,
                "bytes", len(holdout_raw),
                "sha256", hashlib.sha256(holdout_raw).hexdigest()[:16],
                flush=True,
            )
        probe = ProbeShards.build(
            probe_raw, probe_aux,
            n_shards=args.rsi_v4_shards,
            positions_per_shard=args.rsi_v4_shard_positions,
            history=upgrade_context, seed=args.seed,
        )
        v4_config = RSIv4Config(
            cycle_updates=args.rsi_v4_cycle_updates,
            horizon=args.rsi_v4_horizon,
            candidates=args.rsi_v4_candidates,
            n_shards=args.rsi_v4_shards,
            positions_per_shard=args.rsi_v4_shard_positions,
            eval_chunk=args.rsi_v4_eval_chunk,
            min_gain=args.rsi_v4_min_gain,
            z_score=args.rsi_v4_z,
            batch_size=args.batch,
            fixed_batch=args.mode in ("t4", "blackwell"),
            branch_batch_size=args.rsi_v4_branch_batch,
            grad_clip=1.0,
            seed=args.seed,
            critic_steps=args.rsi_v4_critic_steps,
            policy_steps=args.rsi_v4_policy_steps,
            entropy_coef=args.rsi_v4_entropy,
            target_updates=args.updates or 0,
        )
        v4_controller = RSIv4Controller(
            model, optimizer, probe, v4_config, device=device,
            meta_path=args.rsi_v4_meta,
        )
        if args.resume and resumed["rng"].get("rsi_v4"):
            v4_controller.load_state_dict(resumed["rng"]["rsi_v4"])
            model.upgrades.router_temperature = float(
                v4_controller.current_action.values()["router_temperature"])
        print(
            "RSI-v4 controller enabled:",
            json.dumps({
                "cycle_updates": v4_config.cycle_updates,
                "horizon": v4_config.horizon,
                "candidates": v4_config.candidates,
                "shards": v4_config.n_shards,
                "positions_per_shard": v4_config.positions_per_shard,
                "min_gain": v4_config.min_gain,
                "z_score": v4_config.z_score,
                "meta_path": str(args.rsi_v4_meta) if args.rsi_v4_meta else None,
            }, sort_keys=True),
            flush=True,
        )
    before = byte_metrics_wrapped(model, probes, n=None,
                                  topk_n=min(args.topk_probes, len(probes)), auxiliary=probe_aux,
                                  local=probe_local)
    print("DEV before:", before)

    def _bit_logit_snapshot() -> torch.Tensor:
        """Per-probe bit logits, so before/after changes cannot look stale."""
        model_device = next(model.parameters()).device
        windows = probes.reshape(-1, probes.shape[-1]).long().to(model_device)
        aux_flat = (probe_aux.reshape(-1, 6).long().to(model_device)
                    if probe_aux is not None else None)
        local_flat = (probe_local.reshape(-1, 4).long().to(model_device)
                      if probe_local is not None else None)
        rows = []
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for index in range(len(windows)):
                    logits, _ = model(
                        windows[index:index + 1, :-1],
                        windows[index:index + 1, -1],
                        aux_flat[index:index + 1] if aux_flat is not None else None,
                        local_flat[index:index + 1] if local_flat is not None else None)
                    rows.append(logits.detach().float().cpu())
        finally:
            model.train(was_training)
        return torch.cat(rows)

    before_bit_logits = _bit_logit_snapshot()
    c256_before = (compare_c256_wrapped(
        model, dev, args.reference_full256, auxiliary=dev_aux,
        local=dev_local, probe_count=min(16, args.max_dev_positions))
        if args.reference_full256 else None)
    if c256_before is not None:
        print("Original C full256 numerical parity BEFORE:", c256_before)

    fast_probe = None
    if args.rsi_fast_adapters > 0:
        fast_raw = raw_tensor
        fast_aux = aux_data
        fast_source = "TRAIN slice (no --rsi-holdout-stream)"
        if args.rsi_holdout_stream:
            holdout_raw = args.rsi_holdout_stream.read_bytes()
            if len(holdout_raw) < (4 * 64 + upgrade_context + 2):
                raise ValueError("RSI holdout stream is too short for fast probes")
            holdout_features, _ = build_train_features(holdout_raw, ranks)
            fast_raw = torch.frombuffer(bytearray(holdout_raw), dtype=torch.uint8)
            fast_aux = torch.from_numpy(
                holdout_features[PARITY_CONTEXT:].copy()).to(dtype=torch.int32)
            fast_source = str(args.rsi_holdout_stream)
        fast_probe = ProbeShards.build(
            fast_raw, fast_aux, n_shards=8, positions_per_shard=64,
            history=upgrade_context, seed=args.seed + 17)
        print("Fast-adapter RSI probe:", fast_source,
              "positions", len(fast_probe.positions), flush=True)

    window_cache = None
    if (device.type == "cuda" and args.mode == "blackwell"
            and not args.no_gpu_window_cache):
        cache_windows, cache_targets = build_window_cache(
            raw_tensor, upgrade_context, device)
        cache_aux = aux_data.to(device=device)
        window_cache = (cache_windows, cache_targets, cache_aux)
        print(
            "GPU window cache enabled:",
            f"raw_bytes={raw_tensor.numel()}",
            f"context={upgrade_context}",
            f"aux_mib={cache_aux.numel() * cache_aux.element_size() / 2**20:.1f}",
            flush=True,
        )

    last_saved_update = 0

    def save_rolling():
        nonlocal last_saved_update
        if args.smoke:
            return
        provenance["sampler"] = sampler
        save_combined(
            args.output, model, optimizer, step, provenance, events,
            training_state={
                "sampler": sampler,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": (torch.cuda.get_rng_state_all()
                             if torch.cuda.is_available() else []),
                "python_rng": random.getstate(),
                "rsi_self_builder": (rsi_builder.state_dict()
                                     if rsi_builder else None),
                "rsi_v4": (v4_controller.state_dict()
                           if v4_controller is not None else None),
            })
        last_saved_update = update_count
        print("ROLLING CHECKPOINT", args.output, "step", step,
              "update", update_count, flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    clock = time.monotonic()
    update_count = 0

    def capture_v4_branch() -> BranchSnapshot:
        action = (v4_controller.current_action.as_tuple()
                  if v4_controller is not None else Action.neutral().as_tuple())
        state = (v4_controller.capture_state()
                 if v4_controller is not None else None)
        return BranchSnapshot(
            model=state,
            optimizer=None,
            sampler=dict(sampler),
            torch_rng=torch.get_rng_state().clone(),
            python_rng=random.getstate(),
            cuda_rng=(torch.cuda.get_rng_state_all()
                      if device.type == "cuda" else []),
            step=step,
            update_count=update_count,
            training=model.training,
            action=action,
            batch_size=(v4_controller.batch_size
                        if v4_controller is not None else args.batch),
            grad_clip=(v4_controller.grad_clip
                       if v4_controller is not None else 1.0),
        )

    def restore_v4_branch(snapshot: BranchSnapshot) -> None:
        nonlocal step, update_count
        if isinstance(snapshot.model, FastStateSnapshot):
            v4_controller.restore_state(snapshot.model)
        else:
            model.load_state_dict(snapshot.model)
            optimizer.load_state_dict(snapshot.optimizer)
        sampler.clear()
        sampler.update(snapshot.sampler)
        torch.set_rng_state(snapshot.torch_rng)
        random.setstate(snapshot.python_rng)
        if device.type == "cuda" and snapshot.cuda_rng:
            torch.cuda.set_rng_state_all(snapshot.cuda_rng)
        step = snapshot.step
        update_count = snapshot.update_count
        model.train(snapshot.training)
        if v4_controller is not None:
            v4_controller.restore_action_state(
                snapshot.action, snapshot.batch_size, snapshot.grad_clip)
            model.upgrades.router_temperature = float(
                v4_controller.current_action.values()["router_temperature"])

    def train_updates(updates: int) -> list[dict]:
        nonlocal step, update_count
        reports = []
        branch_batch = (getattr(v4_controller, "active_branch_batch", None)
                        if v4_controller is not None else None)
        batch = branch_batch or (v4_controller.batch_size
                                 if v4_controller is not None else args.batch)
        clip = v4_controller.grad_clip if v4_controller is not None else 1.0
        for _ in range(updates):
            indices = coverage_indices(
                sampler["cursor"], batch, len(train_data),
                sampler["stride"], torch.device("cpu"))
            if window_cache is not None:
                gpu_indices = indices.to(device, non_blocking=True)
                positions = gpu_indices + PARITY_CONTEXT
                cache_windows, cache_targets, cache_aux = window_cache
                sample = torch.cat(
                    (cache_windows[positions], cache_targets[positions, None]),
                    dim=1).long()
                sample_aux = cache_aux[gpu_indices]
            else:
                sample = build_causal_contexts(
                    raw_tensor, indices + PARITY_CONTEXT,
                    upgrade_context).to(device)
                sample_aux = aux_data[indices].to(device=device)
            if rsi_builder is not None:
                report = rsi_builder.train_step(
                    train_one_batch, sample[:, :-1], sample[:, -1], sample_aux)
            else:
                unit_batch = (unit_labels.gather(
                    indices + PARITY_CONTEXT - 1)
                    if unit_labels is not None else None)
                report = train_one_batch(
                    model, optimizer, sample[:, :-1], sample[:, -1],
                    sample_aux, grad_clip=clip, microbatch=args.microbatch,
                    upgrade_loss_weight=args.upgrade_loss_weight,
                    boundary_loss_weight=args.boundary_loss_weight,
                    unit_targets=unit_batch,
                    unit_loss_weight=args.unit_loss_weight,
                    distill_weight=args.distill_weight,
                    distill_samples=args.distill_samples,
                    moe_loss_weight=args.moe_loss_weight,
                    unit_byte_loss_weight=args.unit_byte_loss_weight,
                    unit_byte_distill_weight=args.unit_byte_distill_weight)
            sampler["cursor"] += batch
            step += 1
            update_count += 1
            reports.append(report)
        return reports

    fast_stats: list[dict] = []

    def fast_tick() -> None:
        if fast_probe is None:
            return
        # Train and validate on the same book-disjoint holdout.  Training on
        # the main stream and validating on the holdout created a distribution
        # mismatch that made every fast adapter look worse than no-op.
        branch_windows = torch.cat(fast_probe.windows[:4], dim=0)
        branch_aux = torch.cat(fast_probe.auxiliary[:4], dim=0)
        probe_windows = torch.cat(fast_probe.windows[4:], dim=0)
        probe_aux = torch.cat(fast_probe.auxiliary[4:], dim=0)
        branch = build_fast_cache(
            model, branch_windows[:, :-1], branch_windows[:, -1], branch_aux)
        probe = build_fast_cache(
            model, probe_windows[:, :-1], probe_windows[:, -1], probe_aux)
        stats = run_fast_adapter_rsi(
            model, branch, probe,
            candidates=args.rsi_fast_adapters,
            rounds=args.rsi_fast_rounds,
            steps=args.rsi_fast_steps,
            lr=args.rsi_fast_lr,
            rank=args.rsi_fast_rank,
            min_gain=args.rsi_fast_min_gain,
            z_score=args.rsi_fast_z,
            shards=args.rsi_fast_shards,
            seed=args.seed + 1000 + len(fast_stats),
        )
        fast_stats.append(stats)
        print("FAST-RSI", json.dumps(stats), flush=True)

    def evaluate_v4_branch() -> list[float]:
        if v4_controller is None:
            return []
        return v4_controller.probe.evaluate(
            model, chunk_size=v4_controller.config.eval_chunk)

    def parallel_branch_runner(controller, base, candidates, cycle_step):
        """Run neutral + candidate branches on cloned models in CUDA streams."""
        limit = max(1, int(args.rsi_v4_parallel_replicas))
        candidate_actions = candidates[:limit]
        branch_batch = int(args.rsi_v4_branch_batch)
        local_cursor = int(sampler["cursor"])
        branch_data = []
        for _ in range(int(args.rsi_v4_horizon)):
            indices = coverage_indices(
                local_cursor, branch_batch, len(train_data),
                sampler["stride"], torch.device("cpu"))
            if window_cache is not None:
                gpu_indices = indices.to(device, non_blocking=True)
                positions = gpu_indices + PARITY_CONTEXT
                cache_windows, cache_targets, cache_aux = window_cache
                sample = torch.cat(
                    (cache_windows[positions], cache_targets[positions, None]),
                    dim=1).long()
                sample_aux = cache_aux[gpu_indices]
            else:
                sample = build_causal_contexts(
                    raw_tensor, indices + PARITY_CONTEXT,
                    upgrade_context).to(device)
                sample_aux = aux_data[indices].to(device=device)
            branch_data.append((
                sample, sample_aux,
                unit_labels.gather(indices + PARITY_CONTEXT - 1)
                if unit_labels is not None else None))
            local_cursor += branch_batch
        torch.cuda.synchronize(device)

        base_model = {name: tensor.detach().clone()
                      for name, tensor in model.state_dict().items()}
        base_optimizer = copy.deepcopy(optimizer.state_dict())
        base_lrs = list(controller.base_lrs)
        base_training = model.training
        clones = []
        for action in candidate_actions:
            compiled = model.__dict__.pop("_compiled_upgrades", None)
            try:
                clone_model = copy.deepcopy(model)
            finally:
                if compiled is not None:
                    model._compiled_upgrades = compiled
            clone_optimizer = build_v103_optimizer(
                clone_model, freeze_residual_gate=True)
            clone_model.load_state_dict(base_model)
            clone_optimizer.load_state_dict(base_optimizer)
            clone_model.upgrades.gradient_checkpointing = (
                model.upgrades.gradient_checkpointing)
            clones.append((clone_model, clone_optimizer, action))
        streams = [torch.cuda.Stream(device=device) for _ in clones]

        def run_one(index):
            clone_model, clone_optimizer, action = clones[index]
            with torch.cuda.stream(streams[index]):
                apply_action(clone_model, clone_optimizer, base_lrs, action)
                values = action.values()
                clone_model.upgrades.router_temperature = float(
                    values["router_temperature"])
                clip = float(values["grad_clip"])
                clone_model.train()
                reports = []
                for sample, sample_aux, unit_batch in branch_data:
                    reports.append(train_one_batch(
                        clone_model, clone_optimizer,
                        sample[:, :-1], sample[:, -1], sample_aux,
                        grad_clip=clip, microbatch=args.microbatch,
                        upgrade_loss_weight=args.upgrade_loss_weight,
                        boundary_loss_weight=args.boundary_loss_weight,
                        unit_targets=unit_batch,
                        unit_loss_weight=args.unit_loss_weight,
                        distill_weight=args.distill_weight,
                        distill_samples=args.distill_samples,
                        moe_loss_weight=args.moe_loss_weight,
                        unit_byte_loss_weight=args.unit_byte_loss_weight,
                    unit_byte_distill_weight=args.unit_byte_distill_weight))
                shard_bpbs = controller.probe.evaluate(
                    clone_model, chunk_size=controller.config.eval_chunk)
                clone_model.train()
                snapshot = BranchSnapshot(
                    model={name: tensor.detach().clone()
                           for name, tensor in clone_model.state_dict().items()},
                    optimizer=copy.deepcopy(clone_optimizer.state_dict()),
                    sampler=dict(base.sampler),
                    torch_rng=base.torch_rng,
                    python_rng=base.python_rng,
                    cuda_rng=base.cuda_rng,
                    step=base.step,
                    update_count=base.update_count,
                    training=base_training,
                    action=action.as_tuple(),
                    batch_size=branch_batch,
                    grad_clip=clip,
                )
                mean_bpb = (statistics.mean(shard_bpbs)
                            if shard_bpbs else float("inf"))
                return BranchResult(
                    action=action, shard_bpbs=shard_bpbs,
                    mean_bpb=mean_bpb, reports=reports, snapshot=snapshot)

        with ThreadPoolExecutor(max_workers=len(clones)) as pool:
            results = list(pool.map(run_one, range(len(clones))))
        torch.cuda.synchronize(device)
        return results

    report = {}
    while ((update_count < args.updates) if args.updates is not None
           else (time.monotonic() - clock < args.seconds)):
        if (args.rsi_fast_adapters > 0 and fast_probe is not None
                and update_count > 0
                and update_count % args.rsi_fast_cycle_updates == 0):
            fast_tick()
        if v4_controller is not None:
            remaining = ((args.updates - update_count)
                         if args.updates is not None else 10 ** 9)
            if v4_controller.should_cycle(step, remaining):
                base_step = step
                base_update = update_count
                base_sampler = dict(sampler)
                for _ in range(args.rsi_v4_cycles_per_tick):
                    v4_controller.run_cycle(
                        capture=capture_v4_branch,
                        restore=restore_v4_branch,
                        train=train_updates,
                        evaluate=evaluate_v4_branch,
                        step=step,
                        branch_runner=(parallel_branch_runner
                                       if args.rsi_v4_parallel_replicas > 0
                                       else None),
                    )
                # RSI branch training must not consume the main sampler or
                # count as kept byte updates. Keep the chosen model/optimizer
                # state, but restore the main cursor/counters.
                step = base_step
                update_count = base_update
                sampler.clear()
                sampler.update(base_sampler)
        report = train_updates(1)[0]
        if v4_controller is not None:
            elapsed = time.monotonic() - clock
            v4_controller.observe(
                report, step=step,
                updates_per_second=update_count / max(elapsed, 1e-9))
        if args.save_every and update_count - last_saved_update >= args.save_every:
            save_rolling()
        if update_count % args.progress_every == 0 or (
                args.updates is not None and update_count == args.updates):
            elapsed = time.monotonic() - clock
            rate = update_count / max(elapsed, 1e-9)
            eta_rate = rate
            if v4_controller is not None:
                compute_done = update_count + v4_controller.branch_updates
                eta_rate = compute_done / max(elapsed, 1e-9)
            if args.updates:
                percent = 100.0 * update_count / args.updates
                eta_seconds = max(0.0, (args.updates - update_count) / max(eta_rate, 1e-9))
                progress = (f"round={update_count}/{args.updates} ({percent:.1f}%) "
                            f"eta={eta_seconds:.0f}s")
            else:
                progress = (f"round={update_count} elapsed={elapsed:.0f}s/"
                            f"{args.seconds:.0f}s")
            if v4_controller is not None:
                rsi_text = (
                    f"RSI-v4 cycles={v4_controller.cycles} "
                    f"proposals={v4_controller.proposals} "
                    f"accepted={v4_controller.accepted} "
                    f"rejected={v4_controller.rejected} "
                    f"branch_updates={v4_controller.branch_updates} "
                    f"compute={update_count + v4_controller.branch_updates} "
                    f"action=[{v4_controller.action_label()}]")
            elif rsi_builder is not None:
                rsi_text = rsi_builder.progress_line()
            elif fast_stats:
                rsi_text = (
                    f"Fast-RSI ticks={len(fast_stats)} "
                    f"proposals={sum(item['proposals'] for item in fast_stats)} "
                    f"accepted={sum(item['accepted'] for item in fast_stats)}")
            else:
                rsi_text = "RSI disabled"
            aux_text = (f" upgrade_byte_bpb={report['upgrade_bpb']:.4f}"
                        if "upgrade_bpb" in report else "")
            print(
                f"PROGRESS legacy_step={step} run_update={update_count} "
                f"{progress} elapsed={elapsed:.1f}s "
                f"rate={rate:.2f} upd/s train_v103_bit_bce={report['bpb']:.6f} "
                f"bit_acc={report['bit_accuracy']:.4f}{aux_text} | {rsi_text}",
                flush=True,
            )
    # Time ONLY the byte optimizer loop here. RSI feedback and held-out DEV
    # evaluation are reported separately; otherwise updates/s is misleading.
    training_duration_seconds = time.monotonic() - clock
    if v4_controller is not None:
        meta_file = args.output.parent / "rsi_v4_replay.json"
        meta_pt = args.output.parent / "rsi_v4_meta.pt"
        v4_controller.save_meta(meta_file)
        v4_controller.save_meta(meta_pt)
        print("RSI-v4 replay/meta saved:", meta_file, meta_pt, flush=True)
    # CRITICAL: v103's original terminal demo forcibly rewards a command
    # returning 'ready' with +1 and 'broken' with -1, regardless of byte BPB.
    # That reward is NOT aligned with low-BPB training and must never fire in
    # the predictive default. Preserve the archived behavior behind explicit
    # --legacy-terminal-demo for reproduction only; RSI builder supplies the
    # real signed-BPB feedback during the training loop.
    rsi_clock = time.monotonic()
    if args.legacy_terminal_demo:
        events.extend(finish_rsi_round(model, optimizer))
    rsi_feedback_seconds = time.monotonic() - rsi_clock
    dev_after_clock = time.monotonic()
    after = byte_metrics_wrapped(model, probes, n=None,
                                 topk_n=min(args.topk_probes, len(probes)), auxiliary=probe_aux,
                                 local=probe_local)
    unit_decoder = None
    unit_gate_param = model.upgrades.hierarchy.unit_byte_gate
    saved_unit_gate = unit_gate_param.detach().clone()
    try:
        with torch.no_grad():
            unit_gate_param.zero_()
        after_no_unit = byte_metrics_wrapped(
            model, probes, n=None,
            topk_n=min(args.topk_probes, len(probes)),
            auxiliary=probe_aux, local=probe_local)
    finally:
        with torch.no_grad():
            unit_gate_param.copy_(saved_unit_gate)
    unit_decoder = {
        "gate": float(torch.tanh(saved_unit_gate)),
        "on_bpb": after["bpb"],
        "off_bpb": after_no_unit["bpb"],
        "delta_bpb": float(after_no_unit["bpb"] - after["bpb"]),
        "definition": "held-out BPB with unit->byte decoder on minus off; "
                      "positive delta means the unit heads lower BPB",
    }
    fast_adapter_delta = None
    fast_adapter = model.upgrades.fast_adapter
    saved_adapter = (
        fast_adapter.A.detach().clone(),
        fast_adapter.B.detach().clone(),
        fast_adapter.bias.detach().clone(),
    )
    try:
        with torch.no_grad():
            fast_adapter.A.zero_()
            fast_adapter.B.zero_()
            fast_adapter.bias.zero_()
        after_no_fast = byte_metrics_wrapped(
            model, probes, n=None,
            topk_n=min(args.topk_probes, len(probes)),
            auxiliary=probe_aux, local=probe_local)
    finally:
        with torch.no_grad():
            fast_adapter.A.copy_(saved_adapter[0])
            fast_adapter.B.copy_(saved_adapter[1])
            fast_adapter.bias.copy_(saved_adapter[2])
    fast_adapter_delta = {
        "on_bpb": after["bpb"],
        "off_bpb": after_no_fast["bpb"],
        "delta_bpb": float(after_no_fast["bpb"] - after["bpb"]),
        "definition": "held-out BPB with fast adapter on minus off; "
                      "positive delta means the fast RSI adapter lowers BPB",
    }
    after_bit_logits = _bit_logit_snapshot()
    bit_logit_delta = {
        "mean_abs": float((after_bit_logits - before_bit_logits).abs().mean()),
        "max_abs": float((after_bit_logits - before_bit_logits).abs().max()),
        "bit_flips": int(((after_bit_logits > 0)
                          != (before_bit_logits > 0)).sum()),
        "total_bits": int(after_bit_logits.numel()),
    }
    after_upgrade_gate_1 = None
    if v4_controller is not None:
        # The v103 core and the 100M upgrade branch are two different
        # predictors.  Report the upgrade branch at gate=1 on the same DEV
        # probes so the progress line's v103 training BCE cannot be mistaken
        # for the unified model's held-out quality.
        saved_gate = model.residual_gate.detach().clone()
        try:
            with torch.no_grad():
                model.residual_gate.fill_(10.0)  # tanh(10) ~= 1
            after_upgrade_gate_1 = byte_metrics_wrapped(
                model, probes, n=None,
                topk_n=min(args.topk_probes, len(probes)),
                auxiliary=probe_aux, local=probe_local)
        finally:
            with torch.no_grad():
                model.residual_gate.copy_(saved_gate)
    c256_after = (compare_c256_wrapped(
        model, dev, args.reference_full256, auxiliary=dev_aux,
        local=dev_local, probe_count=min(16, args.max_dev_positions))
        if args.reference_full256 else None)
    if c256_after is not None:
        print("Original C full256 numerical parity AFTER:", c256_after)
    dev_after_seconds = time.monotonic() - dev_after_clock
    total_train_rsi_dev_seconds = time.monotonic() - clock
    receipt = {"mode": args.mode, "upgrade_context": upgrade_context,
               "cpu_threads": args.cpu_threads if args.mode == "cpu" else None,
               "requested_updates": args.updates,
               "actual_byte_updates_this_round": update_count,
               "training_duration_seconds": training_duration_seconds,
               "rsi_feedback_seconds": rsi_feedback_seconds,
               "dev_after_seconds": dev_after_seconds,
               "total_train_rsi_dev_seconds": total_train_rsi_dev_seconds,
               "updates_per_second": update_count / max(training_duration_seconds, 1e-9),
               "dev_probes": len(probes),
               "topk_probes": min(args.topk_probes, len(probes)),
               "legacy_step": provenance["legacy_step"],
               "final_step": step, "updates_this_round": step - (resumed["step"] if args.resume else int(source["steps"]) if source else 0),
               "fresh_start_untrained_v103": provenance.get("fresh_start_untrained_v103", False),
               "before": before, "after": after,
               "bit_logit_delta": bit_logit_delta,
               "unit_decoder": unit_decoder,
               "fast_adapter": fast_adapter_delta,
               "after_upgrade_gate_1": after_upgrade_gate_1,
               "sampler": sampler, "feature_digest": manifest["feature_sha256"],
               "terminal_events": events,
               "terminal_reward_mode": ("legacy_fixed_command_demo" if args.legacy_terminal_demo
                                        else "predictive_dual_train_bpb"),
               "legacy_terminal_demo_disabled": not args.legacy_terminal_demo,
               "rsi_self_construction": (rsi_builder.state_dict() if rsi_builder else None),
               "rsi_mutations_verified": ((rsi_builder.proposals if rsi_builder
                                           else v4_controller.proposals
                                           if v4_controller else 0)),
               "rsi_mutations_accepted": ((rsi_builder.accepted if rsi_builder
                                           else v4_controller.accepted
                                           if v4_controller else 0)),
               "rsi_version": args.rsi_version,
               "rsi_fast": ({
                   "ticks": len(fast_stats),
                   "proposals": sum(item["proposals"] for item in fast_stats),
                   "accepted": sum(item["accepted"] for item in fast_stats),
                   "stats": fast_stats,
               } if fast_stats else None),
               "rsi_v4": ({"cycles": v4_controller.cycles,
                           "proposals": v4_controller.proposals,
                           "accepted": v4_controller.accepted,
                           "rejected": v4_controller.rejected,
                           "branch_updates": v4_controller.branch_updates,
                           "events": v4_controller.events,
                           "config": v4_controller._config_state()}
                          if v4_controller is not None else None),
               "book_disjoint_dev_digest": dev_digest,
               "new_dev_books": fresh_dev_metadata["books"] if fresh_dev_metadata else None,
               "new_dev_scored_positions": fresh_dev_metadata["scored_positions"] if fresh_dev_metadata else None,
               "new_dev_id_level_train_disjoint_proven": (fresh_dev_metadata["id_level_train_disjoint_proven"]
                                                          if fresh_dev_metadata else None),
               "original_c256_parity_before": c256_before,
               "original_c256_parity_after": c256_after,
               "test_untouched": True,
               "ablation": model.upgrades.hierarchy.ablation,
               "compile_upgrade": bool(compiled_upgrade is not None),
               "old_v103_logits_identical_at_gate_zero": proof["exact_parity"]}
    hierarchy = model.upgrades.hierarchy
    receipt["parameter_count"] = model.count_parameters()
    receipt["hierarchy"] = {
        "windows": list(hierarchy.windows),
        "gates": {
            "bottom_up": torch.tanh(
                hierarchy.bottom_up_gates).detach().cpu().tolist(),
            "top_down": torch.tanh(
                hierarchy.top_down_gates).detach().cpu().tolist(),
            "lateral": torch.tanh(
                hierarchy.lateral_gates).detach().cpu().tolist(),
        },
        "last_boundary_bce": (report.get("boundary_bce")
                              if isinstance(report, dict) else None),
        "definition": "byte->subword->word->sentence bottom-up; "
                      "sentence->word->subword->byte top-down; "
                      "word/subword lateral byte priors",
    }
    if args.generation_steps:
        from .hierarchical_generate import generate_bytes
        generation = generate_bytes(
            model, args.generation_prompt.encode("utf-8"),
            args.generation_steps,
            temperature=args.generation_temperature,
            top_k=args.generation_top_k, seed=103, device=device,
            word_vocab=(unit_labels.word_vocab if unit_labels is not None
                        else None),
            word_onset_weight=args.generation_word_onset_weight,
            fast=args.fast_generation)
        records = generation["records"]
        receipt["generation"] = {
            "prompt": args.generation_prompt,
            "text": generation["text"],
            "bytes": len(generation["bytes"]),
            "temperature": args.generation_temperature,
            "top_k": args.generation_top_k,
            "latent_word_boundaries": sum(
                1 for record in records if record["boundary_word"] > 0.5),
            "latent_sentence_boundaries": sum(
                1 for record in records if record["boundary_sentence"] > 0.5),
            "mean_word_boundary_probability": (
                sum(record["boundary_word"] for record in records)
                / max(len(records), 1)),
            "mean_sentence_boundary_probability": (
                sum(record["boundary_sentence"] for record in records)
                / max(len(records), 1)),
            "gate_snapshot": generation["hierarchy_gates"],
        }
        print("BYTE-DRIVEN GENERATION:", json.dumps(
            receipt["generation"], indent=2), flush=True)
    if args.hierarchical_eval_books:
        if args.dev_manifest is None:
            p.error("Hierarchical DEV evaluation requires --dev-manifest")
        from .hierarchical_metrics import evaluate_dev_excerpts
        receipt["hierarchical"] = evaluate_dev_excerpts(
            model,
            corpus_dir=args.hierarchical_eval_corpus_dir,
            dev_manifest=args.dev_manifest,
            bpe_merges=args.bpe_merges,
            context=upgrade_context,
            books=args.hierarchical_eval_books,
            max_positions=args.hierarchical_eval_max_positions,
            eval_batch=args.hierarchical_eval_batch,
            device=device,
        )
        print("HIERARCHICAL HELD-OUT METRICS:",
              json.dumps(receipt["hierarchical"], indent=2), flush=True)
        if unit_labels is not None and args.unit_loss_weight > 0:
            from .unit_heads import evaluate_unit_head_dev
            receipt["unit_heads_dev"] = evaluate_unit_head_dev(
                model,
                corpus_dir=args.hierarchical_eval_corpus_dir,
                dev_manifest=args.dev_manifest,
                bpe_merges=args.bpe_merges,
                context=upgrade_context,
                books=min(2, args.hierarchical_eval_books),
                max_positions=min(128, args.hierarchical_eval_max_positions
                                  or 128),
                device=device,
                vocabulary=unit_labels,
            )
            print("UNIT-HEAD HELD-OUT METRICS:",
                  json.dumps(receipt["unit_heads_dev"], indent=2), flush=True)
    provenance["sampler"] = sampler
    if args.smoke:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            receipt["gpu"] = {
                "device_name": torch.cuda.get_device_name(device),
                "total_vram_gib": round(
                    torch.cuda.get_device_properties(device).total_memory / 2**30, 3),
                "peak_allocated_gib": round(
                    torch.cuda.max_memory_allocated(device) / 2**30, 3),
                "peak_reserved_gib": round(
                    torch.cuda.max_memory_reserved(device) / 2**30, 3),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        receipt["smoke"] = True
        receipt["smoke_wall_seconds"] = time.monotonic() - started
        print("SMOKE COMPLETE:", json.dumps(receipt.get("gpu"), sort_keys=True),
              flush=True)
    else:
        save_combined(args.output, model, optimizer, step, provenance, events,
                      training_state={"sampler": sampler,
                                      "torch_rng": torch.get_rng_state(),
                                      "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                                      "python_rng": random.getstate(),
                                      "rsi_self_builder": (rsi_builder.state_dict()
                                                           if rsi_builder else None),
                                      "rsi_v4": (v4_controller.state_dict()
                                                 if v4_controller is not None else None)})
        # Exact reload guard on CPU or GPU: no numerical reinterpretation on save.
        saved = torch.load(args.output, map_location="cpu", weights_only=True)
        if saved["schema"] != COMBINED_SCHEMA or saved["steps"] != step:
            raise AssertionError("Terminal-RSI combined checkpoint reload failed")
        if saved["legacy_provenance"]["feature_manifest"] != manifest:
            raise AssertionError("Frozen v103 feature digest lost in combined checkpoint")
        with torch.no_grad():
            original_state = model.state_dict()
            for name, value in original_state.items():
                if not torch.equal(value.cpu(), saved["model"][name]):
                    raise AssertionError("Combined checkpoint changed tensor: " + name)
        if args.champion_dir:
            receipt["champion_guard"] = consider_candidate(
                args.output, receipt, args.champion_dir)
            print("RSI IMMUTABLE DEV CHAMPION:", receipt["champion_guard"], flush=True)
    if args.metrics_output:
        _write_if_unchanged_or_new(args.metrics_output,
            (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode())
    print("Unified RSI CPU/T4/Blackwell training receipt:", json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    main()
