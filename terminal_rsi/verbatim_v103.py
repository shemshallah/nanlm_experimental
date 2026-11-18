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
from .unit_heads import word_prefix_byte_log_probs
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
        self.word_prior_gate = nn.Parameter(torch.zeros(()))
        self.register_buffer("word_bytes", torch.zeros(1, 1, dtype=torch.long),
                             persistent=False)
        self.register_buffer("word_lengths", torch.zeros(1, dtype=torch.long),
                             persistent=False)

    @torch.no_grad()
    def set_word_vocab(self, vocabulary: dict[bytes, int], maxlen: int = 24):
        size = max(vocabulary.values(), default=0) + 1
        table = torch.zeros((size, maxlen), dtype=torch.long)
        lengths = torch.zeros(size, dtype=torch.long)
        for word, index in vocabulary.items():
            piece = torch.tensor(list(word[:maxlen]), dtype=torch.long)
            table[index, :len(piece)] = piece
            lengths[index] = min(len(word), maxlen)
        self.word_bytes = table
        self.word_lengths = lengths

    @staticmethod
    def _byte_to_conditional_bit_logits(byte_logits: torch.Tensor,
                                        target: torch.Tensor):
        """Exact differentiable 256-way factorization into causal bit logits.

        Hierarchical pairwise log-sum-exp: at each depth only the target-prefix
        node is gathered.  This replaces eight full 256-way masked reductions
        with 255 pair reductions, which matters at Blackwell batch 8192.
        """
        if byte_logits.ndim != 2 or byte_logits.shape[-1] != 256:
            raise ValueError("Byte logits must be [batch,256]")
        target = target.long().to(byte_logits.device)
        values = F.log_softmax(byte_logits.float(), dim=-1)
        out: list[torch.Tensor] = [None] * 8  # type: ignore[list-item]
        for bit in range(7, -1, -1):
            groups = values.reshape(values.shape[0], -1, 2)
            diff = groups[..., 1] - groups[..., 0]
            prefix = (target >> (8 - bit)).clamp(0, diff.shape[1] - 1)
            out[bit] = diff.gather(1, prefix.view(-1, 1)).squeeze(1)
            values = torch.logsumexp(groups, dim=-1)
        return torch.stack(out, dim=1)

    def forward(self, previous: torch.Tensor, target: torch.Tensor,
                aux: torch.Tensor | None = None,
                local: torch.Tensor | None = None,
                return_aux: bool = False,
                return_hierarchy: bool = False,
                use_fast_adapter: bool = True):
        if previous.ndim != 2 or previous.shape[1] < 32:
            raise ValueError("Exact v103 needs >=32 observed bytes; legacy uses LAST 32")
        # The archived v103 model always receives exactly its original 32
        # bytes; additive attention may see up to preset.context causal bytes.
        old_logits, bits = self.original(previous[:, -32:], target, aux, local)
        # When untrained, this extra model receives context but cannot perturb
        # the inherited v103 prediction because residual_gate is exact zero.
        # Blackwell: BF16 tensor cores and native fused causal SDPA are used
        # ONLY for the additive specialist. Keep archived v103 logits, RSI
        # actor/critic and inherited AdamW groups in their original FP32
        # execution path. This changes no checkpoint keys or parameters.
        upgrade_module = getattr(self, "_compiled_upgrades", None)
        if upgrade_module is None:
            upgrade_module = self.upgrades
        if self.cfg.name == "blackwell" and previous.is_cuda:
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("Blackwell upgrade requires BF16-capable CUDA")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                upgrade_out = upgrade_module(previous.long())
                current = upgrade_out["logits"][0, :, -1, :]
        else:
            upgrade_out = upgrade_module(previous.long())
            current = upgrade_out["logits"][0, :, -1, :]
        proposed = self._byte_to_conditional_bit_logits(current, target)
        # Distillation views: the parent's target-conditioned bit logits are
        # the teacher; the 100M branch's bit path is the student.  No teacher
        # gradient enters the parent.
        upgrade_out["teacher_bits"] = old_logits.detach()
        upgrade_out["student_bits"] = proposed
        # Recursive RSI is the mixing-policy controller in ALL three modes.
        # Its persistent state comes from prior byte errors and verified
        # terminal outcomes only; no future target enters this trust signal.
        # With residual_gate=0 the inherited v103 logits remain EXACT, while
        # a learned nonzero gate adapts continuously to verified RSI feedback.
        trust = 1.0 + 0.25 * torch.tanh(
            self.original.feedback_state.detach().float().mean())
        gate = torch.tanh(self.residual_gate) * trust.to(old_logits.dtype)
        final_logits = old_logits + gate * (proposed - old_logits)
        # Milestone M1: the unit heads' byte decoder is a separate predictor.
        # It lives outside the residual gate, so it can lower held-out BPB even
        # while the 100M upgrade branch is still gated off.
        unit_byte = upgrade_out.get("unit_byte_logits")
        if unit_byte is not None:
            unit_bits = self._byte_to_conditional_bit_logits(unit_byte, target)
            final_logits = final_logits + unit_bits.to(final_logits.dtype)
        word_logits = upgrade_out.get("unit_logits", {}).get("current_word")
        if word_logits is not None and self.word_bytes.numel() > 1:
            prior = word_prefix_byte_log_probs(
                word_logits, previous, self.word_bytes, self.word_lengths,
                top_k=8)
            prior_bits = self._byte_to_conditional_bit_logits(prior, target)
            final_logits = final_logits + torch.tanh(
                self.word_prior_gate).to(final_logits.dtype) * prior_bits.to(
                    final_logits.dtype)
        if use_fast_adapter:
            adapter_input = upgrade_out.get("adapter_input")
            if adapter_input is not None:
                adapter_logits = self.upgrades.fast_adapter(adapter_input)
                adapter_bits = self._byte_to_conditional_bit_logits(
                    adapter_logits, target)
                final_logits = final_logits + adapter_bits.to(final_logits.dtype)
        base_last = upgrade_out.get("last_base_logits")
        refined_last = upgrade_out.get("last_refined_logits")
        if base_last is not None and refined_last is not None:
            base_bits = self._byte_to_conditional_bit_logits(base_last, target)
            refined_bits = self._byte_to_conditional_bit_logits(
                refined_last, target)
            feedback_gate = torch.tanh(
                self.upgrades.symbiosis_output_gate).to(final_logits.dtype)
            final_logits = final_logits + feedback_gate * (
                refined_bits - base_bits).to(final_logits.dtype)
        if return_hierarchy:
            return final_logits, bits, current, upgrade_out
        if return_aux:
            return final_logits, bits, current
        return final_logits, bits

    @torch.no_grad()
    def observe_verified_byte(self, bpb: float, bit_accuracy: float):
        """Reuses the original v103 RSI critic; AFTER scoring only."""
        self.original.observe_byte_loss(bpb, bit_accuracy)

    def load_state_dict(self, state_dict, strict: bool = True,
                        assign: bool = False):
        from .unified import TerminalRSI
        migrated = TerminalRSI.migrate_state_dict(dict(state_dict))
        missing, unexpected = super().load_state_dict(
            migrated, strict=False, assign=assign)
        allowed = ("upgrades.fast_weight.", "upgrades.depth_head.",
                   "upgrades.fast_adapter.", "word_prior_gate.",
                   "upgrades.hierarchy.pair_memory.",
                   "upgrades.hierarchy.unit_heads.")
        bad_missing = [
            key for key in missing
            if not key.startswith(allowed) and ".moe." not in key]
        if strict and (unexpected or bad_missing):
            raise RuntimeError(
                f"Checkpoint mismatch: missing={bad_missing[:4]} "
                f"unexpected={unexpected[:4]}")
        return torch.nn.modules.module._IncompatibleKeys(missing, unexpected)

    @torch.no_grad()
    def observe_verified_terminal(self, success: bool):
        self.original.observe_terminal(success)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


def build_v103_optimizer(model: VerbatimV103RSI, *,
                         freeze_residual_gate: bool = False):
    """Preserve the ORIGINAL two AdamW learning-rate groups, plus upgrades.

    The historical v103 notebook used 2e-3 for inherited GPUFullSystem
    parameters and 5e-4 for new v103 specialists. New upgrade tensors live
    in a third group; they never alter the old group's optimizer histories.

    With ``freeze_residual_gate=True`` the residual mixing gate is owned by
    the RSI controller, not by direct gradient descent.  This is required for
    the v4 path: the 100M upgrade branch is random at gate 0, so a direct gate
    gradient would inject random logits before RSI has verified any branch.
    """
    parent = list(model.original.named_parameters())
    base = [p for name, p in parent if not name.startswith("v103_")]
    specialists = [p for name, p in parent if name.startswith("v103_")]
    side = []
    added = []
    for name, parameter in model.upgrades.named_parameters():
        if not parameter.requires_grad:
            continue
        if (name.startswith("hierarchy.unit_heads.")
                or name.startswith("hierarchy.unit_to_byte.")
                or name == "hierarchy.unit_byte_gate"):
            side.append(parameter)
        else:
            added.append(parameter)
    if freeze_residual_gate:
        model.residual_gate.requires_grad_(False)
    else:
        added.insert(0, model.residual_gate)
    if not base or not specialists or not added:
        raise RuntimeError("Historical two-way v103 parameter partition unavailable")
    return torch.optim.AdamW([
        {"params": base, "lr": 2e-3, "name": "original_base"},
        {"params": specialists, "lr": 5e-4, "name": "original_v103"},
        {"params": added, "lr": model.cfg.lr, "name": "recursive_upgrades"},
        {"params": side, "lr": 1e-3, "name": "unit_side_heads"},
    ])


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
    old_groups = source["optimizer"]["param_groups"]
    old_ids = [pid for group in old_groups for pid in group["params"]]
    declared_names = [name for name, _ in model.original.named_parameters()]
    # PyTorch enumerates direct v103_emit before registered modules; the
    # ORIGINAL two-group notebook filters all parameters by name prefix.
    # Flat optimizer ID ↔ named_parameter zip silently misplaces Adam moments.
    base_names = [n for n in declared_names if not n.startswith("v103_")]
    v103_names = [n for n in declared_names if n.startswith("v103_")]
    if len(old_groups) == 2:
        if [len(g["params"]) for g in old_groups] != [len(base_names), len(v103_names)]:
            raise ValueError("Original two-group sizes do not match verbatim v103 source")
        old_names = base_names + v103_names
    else:
        old_names = declared_names
    if len(old_ids) != len(old_names) or len(set(old_ids)) != len(old_ids):
        raise ValueError("Original v103 model and grouped Adam parameter order mismatch")
    # The original checkpoint may have one group (v1) or the ACTUAL v103
    # notebook's two groups. Never overwrite separate v103 learning rates.
    current = optimizer.state_dict()
    new_ids = [pid for group in current["param_groups"] for pid in group["params"]]
    names = [name for name, param in model.named_parameters()
             if param.requires_grad]
    # model.named_parameters() puts residual_gate first; optimizer factory
    # owns the tensors in semantic order, so derive IDs from optimizer itself.
    live_to_name = {id(param): name for name, param in model.named_parameters()}
    ordered_names = [live_to_name[id(param)] for group in optimizer.param_groups
                     for param in group["params"]]
    if len(new_ids) != len(ordered_names) or set(ordered_names) != set(names):
        raise ValueError("Unified optimizer must own every tensor exactly once")
    name_to_id = dict(zip(ordered_names, new_ids))
    original_adam = source["optimizer"]["state"]
    if set(original_adam) - set(old_ids):
        raise ValueError("Optimizer has states for nonexistent historical parameters")
    current["state"] = {
        name_to_id["original." + name]: original_adam[old_id]
        for name, old_id in zip(old_names, old_ids)
        if old_id in original_adam
    }
    if len(current["state"]) != len(original_adam):
        raise ValueError("Failed to map all original Adam moments by parameter name")
    if len(old_groups) == 2:
        if len(optimizer.param_groups) not in (3, 4):
            raise ValueError("Historical v103 needs TWO original Adam groups + upgrade/side groups; use build_v103_optimizer")
        for idx, group in enumerate(old_groups):
            target = current["param_groups"][idx]
            for key, value in group.items():
                if key != "params":
                    target[key] = value
    elif len(old_groups) == 1:
        # This is permitted for synthetic v103 or early one-group checkpoint.
        # Preserve its LR for both inherited groups when using the factory.
        if len(optimizer.param_groups) == 3:
            for idx in (0, 1):
                for key, value in old_groups[0].items():
                    if key != "params":
                        current["param_groups"][idx][key] = value
        elif len(optimizer.param_groups) == 1:
            for key, value in old_groups[0].items():
                if key != "params":
                    current["param_groups"][0][key] = value
        else:
            raise ValueError("Unexpected synthetic optimizer grouping")
    else:
        raise ValueError("Unsupported historical group count; do not discard any Adam history")
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
                        target: torch.Tensor, aux: torch.Tensor | None = None,
                        local: torch.Tensor | None = None):
    """Read-only exact same-device parity, including saved online buffers."""
    model.eval()
    with torch.no_grad():
        baseline, bits_a = model.original(previous[:, -32:], target, aux, local)
        migrated, bits_b = model(previous, target, aux, local)
    if not torch.equal(bits_a, bits_b) or not torch.equal(baseline, migrated):
        raise AssertionError("VERBATIM V103 PARITY FAILED. Refuse further training.")
    return {"exact_parity": True, "samples": int(len(target)),
            "max_abs_logit_error": float((baseline - migrated).abs().max().item())}


@torch.no_grad()
def exhaustive_parent_byte_parity(model: VerbatimV103RSI, observed: torch.Tensor,
                                  aux: torch.Tensor | None = None,
                                  local: torch.Tensor | None = None):
    """All 256 byte outcomes; exact old-vs-wrapper probabilities and mass.

    Unlike a four-target random smoke check, this catches target-prefix tree
    and old v103 causality failures for each of the eight bit predictions.
    No user data or optimizer state is changed.
    """
    if observed.ndim != 2 or observed.shape[1] < 32:
        raise ValueError("Need historical causal byte contexts")
    n = len(observed)
    candidates = torch.arange(256, device=observed.device).repeat(n)
    context = observed.repeat_interleave(256, dim=0)
    expanded_aux = aux.repeat_interleave(256, dim=0) if aux is not None else None
    expanded_local = local.repeat_interleave(256, dim=0) if local is not None else None
    result = exact_parent_parity(model, context, candidates, expanded_aux,
                                 expanded_local)
    logits, bits = model(context, candidates, expanded_aux, expanded_local)
    logp = -F.binary_cross_entropy_with_logits(logits, bits,
                                                reduction="none").sum(-1)
    probs = logp.reshape(n, 256).exp()
    if not torch.isfinite(probs).all():
        raise AssertionError("A causal byte probability is not finite")
    mass_error = (probs.sum(-1) - 1).abs().max().item()
    if mass_error > 2e-3:
        raise AssertionError("Inherited 256-way v103 probability mass not normalized")
    result.update({"candidate_bytes_per_history": 256,
                   "maximum_byte_mass_error": float(mass_error),
                   "context_histories": n,
                   "all_256_old_wrapper_probabilities_exact": True})
    return result


def save_combined(path: Path, model: VerbatimV103RSI, optimizer, step: int,
                  provenance: dict, feedback_events=None,
                  training_state: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"schema": "terminal-rsi-verbatim-v103-plus-upgrades-v1",
              "mode": model.cfg.name, "config": vars(model.cfg),
              "model": model.state_dict(), "optimizer": optimizer.state_dict(),
              "steps": step, "legacy_provenance": provenance,
              "feedback_events": list(feedback_events or []),
              "training_state": training_state or {
                  "torch_rng": torch.get_rng_state(),
                  "cuda_rng": (torch.cuda.get_rng_state_all()
                               if torch.cuda.is_available() else []),
                  "python_rng": __import__("random").getstate(),
                  "sampler": provenance.get("sampler")}}
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".incomplete", delete=False) as f:
        tmp = Path(f.name)
    try:
        torch.save(bundle, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def resume_combined(path: Path, model: VerbatimV103RSI, optimizer,
                    *, source_sha256: str, feature_manifest: dict,
                    dev_sha256: str):
    """One exact-mode checkpoint with all old/new optimizer, RNG and RSI state.

    Cross-preset growth needs separate audited migration, never an implicit
    load_state_dict(strict=False) that loses specialist histories.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != "terminal-rsi-verbatim-v103-plus-upgrades-v1":
        raise ValueError("Not a complete unified Terminal-RSI checkpoint")
    if checkpoint.get("config") != vars(model.cfg):
        raise ValueError("Mode/architecture changed: require explicit capacity migration")
    prior = checkpoint["legacy_provenance"]
    if (prior.get("source_sha256") != source_sha256 or
            prior.get("feature_manifest") != feature_manifest or
            prior.get("dev_sha256") != dev_sha256):
        raise ValueError("Original train/BPE/DEV identity changed across RSI resume")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if len(optimizer.param_groups) != 3:
        raise ValueError("Inherited two Adam groups and additive upgrade group lost")
    state = checkpoint.get("training_state")
    if not state or "sampler" not in state or "torch_rng" not in state:
        raise ValueError("Unified RSI checkpoint lacks restartable RNG/sampler state")
    return {"step": checkpoint["steps"], "sampler": dict(state["sampler"]),
            "events": list(checkpoint.get("feedback_events", [])),
            "rng": state, "legacy_provenance": prior}


def main():
    parser = argparse.ArgumentParser(description="Read-only v103 checkpoint parity before any CPU/GPU training")
    parser.add_argument("--mode", choices=PRESETS, default="cpu")
    parser.add_argument("--legacy-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    dev = device_for(args.mode)
    model = VerbatimV103RSI(args.mode).to(dev)
    opt = build_v103_optimizer(model)
    audit = restore_v103_parent(args.legacy_checkpoint, model, opt)
    # Real checkpoint state, deterministic observed byte context; no target-future input.
    g = torch.Generator().manual_seed(103)
    p = torch.randint(0, 256, (4, 32), generator=g).to(dev)
    t = torch.randint(0, 256, (4,), generator=g).to(dev)
    report = exact_parent_parity(model, p, t)
    byte_report = exhaustive_parent_byte_parity(
        model, p, model.original._default_aux(p).to(torch.int32))
    print(json.dumps({"checkpoint": audit, "exhaustive_256_byte_parity": byte_report,
                      "inherited_model_parameter_count":
                      model.original.count_parameters() if hasattr(model.original, "count_parameters")
                      else sum(v.numel() for v in model.original.parameters()),
                      "unified_parameter_count": model.count_parameters(),
                      "parity": report}, indent=2))


if __name__ == "__main__":
    main()
