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
from .unit_heads import WORD_VOCAB_SIZE, word_prefix_byte_log_probs
from .word_bridge import (
    WORD4_BUCKETS, WORD4_TOPK, WordTokenBridge, build_word4_table, word4_bucket,
)

# Trust region for the word-prefix prior.  The prior is a classifier-derived
# distribution, so a free gate lets it dominate the parent before the word
# head is accurate.  0.05 caps its logit contribution at ~0.4.
WORD_PRIOR_MAX_SCALE = 0.05
from .unified import PRESETS, TerminalRSI, device_for


class VerbatimV103RSI(nn.Module):
    def __init__(self, mode="cpu", word_vocab_size=WORD_VOCAB_SIZE):
        super().__init__()
        if mode not in PRESETS:
            raise ValueError("Unknown CPU/T4/Blackwell mode")
        if int(word_vocab_size) < 2:
            raise ValueError("word_vocab_size must be at least 2")
        self.cfg = PRESETS[mode]
        self.word_vocab_size = int(word_vocab_size)
        self.original = V103ParityGPUFullSystem()
        self.upgrades = TerminalRSI(
            self.cfg, word_vocab_size=self.word_vocab_size)
        self.residual_gate = nn.Parameter(torch.zeros(()))
        self.word_prior_gate = nn.Parameter(torch.zeros(()))
        self.word_bridge = WordTokenBridge(self.word_vocab_size)
        self.word_bridge_scale = nn.Parameter(torch.tensor(0.1))
        self.word_ngram_scale = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("word_bytes", torch.zeros(1, 1, dtype=torch.long),
                             persistent=False)
        self.register_buffer("word_lengths", torch.zeros(1, dtype=torch.long),
                             persistent=False)
        self.register_buffer("word_bigram_log_probs", torch.zeros(0),
                             persistent=False)
        self.register_buffer("word_unigram_log_probs", torch.zeros(0),
                             persistent=False)
        self.register_buffer("word4_next", torch.zeros(0), persistent=False)
        self.register_buffer("word4_logp", torch.zeros(0), persistent=False)
        self._word_vocab: dict[bytes, int] | None = None

    @torch.no_grad()
    def set_word_vocab(self, vocabulary: dict[bytes, int], maxlen: int = 24):
        # Match the actual head width, not a module constant.
        size = int(self.upgrades.hierarchy.unit_heads.word.out_features)
        if max(vocabulary.values(), default=0) >= size:
            raise ValueError("word vocabulary exceeds the word head width")
        table = torch.zeros((size, maxlen), dtype=torch.long)
        lengths = torch.zeros(size, dtype=torch.long)
        for word, index in vocabulary.items():
            piece = torch.tensor(list(word[:maxlen]), dtype=torch.long)
            table[index, :len(piece)] = piece
            lengths[index] = min(len(word), maxlen)
        self.word_bytes = table
        self.word_lengths = lengths
        self._word_vocab = dict(vocabulary)

    @torch.no_grad()
    def set_word_ngram(self, raw: bytes, alpha: float = 0.1):
        """Build a 4k-vocab word bigram prior from the training stream."""
        if not self._word_vocab:
            return
        from .unit_heads import _word_tokens
        ids = [self._word_vocab.get(word, 0)
               for _start, _end, word in _word_tokens(raw)]
        # Keep UNK (id 0) in the stream.  Dropping it merged transitions across
        # OOV words and made the prior unable to predict UNK, even though UNK is
        # a valid target class in the word-head evaluation.
        size = int(self.upgrades.hierarchy.unit_heads.word.out_features)
        counts = torch.zeros((size, size), dtype=torch.float32)
        if len(ids) > 1:
            previous = torch.tensor(ids[:-1], dtype=torch.long)
            following = torch.tensor(ids[1:], dtype=torch.long)
            counts.index_put_((previous, following),
                              torch.ones(len(previous)), accumulate=True)
        row = counts.sum(1, keepdim=True)
        logp = torch.log((counts + alpha) / (row + alpha * size))
        unigram = counts.sum(0)
        logu = torch.log((unigram + alpha)
                         / (unigram.sum() + alpha * size))
        device = next(self.parameters()).device
        self.word_bigram_log_probs = logp.to(device)
        self.word_unigram_log_probs = logu.to(device)
        next_ids, ngram_logp = build_word4_table(ids)
        self.word4_next = next_ids.to(device)
        self.word4_logp = ngram_logp.to(device)
        self.word_ngram_counts = int(len(ids))

    def word_ngram_prior(self, prev1: torch.Tensor, prev2: torch.Tensor,
                          prev3: torch.Tensor) -> torch.Tensor:
        """Bigram backoff + hashed 4-gram top-k prior [batch,vocab]."""
        prev1 = prev1.long()
        prior = None
        if self.word_bigram_log_probs.numel():
            prior = self.word_bigram_log_probs[
                prev1.clamp(0, self.word_bigram_log_probs.shape[0] - 1)]
        if self.word4_next.numel():
            buckets = word4_bucket(prev1, prev2, prev3, self.word4_next.shape[0])
            ids = self.word4_next[buckets].long()
            values = self.word4_logp[buckets].float()
            if prior is None:
                prior = torch.zeros(
                    (prev1.shape[0], self.word_vocab_size), device=prev1.device)
            prior = prior.clone()
            prior.scatter_add_(1, ids.clamp(0, prior.shape[1] - 1), values)
        if prior is None:
            prior = torch.zeros((prev1.shape[0], self.word_vocab_size),
                                device=prev1.device)
        return prior

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
        gate_features = upgrade_out.get("gate_features")
        if gate_features is not None:
            # Per-token gate: residual_gate is the global RSI knob, while the
            # head learns which positions trust the 100M branch.
            token_gate = torch.tanh(
                self.residual_gate + self.upgrades.upgrade_gate(
                    gate_features).squeeze(-1))
            gate = (token_gate * trust.to(old_logits.dtype))[:, None]
        else:
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
            prior = torch.nan_to_num(prior, nan=-30.0, neginf=-30.0,
                                     posinf=0.0)
            prior_bits = self._byte_to_conditional_bit_logits(prior, target)
            word_scale = (WORD_PRIOR_MAX_SCALE
                          * float(getattr(self, "word_prior_byte_scale", 1.0))
                          * torch.tanh(self.word_prior_gate))
            final_logits = final_logits + word_scale.to(
                final_logits.dtype) * prior_bits.to(final_logits.dtype)
        if use_fast_adapter:
            adapter_input = upgrade_out.get("adapter_input")
            if adapter_input is not None:
                adapter_logits = torch.nan_to_num(
                    self.upgrades.fast_adapter(adapter_input),
                    nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
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
                   "upgrades.upgrade_gate.", "upgrades.fast_adapter.",
                   "word_bridge.", "word_bridge_scale.",
                   "word_ngram_scale.", "word_prior_gate.",
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
                         freeze_residual_gate: bool = False,
                         freeze_fast_adapter: bool = False):
    """Preserve the ORIGINAL two AdamW learning-rate groups, plus upgrades.

    The historical v103 notebook used 2e-3 for inherited GPUFullSystem
    parameters and 5e-4 for new v103 specialists. New upgrade tensors live
    in a third group; they never alter the old group's optimizer histories.

    With ``freeze_residual_gate=True`` the residual mixing gate is owned by
    the RSI controller, not by direct gradient descent.  This is required for
    the v4 path: the 100M upgrade branch is random at gate 0, so a direct gate
    gradient would inject random logits before RSI has verified any branch.
    """
    model.upgrades.fast_adapter.requires_grad_(not freeze_fast_adapter)
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
                or name == "hierarchy.unit_byte_gate"
                or name.startswith("fast_adapter.")):
            side.append(parameter)
        else:
            added.append(parameter)
    # The word-prefix prior and word-token bridge are side heads.
    side.append(model.word_prior_gate)
    side.append(model.word_bridge_scale)
    side.append(model.word_ngram_scale)
    side.extend(model.word_bridge.parameters())
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
        {"params": side, "lr": 5e-4, "name": "unit_side_heads"},
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
              "word_vocab_size": int(getattr(
                  model, "word_vocab_size", WORD_VOCAB_SIZE)),
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


def load_optimizer_allow_extra_params(optimizer, state_dict, allowed_extra):
    """Load an older optimizer state after new parameters joined a group.

    Legacy four-group checkpoints predate the trainable fast adapter.  Their
    saved side group is shorter than the current one.  Adam creates moments
    lazily, so the new parameters need no saved state: temporarily shrink the
    group to the saved length, load every inherited history, then append the
    new parameters.  Only the explicitly allowed parameters may be appended.
    """
    try:
        optimizer.load_state_dict(state_dict)
        return False
    except ValueError as error:
        failure = error
    current = optimizer.param_groups
    saved = state_dict.get("param_groups") or []
    if len(current) != len(saved):
        raise ValueError("Optimizer group count changed unexpectedly") from failure
    differences = []
    for index, (group, saved_group) in enumerate(zip(current, saved)):
        extra = len(group["params"]) - len(saved_group["params"])
        if extra:
            differences.append((index, extra))
    if len(differences) != 1 or differences[0][1] < 1:
        raise ValueError("Optimizer groups changed beyond the fast adapter")
    index, extra = differences[0]
    group = current[index]
    full = list(group["params"])
    missing = full[-extra:]
    allowed = {id(parameter) for parameter in allowed_extra}
    if any(id(parameter) not in allowed for parameter in missing):
        raise ValueError("Unexplained optimizer parameters cannot be migrated")
    group["params"] = full[:-extra]
    optimizer.load_state_dict(state_dict)
    optimizer.param_groups[index]["params"].extend(missing)
    return True


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
    saved_vocab = int(checkpoint.get("word_vocab_size", WORD_VOCAB_SIZE))
    if saved_vocab != int(getattr(model, "word_vocab_size", WORD_VOCAB_SIZE)):
        raise ValueError("Word vocabulary size changed across resume")
    prior = checkpoint["legacy_provenance"]
    if (prior.get("source_sha256") != source_sha256 or
            prior.get("feature_manifest") != feature_manifest or
            prior.get("dev_sha256") != dev_sha256):
        raise ValueError("Original train/BPE/DEV identity changed across RSI resume")
    model.load_state_dict(checkpoint["model"], strict=True)
    load_optimizer_allow_extra_params(
        optimizer, checkpoint["optimizer"],
        model.upgrades.fast_adapter.parameters())
    if len(optimizer.param_groups) < 3:
        raise ValueError("Optimizer group partition lost")
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
