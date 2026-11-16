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
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F
from torch import nn

from .legacy_v103_reference import (
    PARITY_CONTEXT, PARITY_SCHEMA, build_train_features, feature_manifest,
    read_bpe_ranks, heldout_windows, heldout_auxiliary, make_windows,
    byte_metrics, coverage_indices, terminal_feedback,
)
from .unified import PRESETS, device_for
from .verbatim_v103 import (
    VerbatimV103RSI, build_v103_optimizer, restore_v103_parent,
    exact_parent_parity, save_combined,
)

COMBINED_SCHEMA = "terminal-rsi-verbatim-v103-plus-upgrades-v1"


def score_byte_observations(logits: torch.Tensor, bits: torch.Tensor):
    """The original v103 BCE over all eight bits; no surrogate byte CE."""
    loss = F.binary_cross_entropy_with_logits(logits, bits)
    bpb = float((8 * loss.detach() / math.log(2)).item())
    accuracy = float(((logits.detach() > 0) == (bits > 0.5)).float().mean().item())
    return loss, bpb, accuracy


def train_one_batch(model: VerbatimV103RSI, optimizer, history: torch.Tensor,
                    target: torch.Tensor, aux: torch.Tensor):
    """One identical v103 observe AFTER optimizer step, RSI critic AFTER that."""
    if aux is None or aux.shape != (len(target), 6):
        raise ValueError("Strict v103 training requires the original six causal features")
    model.train()
    logits, bits = model(history, target, aux)
    loss, bpb, accuracy = score_byte_observations(logits, bits)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    model.original.observe_online_bits(history, target, aux)
    model.observe_verified_byte(bpb, accuracy)
    return {"bpb": bpb, "bit_accuracy": accuracy}


def verify_parent_source(source: dict, raw: bytes, ranks_path: Path,
                         feature_info: dict, dev_digest: str):
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
    if orig_dev and orig_dev != dev_digest:
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
    p.add_argument("--legacy-checkpoint", type=Path, required=True)
    p.add_argument("--train-stream", type=Path, required=True)
    p.add_argument("--bpe-merges", type=Path, required=True)
    p.add_argument("--dev-bin", type=Path, required=True,
                   help="Original book-disjoint NaNCM win_dev.bin, NOT test")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=0.0,
                   help="Start with 0 CPU parity; only then authorize training")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max-dev-positions", type=int, default=16)
    p.add_argument("--reference-full256", type=Path,
                   help="Optional original C 256-probability oracle (matched SHA)")
    args = p.parse_args(argv)
    if args.seconds < 0 or args.batch < 1 or args.max_dev_positions < 1:
        p.error("Nonnegative seconds and positive batch/dev sample required")
    if args.output.resolve() == args.legacy_checkpoint.resolve():
        p.error("Never overwrite the original trained v103 checkpoint")
    if args.output.exists():
        p.error("Destination exists; preserve candidate until audited resume support")
    device = device_for(args.mode)
    source = torch.load(args.legacy_checkpoint, map_location="cpu", weights_only=True)
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
    sampler = verify_parent_source(source, raw, args.bpe_merges, manifest, dev_digest)
    # CPU-resident features avoid filling 96GB GPU memory with 10M × 6 fields.
    train_data, _ = make_windows(raw, torch.device("cpu"), train_all=True,
                                 history=PARITY_CONTEXT)
    aux_data = torch.from_numpy(train_features[PARITY_CONTEXT:].copy()).to(dtype=torch.int32)
    if len(train_data) != len(aux_data):
        raise AssertionError("Original feature-target offset mismatch")
    dev_aux = heldout_auxiliary(str(args.dev_bin), device, ranks, memory)
    model = VerbatimV103RSI(args.mode).to(device)
    optimizer = build_v103_optimizer(model)
    provenance = restore_v103_parent(args.legacy_checkpoint, model, optimizer)
    # The old CPU↔GPU path must not implicitly replace the inherited model;
    # this also verifies bit-for-bit equality with actual BPE/match feature IDs.
    actual = train_data[:args.batch].to(device=device, dtype=torch.long)
    actual_aux = aux_data[:args.batch].to(device=device)
    proof = exact_parent_parity(model, actual[:, :-1], actual[:, -1], actual_aux)
    before = byte_metrics(model, dev, n=args.max_dev_positions,
                          topk_n=args.max_dev_positions, auxiliary=dev_aux)
    print("VERBATIM CPU-compatible v103 parent parity:", json.dumps(proof))
    print("DEV before:", before)
    provenance.update({"source_sha256": hashlib.sha256(raw).hexdigest(),
                       "feature_manifest": manifest, "dev_sha256": dev_digest,
                       "original_two_adam_groups": [g["lr"] for g in optimizer.param_groups[:2]]})
    step = int(source["steps"])
    events = []
    clock = time.monotonic()
    while time.monotonic() - clock < args.seconds:
        indices = coverage_indices(sampler["cursor"], args.batch,
                                   len(train_data), sampler["stride"],
                                   torch.device("cpu"))
        sample = train_data[indices].to(device=device, dtype=torch.long)
        sample_aux = aux_data[indices].to(device=device)
        report = train_one_batch(model, optimizer, sample[:, :-1],
                                 sample[:, -1], sample_aux)
        sampler["cursor"] += args.batch
        step += 1
        if step % 50 == 0:
            print("RSI step", step, "train", report, flush=True)
    if args.seconds:
        # Preserve original Terminal-RSI's REAL verified actor-critic loop.
        events.extend(terminal_feedback(model.original, optimizer))
    after = byte_metrics(model, dev, n=args.max_dev_positions,
                         topk_n=args.max_dev_positions, auxiliary=dev_aux)
    receipt = {"mode": args.mode, "legacy_step": provenance["legacy_step"],
               "final_step": step, "updates": step - int(source["steps"]),
               "before": before, "after": after,
               "sampler": sampler, "feature_digest": manifest["feature_sha256"],
               "terminal_events": events,
               "book_disjoint_dev_digest": dev_digest,
               "test_untouched": True,
               "old_v103_logits_identical_at_gate_zero": proof["exact_parity"]}
    provenance["sampler"] = sampler
    save_combined(args.output, model, optimizer, step, provenance, events)
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
    print("Unified RSI CPU/T4/Blackwell training receipt:", json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    main()
