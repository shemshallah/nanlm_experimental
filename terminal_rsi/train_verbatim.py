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

from .build_fresh_dev import validate_fresh_dev, _write_if_unchanged_or_new
from .legacy_v103_reference import (
    PARITY_CONTEXT, PARITY_SCHEMA, build_train_features, feature_manifest,
    read_bpe_ranks, heldout_windows, heldout_auxiliary, make_windows,
    coverage_indices, co_prime_stride, terminal_feedback,
)
from .unified import PRESETS, device_for
from .rsi_self_builder import RSISelfBuilder
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
    model.original.observe_online_bits(history[:, -32:], target, aux)
    model.observe_verified_byte(bpb, accuracy)
    return {"bpb": bpb, "bit_accuracy": accuracy}


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
    p.add_argument("--seconds", type=float, default=0.0,
                   help="Optional wall-clock budget; cannot combine with --updates")
    p.add_argument("--updates", type=int,
                   help="Exact number of optimizer updates, e.g. 1000 on CPU; independent of speed")
    p.add_argument("--cpu-threads", type=int, choices=(1, 2, 4), default=2,
                   help="CPU worker math threads: 1, 2 or 4; for GitHub/Molab")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--rsi-cycle-updates", type=int, default=50,
                   help="RSI self-construction proposal/verification every N byte steps; 0 disables for ablation")
    p.add_argument("--rsi-probe-positions", type=int, default=8,
                   help="Causal internal TRAIN probes for RSI verification; DEV/TEST never used")
    p.add_argument("--max-dev-positions", type=int, default=16)
    p.add_argument("--topk-probes", type=int, default=8,
                   help="Bound exact 256-way byte top1/top5 probes to avoid 256x CPU evaluations")
    p.add_argument("--upgrade-context", type=int, default=None,
                   help="Observed causal bytes for additive model, up to preset context")
    p.add_argument("--reference-full256", type=Path,
                   help="Optional original C 256-probability oracle (matched SHA)")
    args = p.parse_args(argv)
    if (args.seconds < 0 or args.batch < 1 or args.max_dev_positions < 1
            or args.topk_probes < 1 or (args.updates is not None and args.updates < 0)):
        p.error("Nonnegative seconds/updates and positive batch/dev/topk sample required")
    if args.rsi_cycle_updates < 0 or args.rsi_probe_positions < 1:
        p.error("RSI cycle must be >=0 and internal probe count positive")
    if args.updates is not None and args.seconds > 0:
        p.error("Choose --updates OR --seconds, not both")
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
    sampler = (verify_parent_source(
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
    dev_aux = heldout_auxiliary(str(args.dev_bin), device, ranks, memory)
    raw_tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    model = VerbatimV103RSI(args.mode).to(device)
    optimizer = build_v103_optimizer(model)
    provenance = (restore_v103_parent(args.legacy_checkpoint, model, optimizer)
                  if source is not None else
                  {"legacy_step": 0, "legacy_checkpoint_sha256": None,
                   "fresh_start_untrained_v103": True,
                   "original_optimizer_histories_inherited": False})
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
        meta_count = min(args.rsi_probe_positions, len(train_data),
                         2 if device.type == "cuda" else args.rsi_probe_positions)
        meta_indices = torch.linspace(0, len(train_data) - 1, meta_count).long()
        meta_window = build_causal_contexts(
            raw_tensor, meta_indices + PARITY_CONTEXT, upgrade_context).to(device)
        meta_aux = aux_data[meta_indices].to(device)
        rsi_builder = RSISelfBuilder(
            model, optimizer, meta_window[:, :-1], meta_window[:, -1],
            meta_aux, interval=args.rsi_cycle_updates)
        if args.resume and resumed["rng"].get("rsi_self_builder"):
            rsi_builder.load_state_dict(resumed["rng"]["rsi_self_builder"])
        elif args.resume and resumed["rng"].get("rsi_self_builder") is None:
            raise ValueError("Resuming pre-RSI checkpoint requires explicit --rsi-cycle-updates 0 or migration")
        print("RSI self-construction CORE enabled:", rsi_builder.state_dict(), flush=True)
    elif args.resume and resumed["rng"].get("rsi_self_builder"):
        raise ValueError("Cannot disable inherited RSI core without explicit migration")
    before = byte_metrics_wrapped(model, probes, n=None,
                                  topk_n=min(args.topk_probes, len(probes)), auxiliary=probe_aux,
                                  local=probe_local)
    print("DEV before:", before)
    c256_before = (compare_c256_wrapped(
        model, dev, args.reference_full256, auxiliary=dev_aux,
        local=dev_local, probe_count=min(16, args.max_dev_positions))
        if args.reference_full256 else None)
    if c256_before is not None:
        print("Original C full256 numerical parity BEFORE:", c256_before)
    clock = time.monotonic()
    update_count = 0
    while ((update_count < args.updates) if args.updates is not None
           else (time.monotonic() - clock < args.seconds)):
        indices = coverage_indices(sampler["cursor"], args.batch,
                                   len(train_data), sampler["stride"],
                                   torch.device("cpu"))
        sample = build_causal_contexts(raw_tensor, indices + PARITY_CONTEXT,
                                       upgrade_context).to(device)
        sample_aux = aux_data[indices].to(device=device)
        report = (rsi_builder.train_step(train_one_batch, sample[:, :-1],
                                        sample[:, -1], sample_aux)
                  if rsi_builder is not None else
                  train_one_batch(model, optimizer, sample[:, :-1],
                                  sample[:, -1], sample_aux))
        sampler["cursor"] += args.batch
        step += 1
        update_count += 1
        if step % 50 == 0:
            print("RSI step", step, "train", report, flush=True)
    # Time ONLY the byte optimizer loop here. RSI feedback and held-out DEV
    # evaluation are reported separately; otherwise updates/s is misleading.
    training_duration_seconds = time.monotonic() - clock
    # The archived v103 original ALWAYS performs both verified actor/critic
    # Adam updates, even when --seconds=0 (zero BYTE optimizer updates).
    # Keep that exact recursive learning behavior in all three presets.
    rsi_clock = time.monotonic()
    events.extend(finish_rsi_round(model, optimizer))
    rsi_feedback_seconds = time.monotonic() - rsi_clock
    dev_after_clock = time.monotonic()
    after = byte_metrics_wrapped(model, probes, n=None,
                                 topk_n=min(args.topk_probes, len(probes)), auxiliary=probe_aux,
                                 local=probe_local)
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
               "sampler": sampler, "feature_digest": manifest["feature_sha256"],
               "terminal_events": events,
               "rsi_self_construction": (rsi_builder.state_dict() if rsi_builder else None),
               "rsi_mutations_verified": (rsi_builder.proposals if rsi_builder else 0),
               "rsi_mutations_accepted": (rsi_builder.accepted if rsi_builder else 0),
               "book_disjoint_dev_digest": dev_digest,
               "new_dev_books": fresh_dev_metadata["books"] if fresh_dev_metadata else None,
               "new_dev_scored_positions": fresh_dev_metadata["scored_positions"] if fresh_dev_metadata else None,
               "new_dev_id_level_train_disjoint_proven": (fresh_dev_metadata["id_level_train_disjoint_proven"]
                                                          if fresh_dev_metadata else None),
               "original_c256_parity_before": c256_before,
               "original_c256_parity_after": c256_after,
               "test_untouched": True,
               "old_v103_logits_identical_at_gate_zero": proof["exact_parity"]}
    provenance["sampler"] = sampler
    save_combined(args.output, model, optimizer, step, provenance, events,
                  training_state={"sampler": sampler,
                                  "torch_rng": torch.get_rng_state(),
                                  "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                                  "python_rng": random.getstate(),
                                  "rsi_self_builder": (rsi_builder.state_dict()
                                                       if rsi_builder else None)})
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
    if args.metrics_output:
        _write_if_unchanged_or_new(args.metrics_output,
            (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode())
    print("Unified RSI CPU/T4/Blackwell training receipt:", json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    main()
