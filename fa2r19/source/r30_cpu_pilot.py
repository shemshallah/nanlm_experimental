#!/usr/bin/env python3
"""Guarded real-book CPU pilot for the FA2 R30 diamond Overmind.

Restores the immutable R19 + R27 + R28 lineage and adds exactly one new
R30 optimizer group. Never trains or overwrites an inherited parameter.
Candidate overlays are exploratory and MUST NOT be treated as an accepted
checkpoint without independent-author and causal long-prefix verification.
"""
import argparse
import copy
import json
import math
import os
import pathlib
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from r13_corpus_cache import read_books_cached
from r23_stream_carry import ContinuousPALState, contiguous_book_picks
from r28_recurrent import bind_recurrent
from r28_restore_unified import restore
from r28_cpu_continue import metrics, equal_nested, sha
from r30_diamond_overmind import attach_r30


def invariant(model, opt, previous_model, previous_opt, parent_digests):
    current = model.state_dict()
    for name, tensor in previous_model.items():
        if not torch.equal(tensor, current[name]):
            raise AssertionError("inherited parameter mutated: " + name)
    now = opt.state_dict()
    count = len(previous_opt["param_groups"])
    if now["param_groups"][:count] != previous_opt["param_groups"]:
        raise AssertionError("inherited Adam group mutated")
    if not equal_nested({k: now["state"][k] for k in previous_opt["state"]},
                        previous_opt["state"]):
        raise AssertionError("inherited Adam moment mutated")
    for path, digest in parent_digests.items():
        if sha(path) != digest:
            raise AssertionError("parent checkpoint modified: " + str(path))


def run(args):
    root = pathlib.Path(args.root).resolve()
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.monotonic()

    model, opt, comp, r6, _, r27, r28 = restore(root, threads=args.threads)
    if r28.get("r28_step") != 16 or len(opt.param_groups) != 13 or len(opt.state) != 153:
        raise ValueError("R30 must start from the verified R28 CPU16 lineage")
    parents = {root / "reports/fa2_r27_unified_candidate.pth":
               sha(root / "reports/fa2_r27_unified_candidate.pth"),
               root / "reports/fa2_r28_adapter_overlay.pt":
               sha(root / "reports/fa2_r28_adapter_overlay.pt")}
    before_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
    before_opt = copy.deepcopy(opt.state_dict())
    books, _ = read_books_cached(root / "nanrouter", comp, r6.vocab, args.max_books)
    training = books["train"][:args.train_books]
    development = books["dev"][:args.dev_books]
    if not training or not development or set(b.book_id for b in training) & set(b.book_id for b in development):
        raise ValueError("missing or overlapping real-book train and development data")

    first_seg = next(contiguous_book_picks(development[0], args.chunk_span))
    with torch.no_grad():
        old_logits = model.forward_chunk(development[0], first_seg,
                           bind_recurrent(ContinuousPALState(development[0].book_id), model))["logits"]
    attach_r30(model, opt, width=args.width, active_heads=args.heads,
               active_pairs=args.pairs, active_triples=args.triples,
               active_apex=args.apex, learning_rate=args.lr)
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("r30."))
    with torch.no_grad():
        new_logits = model.forward_chunk(development[0], first_seg,
                           bind_recurrent(ContinuousPALState(development[0].book_id), model))["logits"]
    if not torch.equal(old_logits, new_logits):
        raise AssertionError("R30 is not bitwise identical to inherited R28 at initialization")
    initial = metrics(model, development, args.dev_chunks, args.chunk_span)
    best = None
    plans = [list(contiguous_book_picks(b, args.chunk_span)) for b in training]
    offsets = [0] * len(training)
    trace = []
    print("R30_INIT " + json.dumps({"baseline": initial, "overmind_parity": True}), flush=True)

    for step in range(1, args.steps + 1):
        idx = (step - 1) % len(training)
        book = training[idx]
        if offsets[idx] >= len(plans[idx]):
            offsets[idx] = 0
        seg = plans[idx][offsets[idx]]
        offsets[idx] += 1
        model.train()
        carry = bind_recurrent(ContinuousPALState(book.book_id), model)
        opt.zero_grad(set_to_none=True)
        row = model.forward_chunk(book, seg, carry)
        token_nll = F.cross_entropy(row["logits"], row["target"], reduction="none")
        weights = torch.where(row["target"] == 32, args.space_weight, 1.)
        loss = (token_nll * weights).sum() / weights.sum()
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.r30.parameters(), args.clip))
        if not math.isfinite(grad) or not torch.isfinite(loss):
            raise ValueError("nonfinite R30 gradient/loss")
        opt.step()
        event = {"step": step, "training_tokens": len(row["target"]),
                 "nll": float(loss.detach()), "grad_norm": grad}
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            current = metrics(model, development, args.dev_chunks, args.chunk_span)
            safe = (current["bpb"] < initial["bpb"] - args.min_gain
                    and current["nonspace_nll"] < initial["nonspace_nll"] - args.min_gain
                    and all(current["books"].get(k, math.inf) <= v + args.perbook_slack
                            for k, v in initial["books"].items()))
            event.update({"dev": current, "passes_development_guards": safe})
            if safe and (best is None or current["bpb"] < best["dev"]["bpb"]):
                best = {"step": step, "dev": current,
                        "new_weights": copy.deepcopy(model.r30.state_dict()),
                        "new_optimizer": copy.deepcopy(opt.state_dict())}
        trace.append(event)
        print("R30_STEP " + json.dumps(event), flush=True)

    invariant(model, opt, before_model, before_opt, parents)
    out = {"format": "FA2_R30_OVERMIND_CPU_PILOT_V1",
           "status": "development_candidate_only" if best else "no_development_winner",
           "inherited_adam_states": len(before_opt["state"]),
           "inherited_optimizer_groups": len(before_opt["param_groups"]),
           "stream_snapshots": len(r27["selected"]["carry"]),
           "r27_sha256": list(parents.values())[0],
           "r28_sha256": list(parents.values())[1],
           "initial": initial, "selected_step": best["step"] if best else 0,
           "selected_dev": best["dev"] if best else initial,
           "trace": trace, "settings": vars(args),
           "elapsed_s": time.monotonic() - started,
           "accepted_model_promoted": False}
    if best is not None:
        snapshot = best["new_optimizer"]
        new_ids = snapshot["param_groups"][-1]["params"]
        payload = {"format": "FA2_R30_OVERMIND_DELTA_V1",
                   "accepted_model_promoted": False,
                   "parent_r27_sha256": out["r27_sha256"],
                   "parent_r28_sha256": out["r28_sha256"],
                   "selected_step": best["step"], "selected_dev": best["dev"],
                   "r30_state": best["new_weights"],
                   "r30_optimizer_group": snapshot["param_groups"][-1],
                   "r30_optimizer_states": {k: snapshot["state"][k] for k in new_ids
                                            if k in snapshot["state"]},
                   "inherited_adam_states": len(before_opt["state"]),
                   "inherited_stream_count": len(r27["selected"]["carry"])}
        destination = root / "reports/fa2_r30_overmind_candidate.pt"
        temporary = destination.with_suffix(".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        out["candidate_overlay_sha256"] = sha(destination)
    report = root / "reports/r30_cpu_pilot.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(out, indent=2) + "\n")
    print("R30_FINAL " + json.dumps({"status": out["status"],
          "selected_step": out["selected_step"], "initial": initial["bpb"],
          "best": out["selected_dev"]["bpb"]}), flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--train-books", type=int, default=2)
    parser.add_argument("--dev-books", type=int, default=2)
    parser.add_argument("--max-books", type=int, default=2)
    parser.add_argument("--chunk-span", type=int, default=12)
    parser.add_argument("--dev-chunks", type=int, default=12)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=30030)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--pairs", type=int, default=24)
    parser.add_argument("--triples", type=int, default=24)
    parser.add_argument("--apex", type=int, default=8)
    parser.add_argument("--lr", type=float, default=.0003)
    parser.add_argument("--clip", type=float, default=2.)
    parser.add_argument("--space-weight", type=float, default=.35)
    parser.add_argument("--min-gain", type=float, default=1e-6)
    parser.add_argument("--perbook-slack", type=float, default=1e-4)
    arguments = parser.parse_args()
    if not .05 <= arguments.space_weight <= 1.:
        raise ValueError("invalid space weight")
    run(arguments)
