#!/usr/bin/env python3
"""FA1 R7.54: checkpoint-backed R7.49 -> R7.53 successor CPU pilot.

One existing FA1 model/optimizer; only four new R7.53 parameters train.
TRAIN only is used for gradients, book-disjoint DEV for selection, TEST sealed.
A successful source test is NOT a claim of trained-model improvement.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from fa1_r723_train import prepare, take, forward
from fa1_r727_checkpoint import load_unified, save_unpromoted, digest
from fa1_r727_optimizer import _deep_equal
from fa1_r744_cpu_pilot import bootstrap_by_book
from fa1_r748_cpu_pilot import verify_train_dev_only
from fa1_r751_cpu_pilot import balanced_batch_indices
from fa1_r753_causal_successor import (
    attach_causal_successors, successor_control, ROUND as R753)
from fa1_r753_corpus import balanced_causal_positions, with_successor_history

PARENT = "FA1_R7_49_CAUSAL_PAIR_ADDRESS_RELIABILITY_V1"
PARENT_SHA256 = "b8dabe54e2417a75defd01c776606367a3f3b0f18f479b032f8390b6fc87ddb5"
NEW_NAMES = (
    "word_backoff.r753_word_gain",
    "word_backoff.r753_geometry_gain",
    "word_backoff.r753_reliability.weight",
    "word_backoff.r753_reliability.bias",
)


def frozen_fingerprint(named, optimizer):
    """Hash inherited weights AND their live optimizer slots without copying."""
    h = hashlib.sha256()
    for name, p in sorted(named.items()):
        h.update(name.encode())
        tensors = [("weight", p.detach())]
        old_state = optimizer._adamw.state.get(p, optimizer._legacy_state.get(p, {}))
        for key in sorted(old_state):
            value = old_state[key]
            if torch.is_tensor(value):
                tensors.append((key, value.detach()))
            else:
                h.update((key + ":" + repr(value)).encode())
        for key, tensor in tensors:
            h.update((key + ":" + str(tensor.dtype) + ":" +
                      repr(tuple(tensor.shape))).encode())
            h.update(tensor.cpu().contiguous().reshape(-1)
                     .view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def attach_histories(a, data, split, context):
    return with_successor_history(data, a.root, split, a.corpus_dir, context)


def foreign_successor_history(data, seed):
    """Counterfactual new history only: inherited R7.49 inputs stay fixed."""
    books = data["book_ids"].cpu().numpy()
    unique = np.unique(books)
    if len(unique) < 2:
        raise ValueError("foreign history requires at least two distinct books")
    rng = np.random.default_rng(seed)
    donors = np.empty(len(books), dtype=np.int64)
    by_book = {int(book): np.flatnonzero(books == book) for book in unique}
    for i, book in enumerate(books):
        candidates = unique[unique != book]
        donor_book = int(rng.choice(candidates))
        donors[i] = rng.choice(by_book[donor_book])
    ids = torch.as_tensor(donors, dtype=torch.long)
    wrong = dict(data)
    for key in ("r753_successor_history_ids", "r753_successor_history_mask"):
        wrong[key] = data[key][ids]
    return wrong


def book_gate(measures, names, books, seed, min_gain=0.):
    """Only independently resampled BOOKS can authorize a positive claim."""
    trials = {}
    for n, name in enumerate(measures):
        trials[name] = bootstrap_by_book(measures[name], books, seed + n)
    passed = all(
        float(measures[name].mean()) > min_gain and
        trials[name]["book_bootstrap_ci95"][0] > 0.
        for name in names
    )
    return bool(passed), trials


@torch.no_grad()
def evaluate(model, dev, seed):
    """Matched target positions for every causal / structural intervention."""
    model.eval()
    with successor_control(model, "disabled"):
        parent, _ = forward(model, dev)
    with successor_control(model, "word_only"):
        word, _ = forward(model, dev)
        word_foreign, _ = forward(model, foreign_successor_history(dev, seed + 1))
    full, ex = forward(model, dev)
    with successor_control(model, "wrong_addresses"):
        wrong_addresses, _ = forward(model, dev)
    with successor_control(model, "wrong_successors"):
        wrong_values, _ = forward(model, dev)
    with successor_control(model, "no_apex"):
        no_apex, _ = forward(model, dev)
    foreign, _ = forward(model, foreign_successor_history(dev, seed + 1))
    measurements = {
        "gain_over_r749": parent - full,
        "word_gain_over_r749": parent - word,
        "word_correct_vs_foreign": word_foreign - word,
        "geometry_over_word_only": word - full,
        "correct_vs_wrong_addresses": wrong_addresses - full,
        "correct_vs_wrong_successors": wrong_values - full,
        "correct_vs_foreign_book": foreign - full,
        "successor_apex_necessity": no_apex - full,
    }
    books = dev["book_ids"]
    word_keys = ("word_gain_over_r749", "word_correct_vs_foreign")
    full_keys = ("gain_over_r749", "geometry_over_word_only",
                 "correct_vs_wrong_addresses", "correct_vs_wrong_successors",
                 "correct_vs_foreign_book", "successor_apex_necessity")
    word_ok, word_boot = book_gate(measurements, word_keys, books, seed + 10)
    geo_ok, geo_boot = book_gate(measurements, full_keys, books, seed + 50)
    measures = {
        "dev_books": int(torch.unique(books).numel()),
        "dev_positions": int(len(parent)),
        "r749_word_nll": float(parent.mean()),
        "word_only_nll": float(word.mean()),
        "candidate_word_nll": float(full.mean()),
        "eligible_geo_fraction": float(ex["r753_geometric_eligible"].float().mean()),
        "mean_word_gain": float(ex["r753_word_weight"].mean()),
        "mean_geometric_gain": float(ex["r753_geometric_weight"].mean()),
        "word_gate_passed": word_ok,
        "strict_geometry_dev_gate_passed": geo_ok and word_ok,
        "book_bootstrap": {**word_boot, **geo_boot},
    }
    measures.update({name: float(values.mean()) for name, values in measurements.items()})
    return measures


def select_score(metrics):
    return (
        int(metrics["strict_geometry_dev_gate_passed"]),
        int(metrics["word_gate_passed"]),
        -metrics["candidate_word_nll"],
    )


def validate_configuration(a):
    if (a.word_steps <= 0 or a.geo_steps <= 0 or a.train < a.minibatch or
            a.dev < 15 or a.minibatch <= 0 or a.threads not in (1, 2, 4) or
            a.context < 2 or a.context > 32768 or
            a.long_context < a.context or a.long_context > 32768 or
            not (0 < a.open_word < 1) or not (0 < a.open_geometry < 1) or
            a.eval_every <= 0):
        raise ValueError("invalid FA1 R7.54 pilot settings")
    if Path(a.output_checkpoint).resolve() == Path(a.checkpoint).resolve():
        raise ValueError("protected R7.49 checkpoint cannot be overwritten")
    if Path(a.report).exists() or Path(a.output_checkpoint).exists():
        raise FileExistsError("R7.54 output already exists")


def run(a):
    validate_configuration(a)
    started = time.monotonic()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)

    corpus_inventory = verify_train_dev_only(a.corpus_dir, a.split_metadata)
    source_sha = digest(a.checkpoint)
    if source_sha != PARENT_SHA256:
        raise ValueError("protected R7.49 checkpoint SHA-256 mismatch")
    model, optimizer, base, compositor, meta = load_unified(
        a.checkpoint, a.root, a.threads)
    if (meta.get("architecture_round") != PARENT or
            meta.get("promoted") is not False or
            len(optimizer.param_groups) != 29):
        raise ValueError("must begin from intact 29-group unpromoted R7.49")

    a.r753_position_selector = balanced_causal_positions
    train = attach_histories(
        a, prepare(a, model, base, compositor, "train", a.train, a.seed + 101),
        "train", a.context)
    dev = attach_histories(
        a, prepare(a, model, base, compositor, "dev", a.dev, a.seed + 202),
        "dev", a.context)
    if int(torch.unique(train["book_ids"]).numel()) < a.min_train_books:
        raise ValueError("TRAIN does not include enough independent books")
    if int(torch.unique(dev["book_ids"]).numel()) != 15:
        raise ValueError("DEV must cover all 15 books; independent TEST stays sealed")

    model.eval()
    with torch.no_grad():
        parent_nll, _ = forward(model, dev)
    original = dict(model.named_parameters())
    original_hash = frozen_fingerprint(original, optimizer)
    for param in original.values():
        param.requires_grad_(False)
    attach_causal_successors(model, optimizer, learning_rate=a.lr)
    with torch.no_grad():
        migrated, _ = forward(model, dev)
    if not torch.equal(parent_nll, migrated):
        raise AssertionError("R7.49 to R7.53 zero-gain parity failed")
    new = {n: p for n, p in model.named_parameters() if n not in original}
    if set(new) != set(NEW_NAMES) or len(optimizer.param_groups) != 30:
        raise AssertionError("incorrect four-tensor/30-group successor topology")
    if frozen_fingerprint(original, optimizer) != original_hash:
        raise AssertionError("inherited state changed during attachment")

    rng = np.random.default_rng(a.seed + 303)
    snapshots = []
    best = None
    saved_weights = None
    saved_moments = None
    step_total = 0

    def capture(stage, step, train_loss):
        nonlocal best, saved_weights, saved_moments
        m = evaluate(model, dev, a.seed + 900 + step_total)
        m.update({"stage": stage, "stage_step": step,
                  "total_updates": step_total, "train_word_nll": train_loss})
        snapshots.append(m)
        rank = select_score(m)
        if best is None or rank > best[0]:
            best = (rank, m)
            saved_weights = {n: p.detach().clone() for n, p in new.items()}
            saved_moments = {n: copy.deepcopy(optimizer._adamw.state.get(p, {}))
                             for n, p in new.items()}
        print("FA1_R754_DEV", json.dumps({
            key: m[key] for key in
            ("stage", "total_updates", "r749_word_nll", "candidate_word_nll",
             "word_gain_over_r749", "geometry_over_word_only",
             "strict_geometry_dev_gate_passed")}), flush=True)

    with torch.no_grad():
        model.word_backoff.r753_word_gain.fill_(a.open_word)
    for stage, steps in (("word_only", a.word_steps), ("geometry", a.geo_steps)):
        if stage == "geometry":
            with torch.no_grad():
                model.word_backoff.r753_geometry_gain.fill_(a.open_geometry)
        for step in range(1, steps + 1):
            batch_ix = balanced_batch_indices(train["book_ids"], a.minibatch, rng)
            batch = take(train, batch_ix)
            model.eval()
            with successor_control(model, stage if stage == "word_only" else "full"):
                nll, _ = forward(model, batch)
            loss = nll.mean()
            optimizer._adamw.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None for p in original.values()):
                raise AssertionError("inherited FA1 parameter received gradients")
            torch.nn.utils.clip_grad_norm_(list(new.values()), 1.0)
            optimizer._adamw.step()
            step_total += 1
            if step in (1, steps) or step % a.eval_every == 0:
                capture(stage, step, float(loss.detach()))

    if best is None or saved_weights is None:
        raise AssertionError("no DEV candidate was measured")
    with torch.no_grad():
        for n, p in new.items():
            p.copy_(saved_weights[n])
    for n, p in new.items():
        if saved_moments[n]:
            optimizer._adamw.state[p] = copy.deepcopy(saved_moments[n])
        else:
            optimizer._adamw.state.pop(p, None)

    selected = evaluate(model, dev, a.seed + 2001)
    long_dev = attach_histories(a, dev, "dev", a.long_context)
    long_selected = evaluate(model, long_dev, a.seed + 2001)
    long_gate = (long_selected["gain_over_r749"] >= 0. and
                 long_selected["correct_vs_foreign_book"] > 0.)
    source_integrity = digest(a.checkpoint) == source_sha
    frozen_integrity = frozen_fingerprint(original, optimizer) == original_hash
    if not source_integrity or not frozen_integrity:
        raise AssertionError("protected R7.49 weights or optimizer were modified")

    # Unpromoted candidate ALWAYS: DEV selection alone never opens TEST.
    checkpoint = save_unpromoted(
        a.output_checkpoint, model, optimizer, a.root, a.checkpoint,
        source_metadata=meta, candidate_trained=True, training_round=R753,
        training_updates=best[1]["total_updates"],
        development_gate_passed=False, architecture_round=R753)
    verified, opt_verified, _, _, saved_meta = load_unified(
        a.output_checkpoint, a.root, a.threads)
    if (saved_meta.get("architecture_round") != R753 or
            saved_meta.get("promoted") is not False or
            len(opt_verified.param_groups) != 30):
        raise AssertionError("reloaded R7.53 metadata/optimizer topology mismatch")
    verified_old = {n: p for n, p in verified.named_parameters() if n in original}
    if frozen_fingerprint(verified_old, opt_verified) != original_hash:
        raise AssertionError("reloaded inherited weights/Adam moments changed")
    loaded_new = dict(verified.named_parameters())
    for n, p in new.items():
        if (not torch.equal(p, loaded_new[n]) or
                not _deep_equal(optimizer._adamw.state.get(p, {}),
                                opt_verified._adamw.state.get(loaded_new[n], {}))):
            raise AssertionError("R7.53 new tensor or Adam moment failed reload")

    report = {
        "architecture": R753, "training_driver": "FA1_R7_54_CPU_PILOT_V1",
        "branch": "master", "promoted": False,
        "sealed_independent_test_opened": False,
        "source_r749_sha256": source_sha,
        "candidate_sha256": checkpoint["sha256"],
        "one_optimizer_groups": 30, "inherited_groups_preserved": 29,
        "added_trainable_tensors": list(NEW_NAMES),
        "frozen_parent_verified": True, "saved_reloaded_exact": True,
        "train_books": int(torch.unique(train["book_ids"]).numel()),
        "dev_books": int(torch.unique(dev["book_ids"]).numel()),
        "training_positions": len(train["target"]),
        "dev_positions": len(dev["target"]),
        "updates_executed": step_total,
        "selected_update": best[1]["total_updates"],
        "selected_stage": best[1]["stage"],
        "word_gate_passed": selected["word_gate_passed"],
        "strict_geometry_dev_gate_passed": selected["strict_geometry_dev_gate_passed"],
        "long_context_nonregression": bool(long_gate),
        "eligible_for_further_dev_only_training": bool(
            selected["strict_geometry_dev_gate_passed"] and long_gate),
        "matched_dev": selected, "long_context_dev": long_selected,
        "source_checkpoint_immutable": source_integrity,
        "corpus_inventory": corpus_inventory,
        "full_mixture_word_top5_available": False,
        "true_contiguous_byte_bpb_available": False,
        "reason_no_bpb": "R7.53 is a normalized word-level mixture, not a raw-byte autoregressive decoder",
        "development_history": snapshots,
        "elapsed_seconds": time.monotonic() - started,
    }
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print("FA1_R754_SAVED", checkpoint["sha256"], "SELECTED",
          report["selected_update"], "STRICT_DEV",
          report["strict_geometry_dev_gate_passed"], flush=True)
    return report


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "corpus_dir", "split_metadata", "checkpoint",
                "output_checkpoint", "report"):
        p.add_argument("--" + key.replace("_", "-"), required=True)
    p.add_argument("--train", type=int, default=2048)
    p.add_argument("--dev", type=int, default=512)
    p.add_argument("--min-train-books", type=int, default=200)
    p.add_argument("--minibatch", type=int, default=16)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--word-steps", type=int, default=256)
    p.add_argument("--geo-steps", type=int, default=744)
    p.add_argument("--eval-every", type=int, default=248)
    p.add_argument("--threads", type=int, choices=(1, 2, 4), default=4)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--long-context", type=int, default=2048)
    p.add_argument("--open-word", type=float, default=.25)
    p.add_argument("--open-geometry", type=float, default=.25)
    p.add_argument("--lr", type=float, default=3.e-4)
    p.add_argument("--seed", type=int, default=754)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
