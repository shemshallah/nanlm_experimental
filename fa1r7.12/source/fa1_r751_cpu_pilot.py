#!/usr/bin/env python3
"""R7.51 CPU continuation: one optimizer, calibration-first, then signed pairs.

R7.49 is immutable. Train uses only book-disjoint NaNROUTER TRAIN; DEV is
used for selection; no TEST file or TEST model score is accessed. Save only an
unpromoted candidate with all 29 inherited Adam groups intact.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from fa1_r723_train import prepare, take, forward
from fa1_r727_checkpoint import load_unified, save_unpromoted, digest
from fa1_r744_cpu_pilot import bootstrap_by_book, foreign_word_history
from fa1_r746_factorial_dev import perturbed_context
from fa1_r748_cpu_pilot import verify_train_dev_only
from fa1_r749_address_reliability import ROUND as PARENT_ROUND
from fa1_r751_bidirectional_evidence import (
    ROUND, attach_bidirectional_pair_evidence, bidirectional_control,
)

PARENT_SHA256 = "b8dabe54e2417a75defd01c776606367a3f3b0f18f479b032f8390b6fc87ddb5"


@torch.no_grad()
def evaluate(model, dev, seed):
    model.eval()
    wrong = perturbed_context(dev, "addresses_only", seed + 1)
    foreign, _ = foreign_word_history(dev, seed + 2)
    full, telemetry = forward(model, dev)
    bad, _ = forward(model, wrong)
    foreign_nll, _ = forward(model, foreign)
    with bidirectional_control(model, "disabled"):
        parent, _ = forward(model, dev)
    with bidirectional_control(model, "calibration_only"):
        calibrated, _ = forward(model, dev)
        wrong_calibrated, _ = forward(model, wrong)
    with bidirectional_control(model, "constant_bank"):
        constant, _ = forward(model, dev)
    with bidirectional_control(model, "permute_pair_values"):
        swapped, _ = forward(model, dev)
    with bidirectional_control(model, "fixed_confidence"):
        fixed, _ = forward(model, dev)
    books = dev["book_ids"]
    measures = {
        "word_nll": full,
        "gain_over_r749": parent - full,
        "gain_over_score_only": calibrated - full,
        "exact_address_specificity": (calibrated - full) -
                                     (wrong_calibrated - bad),
        "learned_pair_bank_necessity": constant - full,
        "swapped_pair_bank_necessity": swapped - full,
        "conditional_confidence_necessity": fixed - full,
        "correct_history_advantage": foreign_nll - full,
    }
    report = {
        "dev_books": int(torch.unique(books).numel()),
        "dev_positions": len(full),
        "r749_word_nll": float(parent.mean()),
        "score_only_word_nll": float(calibrated.mean()),
        "mean_pair_delta": float(telemetry["r751_signed_delta"].mean()),
        "exact_pair_fraction": float(telemetry["r751_exact_hit_fraction"].mean()),
        "mean_copy_repeat_confidence": float(
            telemetry["r751_copy_confidence"].mean()),
        "mean_context_trust": float(telemetry["r751_context_trust"].mean()),
    }
    for i, (name, tensor) in enumerate(measures.items()):
        report[name] = float(tensor.mean())
        report[name + "_whole_book_bootstrap"] = bootstrap_by_book(
            tensor, books, seed + 40 + i)
    mandatory = (
        "gain_over_r749", "gain_over_score_only",
        "exact_address_specificity", "learned_pair_bank_necessity",
        "swapped_pair_bank_necessity", "correct_history_advantage",
    )
    report["strict_dev_gate_passed"] = all(
        report[name] > 0. and
        report[name + "_whole_book_bootstrap"]["book_bootstrap_ci95"][0] > 0.
        for name in mandatory)
    return report


def balanced_batch_indices(book_ids, batch_size, rng):
    """Choose distinct available books before sampling multiple positions."""
    books = book_ids.detach().cpu().numpy()
    chosen_books = rng.choice(np.unique(books), size=batch_size,
                              replace=batch_size > len(np.unique(books)))
    idx = [int(rng.choice(np.flatnonzero(books == book)))
           for book in chosen_books]
    return torch.tensor(idx, dtype=torch.long)


def train_step(model, optimizer, batch, new, score_only):
    model.eval()
    good, _ = forward(model, batch)
    # The same full next-word NLL supervises each stage. This avoids a
    # synthetic attribution reward that might harm language prediction.
    bias = model.word_backoff.mirror_r751_global_pair_bias
    regularizer = (1.e-5 * bias.square() +
                   1.e-6 * model.word_backoff.mirror_r751_positive.weight.square().mean() +
                   1.e-6 * model.word_backoff.mirror_r751_negative.weight.square().mean())
    loss = good.mean() + regularizer
    optimizer._adamw.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for _, p in new if p.requires_grad], 1.)
    optimizer._adamw.step()
    return float(loss.detach())


def run(a):
    if Path(a.output_checkpoint).exists() or Path(a.report).exists():
        raise FileExistsError("R7.51 outputs must not overwrite existing work")
    start = time.monotonic()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    inventory = verify_train_dev_only(a.corpus_dir, a.split_metadata)
    parent_hash = digest(a.checkpoint)
    if parent_hash != a.expected_sha256:
        raise ValueError("R7.49 checkpoint SHA-256 does not match protected lineage")
    model, optimizer, base, comp, meta = load_unified(a.checkpoint, a.root,
                                                      a.threads)
    if (meta["architecture_round"] != PARENT_ROUND or meta["promoted"] is not False
            or len(optimizer.param_groups) != 29):
        raise ValueError("R7.51 must extend the original unpromoted R7.49")
    train = prepare(a, model, base, comp, "train", a.train, a.seed + 101)
    dev = prepare(a, model, base, comp, "dev", a.dev, a.seed + 202)
    seen_train = int(torch.unique(train["book_ids"]).numel())
    if seen_train < a.min_train_books:
        raise ValueError("training pool does not cover enough TRAIN books")
    if int(torch.unique(dev["book_ids"]).numel()) != 15:
        raise ValueError("all 15 DEV books must be sampled, TEST stays sealed")
    with torch.no_grad():
        model.eval()
        before, _ = forward(model, dev)
    old = {name: p for name, p in model.named_parameters()}
    for p in old.values():
        p.requires_grad_(False)
    attach_bidirectional_pair_evidence(model, optimizer, a.lr)
    with torch.no_grad():
        after, _ = forward(model, dev)
    if not torch.equal(before, after):
        raise AssertionError("bitwise R7.49 -> R7.51 zero-gate parity failed")
    new = [(name, p) for name, p in model.named_parameters() if name not in old]
    if len(new) != 7 or len(optimizer.param_groups) != 30:
        raise AssertionError("R7.51 must retain 29 groups and append only 7 tensors")
    for _, param in new:
        param.requires_grad_(False)
    model.word_backoff.mirror_r751_temperature.requires_grad_(True)
    rng = np.random.default_rng(a.seed + 303)
    history = []
    best = None
    saved_weights = None
    saved_moments = None

    def capture(stage, step, loss):
        nonlocal best, saved_weights, saved_moments
        measures = evaluate(model, dev, a.seed + 900 + step)
        measures.update({"stage": stage, "step": step, "train_loss": loss})
        history.append(measures)
        score = (int(measures["strict_dev_gate_passed"]),
                 -measures["word_nll"])
        if best is None or score > best[0]:
            best = (score, measures)
            saved_weights = {name: p.detach().clone() for name, p in new}
            saved_moments = {name: copy.deepcopy(optimizer._adamw.state.get(p, {}))
                             for name, p in new}
        print("R751 DEV", stage, step, json.dumps({
            name: measures[name] for name in (
                "word_nll", "score_only_word_nll", "gain_over_score_only",
                "exact_address_specificity", "strict_dev_gate_passed",
            )}), flush=True)
        return measures

    calibrator_dev = None
    for step in range(1, a.calibration_steps + 1):
        batch = take(train, balanced_batch_indices(train["book_ids"],
                                                     a.minibatch, rng))
        loss = train_step(model, optimizer, batch, new, True)
        if any(p.grad is not None for p in old.values()):
            raise AssertionError("an inherited FA1 parameter received gradients")
        if step in (1, a.calibration_steps):
            calibrator_dev = capture("score_only", step, loss)
    for _, param in new:
        param.requires_grad_(True)
    # Open only after exact zero parity and calibration-only pretraining.
    with torch.no_grad():
        model.word_backoff.mirror_r751_address_gain.fill_(a.open_address_gain)
    checks = set(a.eval_steps) | {1, a.joint_steps}
    for step in range(1, a.joint_steps + 1):
        batch = take(train, balanced_batch_indices(train["book_ids"],
                                                     a.minibatch, rng))
        loss = train_step(model, optimizer, batch, new, False)
        if any(p.grad is not None for p in old.values()):
            raise AssertionError("an inherited FA1 parameter received gradients")
        if step in checks:
            capture("bidirectional", step, loss)
    if best is None:
        raise AssertionError("no DEV candidate selected")
    for name, p in new:
        with torch.no_grad():
            p.copy_(saved_weights[name])
        optimizer._adamw.state[p] = copy.deepcopy(saved_moments[name])
    if digest(a.checkpoint) != parent_hash:
        raise AssertionError("protected R7.49 checkpoint changed")
    selected = evaluate(model, dev, a.seed + 1800)
    saved = save_unpromoted(a.output_checkpoint, model, optimizer, a.root,
                            a.checkpoint, source_metadata=meta,
                            candidate_trained=True, training_round=ROUND,
                            training_updates=best[1]["step"],
                            development_gate_passed=False,
                            architecture_round=ROUND)
    result = {
        "round": ROUND, "branch": "master", "promoted": False,
        "independent_test_opened": False, "source_sha256": parent_hash,
        "optimizer_count": 1, "optimizer_groups": 30,
        "inherited_groups_frozen": 29, "inherited_adam_state_preserved": True,
        "train_books_total": inventory["train"]["books"],
        "train_books_in_sample": seen_train, "dev_books": 15,
        "calibration_steps": a.calibration_steps, "joint_steps": a.joint_steps,
        "selected_stage": best[1]["stage"],
        "selected_update": best[1]["step"],
        "score_only_pretraining_dev": calibrator_dev,
        "selected_in_memory_candidate_dev": selected,
        "strict_dev_gate_passed": selected["strict_dev_gate_passed"],
        "candidate_checkpoint": saved, "development_history": history,
        "full_word_top5_available": False, "raw_text_bpb_available": False,
        "promotion": "NOT PROMOTED: TEST and long-context gates remain sealed",
        "elapsed_seconds": time.monotonic() - start,
    }
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("R751 UNPROMOTED", saved["sha256"], flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--split-metadata", required=True)
    p.add_argument("--corpus-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-checkpoint", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--expected-sha256", default=PARENT_SHA256)
    p.add_argument("--train", type=int, default=2048)
    p.add_argument("--dev", type=int, default=512)
    p.add_argument("--min-train-books", type=int, default=150)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--minibatch", type=int, default=16)
    p.add_argument("--calibration-steps", type=int, default=32)
    p.add_argument("--joint-steps", type=int, default=96)
    p.add_argument("--eval-steps", type=int, nargs="*",
                   default=[16, 32, 48, 72, 96])
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=751)
    p.add_argument("--lr", type=float, default=4.e-4)
    p.add_argument("--open-address-gain", type=float, default=.5)
    run(p.parse_args())
