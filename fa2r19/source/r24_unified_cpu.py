#!/usr/bin/env python3
"""FA2 R24: one trainable checkpoint/optimizer, contiguous-book CPU TBPTT.

Restores the accepted R18 model and all its optimizer moments, incorporates
the historical R21 and R22 optimizer states, and adds R23+mirror-MoE groups.
The original checkpoint, its SHA and frozen adapter files are never written.
A single experimental candidate (NOT a promotion) is selected on development.
"""
import argparse
import copy
import json
import math
import pathlib
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

import r14_slim_base as pal
from r13_corpus_cache import read_books_cached
from r18_joint_continue import build_joint_optimizer
from r18_long_memory import R18LongMemory
from r21_relevance_novelty import RelevanceNoveltyGate
from r22_multiscale_memory import R22Residual
from r23_context64 import (ContextualSpanMemory,
                            multipositive_value_loss,
                            multipositive_semantic_loss)
from r23_cpu_parallel import semantic_data
from r23_stream_carry import ContinuousPALState, contiguous_book_picks
from r24_unified_model import UnifiedFA2

ACCEPTED_SHA = "dd040106568786e7fbad2a880f4010635b626b51c3697e2ab08712905a2de91f"


def attach_historical_group(optimizer, module, checkpoint, lr, name):
    """Migrate AdamW moments by deterministic param-group order, not reset."""
    params = list(module.parameters())
    prior = checkpoint.get("optimizer")
    if not isinstance(prior, dict) or len(prior.get("param_groups", [])) != 1:
        raise RuntimeError(name + " has no compatible one-group Adam history")
    ids = prior["param_groups"][0]["params"]
    if len(ids) != len(params):
        raise RuntimeError(name + " optimizer parameter count mismatch")
    optimizer.add_param_group({"params": params, "lr": lr, "weight_decay": 1e-4})
    transferred = 0
    for param, ident in zip(params, ids):
        previous = prior.get("state", {}).get(ident, {})
        if previous:
            for key, value in previous.items():
                if isinstance(value, torch.Tensor) and key in ("exp_avg", "exp_avg_sq"):
                    if value.shape != param.shape:
                        raise RuntimeError(name + " Adam shape mismatch: " + key)
            optimizer.state[param] = copy.deepcopy(previous)
            transferred += 1
    if not transferred:
        raise RuntimeError(name + " has no Adam moments to preserve")
    return transferred


def initialize(root, args):
    accepted_path = root / "checkpoint/r12_active.pal12"
    if pal.sha(accepted_path) != ACCEPTED_SHA:
        raise RuntimeError("R18 parent checkpoint digest mismatch")
    r21_path = root / "reports/fa2_2_r21_gate_candidate.pth"
    r22_path = root / "reports/r22_multiscale_candidate.pth"
    h21, h22 = pal.sha(r21_path), pal.sha(r22_path)
    state21 = torch.load(r21_path, map_location="cpu", weights_only=False)
    state22 = torch.load(r22_path, map_location="cpu", weights_only=False)
    if (state21["parent_checkpoint_sha256"] != ACCEPTED_SHA or
            state22["parent_checkpoint_sha256"] != ACCEPTED_SHA or
            state22["frozen_r21_gate_sha256"] != h21 or
            state22.get("accepted_model_promoted", True)):
        raise RuntimeError("historical adapter lineage mismatch")

    saved, r6, comp, model, bridge, adapter, base_opt, _ = (
        pal.load_for_arm(root, accepted_path, True))
    if len(base_opt.param_groups) != 4 or "r18_memory" not in saved:
        raise RuntimeError("inherited optimizer groups are missing")
    r18 = R18LongMemory(model.memory.piece_emb.weight)
    r18.load_state_dict(saved["r18_memory"])
    base_opt, history = build_joint_optimizer(base_opt, r18, saved, args.r18_lr)
    r21 = RelevanceNoveltyGate()
    r21.load_state_dict(state21["gate"])
    r22 = R22Residual()
    r22.load_state_dict(state22["model"])
    # Only initialization uses a new seed: no new independent base model.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed + 24000)
        r23 = ContextualSpanMemory(model.memory.piece_emb.weight,
                                   minimum_query_position=192)
        unified = UnifiedFA2(
            model, bridge, adapter, r18, r21, r22, r23,
            local_history=args.local_history, expert_width=args.expert_width,
            expert_count=args.experts, expert_topk=args.expert_topk)
    m21 = attach_historical_group(base_opt, r21, state21, args.r21_lr, "R21")
    m22 = attach_historical_group(base_opt, r22, state22, args.r22_lr, "R22")
    base_opt.add_param_group({"params": list(r23.parameters()), "lr": args.r23_lr,
                              "weight_decay": 1e-4})
    base_opt.add_param_group({"params": list(unified.moe.parameters()),
                              "lr": args.expert_lr, "weight_decay": 1e-4})
    lrs = (args.core_lr, args.subword_lr, args.bridge_lr, args.local_lr,
           args.r18_lr, args.r21_lr, args.r22_lr, args.r23_lr, args.expert_lr)
    if len(base_opt.param_groups) != len(lrs):
        raise RuntimeError("expected one optimizer with NINE inherited/new groups")
    for group, lr in zip(base_opt.param_groups, lrs):
        group["lr"] = lr
    optimizer_ids = [id(p) for group in base_opt.param_groups for p in group["params"]]
    owned = set(optimizer_ids)
    # The accepted PAL checkpoint contains unused legacy heads not owned by
    # its Adam optimizer. Preserve them as frozen weights; never reset or
    # silently add parameters that were not trained in the accepted lineage.
    frozen_legacy = []
    for name, parameter in unified.named_parameters():
        if id(parameter) not in owned:
            parameter.requires_grad_(False)
            frozen_legacy.append(name)
    expected_ids = {id(p) for p in unified.parameters() if p.requires_grad}
    if owned != expected_ids or len(optimizer_ids) != len(owned):
        raise RuntimeError("shared optimizer does not own every TRAINABLE parameter exactly once")
    provenance = {"accepted_sha256": ACCEPTED_SHA,
                  "r21_sha256": h21, "r22_sha256": h22,
                  "r18_optimizer_migration": history,
                  "r21_adam_moments": m21, "r22_adam_moments": m22,
                  "optimizer_groups": len(base_opt.param_groups)}
    return unified, base_opt, comp, r6, provenance


def semantic_scores(unified, data):
    q, keys, positives = data
    pieces = unified.model.memory.piece_emb.weight
    qe = unified.r23.encode(q, True, piece_table=pieces)
    ke = unified.r23.encode(keys, False, piece_table=pieces)
    scores = (qe[:, None, :] * ke).sum(-1)
    return scores * unified.r23.log_temperature.exp().clamp(.1, 40), positives


@torch.no_grad()
def semantic_evaluate(unified, data):
    scores, positives = semantic_scores(unified, data)
    predicted = scores.argmax(-1)
    accuracy = positives.gather(1, predicted[:, None]).float().mean()
    loss = multipositive_semantic_loss(scores, positives)
    return {"accuracy": float(accuracy), "nll": float(loss),
            "examples": len(scores), "chance": 2 / 9}


def frozen_carry(carry):
    def clone(x):
        return None if x is None else x.detach().cpu().clone()
    return {"book_id": carry.book_id, "cursor": carry.cursor,
            "state": clone(carry.state), "previous": clone(carry.previous),
            "last_tag": clone(carry.last_tag),
            "key_history": [clone(x) for x in carry.key_history],
            "value_history": [clone(x) for x in carry.value_history]}


def restore_carry(data):
    state = ContinuousPALState(data["book_id"])
    state.cursor = int(data["cursor"])
    for key in ("state", "previous", "last_tag", "key_history", "value_history"):
        setattr(state, key, copy.deepcopy(data[key]))
    return state


def snapshot(unified, optimizer, step, cursor, carriers):
    return {"model": copy.deepcopy(unified.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "step": step, "cursor": list(cursor),
            "carry": [frozen_carry(x) for x in carriers],
            "torch_rng": torch.get_rng_state().clone(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate()}


def split_chunks(docs, span, max_chunks):
    return [list(contiguous_book_picks(book, span))[:max_chunks] for book in docs]


@torch.no_grad()
def evaluate(unified, books, chunks):
    unified.eval()
    total = base = scrambled = other = other_base = 0.0
    nbytes = nnonspace = ntokens = selected = 0
    donor_total = donor_main = donor_tokens = 0.0
    coverage = {"recent": 0, "middle": 0, "far": 0}
    furthest = {"recent": 0, "middle": 0, "far": 0}
    for bi, (book, pieces) in enumerate(zip(books, chunks)):
        state = ContinuousPALState(book.book_id)
        donor = books[(bi + 1) % len(books)] if len(books) > 1 else None
        for segment in pieces:
            row = unified.forward_chunk(
                book, segment, state, counterfactual=True, donor_book=donor)
            y = row["target"]
            main = F.cross_entropy(row["logits"], y, reduction="none")
            old = F.cross_entropy(row["without_r23"], y, reduction="none")
            wrong = F.cross_entropy(row["scrambled_values"], y, reduction="none")
            nonspace = y.ne(32)
            if row["unrelated_history"] is not None:
                nll_donor = F.cross_entropy(
                    row["unrelated_history"], y, reduction="sum")
                donor_total += float(nll_donor)
                donor_main += float(main.sum())
                donor_tokens += len(y)
            total += float(main.sum())
            base += float(old.sum())
            scrambled += float(wrong.sum())
            other += float(main[nonspace].sum())
            other_base += float(old[nonspace].sum())
            nbytes += row["bytes"]
            nnonspace += int(nonspace.sum())
            ntokens += len(y)
            if row["memory_detail"] is not None:
                detail = row["memory_detail"]
                selected += int(detail["eligible"].sum())
                # The retrieval index is source-only and deterministic.
                idx = row["memory_index"]
                for tier in coverage:
                    coverage[tier] += int(idx["coverage"][tier])
                    furthest[tier] = max(furthest[tier],
                                         int(idx["max_distance"][tier]))
        state.detach()
    if nbytes <= 0 or ntokens <= 0:
        raise ValueError("no validation bytes")
    return {"bpb": total / nbytes / math.log(2),
            "without_r23_bpb": base / nbytes / math.log(2),
            "scrambled_values_bpb": scrambled / nbytes / math.log(2),
            "correct_history_advantage_nats": (scrambled - total) / ntokens,
            "unrelated_book_advantage_nats":
                (donor_total - donor_main) / donor_tokens if donor_tokens else None,
            "unrelated_book_evaluated_tokens": int(donor_tokens),
            "nonspace_nll": other / max(1, nnonspace),
            "without_r23_nonspace_nll": other_base / max(1, nnonspace),
            "tokens": ntokens, "bytes": nbytes,
            "eligible_queries": selected,
            "indexed_candidates": coverage, "max_read_distance": furthest}


def train(args):
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    root = pathlib.Path(args.root).resolve()
    unified, optimizer, comp, r6, parent = initialize(root, args)
    books, manifest = read_books_cached(root / "nanrouter", comp, r6.vocab, 0)
    if not books["train"] or not books["dev"] or not books["test"]:
        raise RuntimeError("train/dev/test book splits missing")
    train_books = books["train"][:args.train_books] if args.train_books else books["train"]
    train_chunks = split_chunks(train_books, args.chunk_span, args.max_chunks)
    dev_books = books["dev"][:args.dev_books]
    test_books = books["test"][:args.test_books]
    dev_chunks = split_chunks(dev_books, args.chunk_span, args.dev_chunks)
    test_chunks = split_chunks(test_books, args.chunk_span, args.test_chunks)
    if any(not x for x in train_chunks) or any(not x for x in dev_chunks + test_chunks):
        raise RuntimeError("empty selected book")
    sem_train = semantic_data(comp.sp, args.semantic_train, args.seed + 2, 20,
                              "train_augmented")
    sem_dev = semantic_data(comp.sp, args.semantic_dev, args.seed + 3, 21,
                            "cross_query")
    sem_test = None  # Test never generated before development selection.
    cursor = [0 for _ in train_chunks]
    carriers = [ContinuousPALState(book.book_id) for book in train_books]
    step0 = 0
    original_dev = None
    if args.resume:
        previous = torch.load(args.resume, map_location="cpu", weights_only=False)
        for key in ("accepted_sha256", "r21_sha256", "r22_sha256"):
            if previous["parent"][key] != parent[key]:
                raise RuntimeError("resume lineage mismatch on " + key)
        same_config = previous["config"]
        for key in ("chunk_span", "local_history", "expert_width", "experts",
                    "expert_topk"):
            if same_config[key] != getattr(args, key):
                raise RuntimeError("resume changed model/stream configuration: " + key)
        if previous["book_ids"] != [x.book_id for x in train_books[:len(previous["book_ids"])]]:
            raise RuntimeError("resume corpus order or identity changed")
        if args.max_chunks < same_config["max_chunks"]:
            raise RuntimeError("cannot shrink available contiguous training history")
        state = previous["selected"]
        unified.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if args.resume_lr_scale != 1:
            for group in optimizer.param_groups:
                group["lr"] *= args.resume_lr_scale
        cursor = list(state["cursor"]) + [0] * (len(train_books) - len(state["cursor"]))
        carriers = [restore_carry(c) for c in state["carry"]]
        carriers.extend(ContinuousPALState(book.book_id) for book in train_books[len(carriers):])
        for i, old_carry in enumerate(carriers[:len(state["cursor"])]):
            if cursor[i] >= len(train_chunks[i]):
                raise RuntimeError("resume cursor beyond available training history")
            if old_carry.cursor != train_chunks[i][cursor[i]][2]:
                raise RuntimeError("resume requires exact next-token stream parity")
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        step0 = state["step"]
        original_dev = previous["initial_dev"]
        if step0 >= args.steps:
            raise ValueError("resume target must exceed selected update step")
    initial = evaluate(unified, dev_books, dev_chunks)
    if original_dev is None:
        original_dev = initial
    initial_sem = semantic_evaluate(unified, sem_dev)
    best = snapshot(unified, optimizer, step0, cursor, carriers)
    best_score = initial["bpb"] + .03 * max(0.0,
                    initial["nonspace_nll"] - original_dev["nonspace_nll"])
    trace = []
    if args.reader_only and args.warmup_target_tokens:
        with torch.no_grad():
            for i, book in enumerate(train_books):
                pieces = train_chunks[i]
                state = carriers[i]
                while cursor[i] < len(pieces) and pieces[cursor[i]][2] < args.warmup_target_tokens:
                    begin, end, t0, t1 = pieces[cursor[i]]
                    ex = torch.as_tensor(book.examples[begin:end].astype(np.int64))[None]
                    lexical = torch.as_tensor(book.context[begin:end].astype(np.float32))[None]
                    tags = torch.as_tensor(book.token_tag[t0:t1].astype(np.int64))[None] - begin
                    tokens = torch.as_tensor(book.tokens[t0:t1].astype(np.int64))[None]
                    hidden = unified.model.core.hidden(
                        ex.reshape(-1, 7), lexical.reshape(-1, 72)
                    ).reshape(1, end-begin, -1)
                    state.forward_chunk(
                        unified.model, unified.bridge, unified.adapter, hidden, tokens, tags,
                        book_id=book.book_id, token_offset=t0, chunk_tag_offset=begin,
                        detach_after=True)
                    cursor[i] += 1
                if cursor[i] == len(pieces):
                    raise RuntimeError("book too short for long-memory training warmup: " + book.book_id)
                if state.cursor != pieces[cursor[i]][2]:
                    raise RuntimeError("causal warmup did not preserve contiguous token cursor")
                print(json.dumps({"phase":"WARMUP", "book":book.book_id,
                                  "token_cursor":state.cursor}), flush=True)
    if args.reader_only:
        for name, parameter in unified.named_parameters():
            if not name.startswith("r23."):
                parameter.requires_grad_(False)
        assert any(p.requires_grad for p in unified.r23.parameters())
        print(json.dumps({"phase":"READER_ONLY", "optimizer_groups":len(optimizer.param_groups),
                          "trainable_tensors":sum(p.requires_grad for p in unified.parameters())}),
              flush=True)

    for step in range(step0 + 1, args.steps + 1):
        unified.train()
        which = (step - 1) % len(train_chunks)
        book = train_books[which]
        segments = train_chunks[which]
        if cursor[which] >= len(segments):
            cursor[which] = 0
            carriers[which].reset(book.book_id)
        carrier = carriers[which]
        terms = []
        n = 0
        positive_counts = {}
        for _ in range(args.unroll):
            if cursor[which] >= len(segments):
                break
            piece = segments[cursor[which]]
            donor = (train_books[(which + 1) % len(train_books)]
                     if args.reader_only and len(train_books) > 1 else None)
            row = unified.forward_chunk(book, piece, carrier,
                                        counterfactual=args.reader_only,
                                        donor_book=donor)
            ce = F.cross_entropy(row["logits"], row["target"])
            extra = row["moe_balance"] * args.balance_weight
            if row["memory_detail"] is not None:
                idx = row["memory_index"]
                span, counts = multipositive_value_loss(
                    row["memory_detail"]["scores"], idx, row["target"])
                extra = extra + args.value_weight * span
                positive_counts = counts
                if args.reader_only and args.margin_weight > 0 and row["unrelated_history"] is not None:
                    positives = (idx["mask"] &
                                 idx["values"][:, :, 0].eq(row["target"][:, None])).any(-1)
                    if bool(positives.any()):
                        targets = row["target"][positives]
                        actual_ce = F.cross_entropy(row["logits"][positives], targets,
                                                    reduction="none")
                        shuffled_ce = F.cross_entropy(row["scrambled_values"][positives],
                                                      targets, reduction="none")
                        donor_ce = F.cross_entropy(row["unrelated_history"][positives],
                                                   targets, reduction="none")
                        margin = args.pairwise_margin
                        # The true target is used ONLY to shape the training loss,
                        # never for memory index admission or inference ranking.
                        extra = extra + args.margin_weight * (
                            F.softplus(6 * (margin + actual_ce - shuffled_ce)).mean() / 6 +
                            F.softplus(6 * (margin + actual_ce - donor_ce)).mean() / 6)
            length = len(row["target"])
            terms.append((ce + extra) * length)
            n += length
            cursor[which] += 1
        if not terms:
            raise RuntimeError("empty truncated-BPTT update")
        loss = torch.stack(terms).sum() / n
        if args.semantic_weight:
            qs, keys, positives = sem_train
            ids = torch.tensor([(step * 11 + args.seed + j * 19) % len(qs)
                                for j in range(args.semantic_batch)])
            score, good = semantic_scores(
                unified, (qs[ids], keys[ids], positives[ids]))
            loss = loss + args.semantic_weight * multipositive_semantic_loss(
                score, good)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(unified.parameters(),
                                                   args.clip))
        if not math.isfinite(grad) or not torch.isfinite(loss):
            raise RuntimeError("nonfinite unified model update")
        optimizer.step()
        carrier.detach()  # TBPTT boundary: one optimizer update, all modules.

        if step % args.eval_every == 0 or step == args.steps:
            dev = evaluate(unified, dev_books, dev_chunks)
            semantic = semantic_evaluate(unified, sem_dev)
            score = dev["bpb"] + .03 * max(0.0,
                            dev["nonspace_nll"] - original_dev["nonspace_nll"])
            if score < best_score - 1e-9:
                best_score = score
                best = snapshot(unified, optimizer, step, cursor, carriers)
            record = {"step": step, "dev": dev, "semantic": semantic,
                      "train_nll": float(loss.detach()),
                      "grad_norm": grad, "value_positive_counts": positive_counts,
                      "best_step": best["step"]}
            trace.append(record)
            print(json.dumps({"phase": "DEV_UNIFIED", "step": step,
                              "bpb": dev["bpb"], "correct_history_advantage_nats":
                              dev["correct_history_advantage_nats"],
                              "best_step": best["step"]}), flush=True)

    unified.load_state_dict(best["model"])
    optimizer.load_state_dict(best["optimizer"])
    selected_dev = evaluate(unified, dev_books, dev_chunks)
    selected_semantic = semantic_evaluate(unified, sem_dev)
    # A better BPB alone cannot promote a frequency-only memory path.
    dev_gates = {
        "bpb_improved": selected_dev["bpb"] < original_dev["bpb"] - 1e-6,
        "nonspace_not_worse": selected_dev["nonspace_nll"] <=
                              original_dev["nonspace_nll"] + 1e-6,
        "correct_values_better_than_shuffled":
            selected_dev["correct_history_advantage_nats"] > 1e-6,
        "long_range_reached": selected_dev["max_read_distance"]["far"] >= 8192,
        "correct_book_better_than_unrelated":
            selected_dev["unrelated_book_evaluated_tokens"] > 0 and
            selected_dev["unrelated_book_advantage_nats"] > 1e-6,
        "semantic_not_worse": selected_semantic["nll"] <=
                              initial_sem["nll"] + 1e-4,
    }
    # Checkpoint stores ONE model and ONE AdamW optimizer, with exact stream
    # cursors and RNG for genuine continuation; no separate-arm weights.
    checkpoint = {"format": "FA2_R24_UNIFIED_CANDIDATE_V1",
                  "parent": parent, "config": vars(args).copy(),
                  "book_ids": [x.book_id for x in train_books],
                  "selected": best, "initial_dev": original_dev,
                  "selected_dev": selected_dev, "gates": dev_gates,
                  "accepted_model_promoted": False}
    target = root / "reports/fa2_unified_candidate.pth"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    torch.save(checkpoint, tmp)
    tmp.replace(target)
    # Historical test accessed ONLY after dev-selected weights were saved.
    sem_test = semantic_data(comp.sp, args.semantic_dev, args.seed + 4, 22,
                             "paraphrase")
    historical = {"lm": evaluate(unified, test_books, test_chunks),
                  "synthetic_paraphrase": semantic_evaluate(unified, sem_test)}
    source_unchanged = (
        pal.sha(root / "checkpoint/r12_active.pal12") == parent["accepted_sha256"]
        and pal.sha(root / "reports/fa2_2_r21_gate_candidate.pth") ==
            parent["r21_sha256"]
        and pal.sha(root / "reports/r22_multiscale_candidate.pth") ==
            parent["r22_sha256"])
    if not source_unchanged:
        raise RuntimeError("inherited immutable checkpoint files modified")
    report = {"round": "FA2 R24 single unified mirror-MoE PAL",
              "parent": parent, "single_optimizer": True,
              "optimizer_groups": len(optimizer.param_groups),
              "selected_step": best["step"], "attempted_step": args.steps,
              "initial_dev": original_dev, "selected_dev": selected_dev,
              "initial_semantic": initial_sem,
              "selected_semantic": selected_semantic,
              "promotion_gates": dev_gates,
              "historical_test_nonblind": historical,
              "train_books": [b.book_id for b in train_books],
              "dev_books": [b.book_id for b in dev_books],
              "test_books": [b.book_id for b in test_books],
              "trace": trace, "elapsed_seconds": time.monotonic() - started,
              "parent_unchanged": source_unchanged, "promoted": False,
              "limitations": [
                "Only contiguous unroll-chunk gradients: forward state can persist longer",
                "32K document-addressed retrieval is not 32K recurrent BPTT",
                "The historical test books have been consulted in earlier PAL rounds",
                "Shuffled-value control is not a separate-book donor control",
                "No frontier-level BPB, reasoning, or semantic transfer has been established",
                "Candidates are not promoted automatically even if dev gates pass"]}
    path = root / "reports/fa2_unified.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"phase": "FINAL_UNIFIED", "selected_step": best["step"],
                      "dev_bpb": selected_dev["bpb"], "historical_test": historical,
                      "promotion_gates": dev_gates, "promoted": False}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--resume", default="")
    p.add_argument("--resume-lr-scale", type=float, default=1.0,
                   help="multiplicatively reduce restored nine-group LR without resetting moments")
    p.add_argument("--reader-only", action="store_true",
                   help="one-model phase: freeze inherited trunk and train existing R23 reader only")
    p.add_argument("--warmup-target-tokens", type=int, default=0,
                   help="causally advance live book state to this token before reader training")
    p.add_argument("--pairwise-margin", type=float, default=0.02)
    p.add_argument("--margin-weight", type=float, default=0.25)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--train-books", type=int, default=0,
                   help="0 uses all available training books; positive N limits a CPU reproduction")
    p.add_argument("--seed", type=int, default=24024)
    p.add_argument("--chunk-span", type=int, default=12)
    p.add_argument("--unroll", type=int, default=2)
    p.add_argument("--local-history", type=int, default=512)
    p.add_argument("--max-chunks", type=int, default=256)
    p.add_argument("--dev-chunks", type=int, default=48)
    p.add_argument("--test-chunks", type=int, default=48)
    p.add_argument("--dev-books", type=int, default=2)
    p.add_argument("--test-books", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--semantic-train", type=int, default=128)
    p.add_argument("--semantic-dev", type=int, default=32)
    p.add_argument("--semantic-batch", type=int, default=8)
    p.add_argument("--value-weight", type=float, default=.15)
    p.add_argument("--semantic-weight", type=float, default=.04)
    p.add_argument("--balance-weight", type=float, default=.002)
    p.add_argument("--clip", type=float, default=2.)
    p.add_argument("--expert-width", type=int, default=384)
    p.add_argument("--experts", type=int, default=4)
    p.add_argument("--expert-topk", type=int, default=2)
    p.add_argument("--core-lr", type=float, default=2e-7)
    p.add_argument("--subword-lr", type=float, default=2e-5)
    p.add_argument("--bridge-lr", type=float, default=1e-4)
    p.add_argument("--local-lr", type=float, default=1e-4)
    p.add_argument("--r18-lr", type=float, default=1e-4)
    p.add_argument("--r21-lr", type=float, default=5e-5)
    p.add_argument("--r22-lr", type=float, default=5e-5)
    p.add_argument("--r23-lr", type=float, default=3e-4)
    p.add_argument("--expert-lr", type=float, default=2e-4)
    args = p.parse_args()
    if not (args.steps >= 1 and args.unroll >= 1 and
            args.chunk_span >= 1 and args.max_chunks >= args.unroll and
            args.eval_every >= 1 and args.dev_books >= 1 and
            args.test_books >= 1 and args.expert_width >= 128 and
            args.train_books >= 0 and 0 < args.resume_lr_scale <= 1 and
            args.warmup_target_tokens >= 0 and args.pairwise_margin >= 0 and
            args.margin_weight >= 0):
        p.error("invalid CPU training configuration")
    train(args)
