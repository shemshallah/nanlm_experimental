#!/usr/bin/env python3
"""FA2 R23: CPU experiments; R18/R21/R22 frozen, R23 candidate never promoted.

Arms --mode context|value|semantic|joint share exactly the same initialization.
Language windows are real, source-verified NaNROUTER books from R19 full release.
4..64-token Q/K and 1..8-token stored values are trained end-to-end over a
persistent 32K memory index; the R18 recurrent decoder still has short BPTT.
Dev selects. Historical test is consulted only after selection, and is NOT blind.
"""
import argparse
import copy
import hashlib
import json
import math
import pathlib
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import r14_slim_base as pal
from r11_raw import windows
from r13_corpus_cache import read_books_cached
from r18_joint_continue import integrated_logits
from r18_long_memory import R18LongMemory
from r19_semantic_frontier import RELATIONS, make_examples
from r21_relevance_novelty import RelevanceNoveltyGate
from r22_multiscale_memory import R22Residual
from r22_cpu_longmemory import build_cache, load_item
from r23_context64 import (ContextualSpanMemory, build_index,
                           multipositive_semantic_loss,
                           multipositive_value_loss)


def encode_sentence(sp, s):
    ids = sp.encode(s, out_type=int)[-64:]
    return [-1] * (64 - len(ids)) + list(ids)


def semantic_data(sp, count, seed, split, mode):
    """Two same-meaning positives, seven adversarial negatives per query."""
    rows = make_examples(count, seed, mode=mode, split=split)
    qs, ks, masks = [], [], []
    for row in rows:
        rel = row["relation"]
        alt = RELATIONS[rel][0][(2 if mode != "train_augmented" else 1)]
        paraphrase = alt.format(s=row["subject"], v=row["value"])
        facts = row["facts"] + [paraphrase]
        mask = [i == row["label"] for i in range(8)] + [True]
        qs.append(encode_sentence(sp, row["query"]))
        ks.append([encode_sentence(sp, x) for x in facts])
        masks.append(mask)
    return (torch.tensor(qs, dtype=torch.long),
            torch.tensor(ks, dtype=torch.long),
            torch.tensor(masks, dtype=torch.bool))


def semantic_scores(net, q, k):
    qe = net.encode(q, True)
    ke = net.encode(k, False)
    return ((qe[:, None, :] * ke).sum(-1) *
            net.log_temperature.exp().clamp(0.1, 40))


@torch.no_grad()
def evaluate_semantic(net, data, batch=16):
    if data is None:
        return None
    q, k, positives = data
    correct = 0
    loss = 0.0
    for start in range(0, len(q), batch):
        stop = min(len(q), start + batch)
        scores = semantic_scores(net, q[start:stop], k[start:stop])
        pred = scores.argmax(-1)
        correct += int(positives[start:stop].gather(1, pred[:, None]).sum())
        loss += float(multipositive_semantic_loss(scores, positives[start:stop])) * (stop - start)
    return {"accuracy": correct / max(1, len(q)), "nll": loss / max(1, len(q)),
            "chance": 2 / 9, "examples": len(q)}


def build_examples(split, docs, choices, span, model, bridge, adapter,
                   memory, r21, r22, path):
    """Frozen local logits + live trainable contextual reader, disk-backed.

    Uses already validated causal R22 feature machinery. No logits from a
    longer future document window are ever part of the index.
    """
    path.mkdir(parents=True, exist_ok=True)
    outputs = []
    frozen_cache = build_cache(split, docs, choices, span, model, bridge,
                               adapter, memory, r21, cache_dir=path / "frozen")
    with torch.no_grad():
        for i, choice in enumerate(choices):
            dest = path / ("r23_%04d.pth" % i)
            book = docs[choice[0]]
            c = load_item(frozen_cache[i])
            start = int(c["token_offset"])
            base, _ = r22(c["r21"], c["proposals"], c["features"])
            index = build_index(book.tokens, start, len(c["y"]))
            data = {"base": base.detach().clone(), "y": c["y"],
                    "bytes": int(c["bytes"]), "index": index,
                    "book": book.book_id, "token_offset": start}
            torch.save(data, dest)
            outputs.append(dest)
    return outputs


def load_example(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def prediction_metrics(net, paths):
    net.eval()
    total = parent = other = other_parent = 0.0
    nbytes = nonspace = correct = count = 0
    with torch.no_grad():
        for path in paths:
            item = load_example(path)
            out, _ = net(item["base"], item["index"])
            ce = F.cross_entropy(out, item["y"], reduction="none")
            parent_ce = F.cross_entropy(item["base"], item["y"], reduction="none")
            mask = item["y"] != 32
            total += float(ce.sum())
            parent += float(parent_ce.sum())
            other += float(ce[mask].sum())
            other_parent += float(parent_ce[mask].sum())
            correct += int(((out.argmax(-1) == item["y"]) & mask).sum())
            nbytes += item["bytes"]
            nonspace += int(mask.sum())
            count += len(item["y"])
    return {"bpb": total / nbytes / math.log(2),
            "r22_bpb": parent / nbytes / math.log(2),
            "nonspace_nll": other / max(1, nonspace),
            "r22_nonspace_nll": other_parent / max(1, nonspace),
            "nonspace_top1": correct / max(1, nonspace),
            "tokens": count, "bytes": nbytes, "windows": len(paths)}


def train(args):
    t0 = time.monotonic()
    root = pathlib.Path(args.root).resolve()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    original = root / "checkpoint/r12_active.pal12"
    parent_sha = pal.sha(original)
    if parent_sha != "dd040106568786e7fbad2a880f4010635b626b51c3697e2ab08712905a2de91f":
        raise RuntimeError("wrong FA2 inherited R18 checkpoint")
    r21_file = root / "reports/fa2_2_r21_gate_candidate.pth"
    r22_file = root / "reports/r22_multiscale_candidate.pth"
    r21_sha = pal.sha(r21_file)
    r22_sha = pal.sha(r22_file)
    r21_state = torch.load(r21_file, map_location="cpu", weights_only=False)
    r22_state = torch.load(r22_file, map_location="cpu", weights_only=False)
    assert r21_state["parent_checkpoint_sha256"] == parent_sha
    assert r22_state["parent_checkpoint_sha256"] == parent_sha
    assert r22_state["frozen_r21_gate_sha256"] == r21_sha
    assert not r22_state["accepted_model_promoted"]
    r21 = RelevanceNoveltyGate()
    r21.load_state_dict(r21_state["gate"])
    r22 = R22Residual()
    r22.load_state_dict(r22_state["model"])
    saved, r6, comp, model, bridge, adapter, base_opt, _ = pal.load_for_arm(root, original, True)
    assert "r18_memory" in saved and len(base_opt.param_groups) == 4
    memory = R18LongMemory(model.memory.piece_emb.weight)
    memory.load_state_dict(saved["r18_memory"])
    for mod in (model, bridge, adapter, memory, r21, r22):
        mod.eval().requires_grad_(False)
    candidate = ContextualSpanMemory(model.memory.piece_emb.weight,
                                     minimum_query_position=args.min_query)
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=args.lr, weight_decay=1e-4)
    # Resume only an experimental R23 state with unchanged full inherited lineage.
    current = 0
    if args.resume_state:
        prior = torch.load(args.resume_state, map_location="cpu", weights_only=False)
        assert prior["parent_sha"] == parent_sha and prior["r21_sha"] == r21_sha
        assert prior["r22_sha"] == r22_sha and prior["mode"] == args.mode
        assert not prior["promoted"]
        candidate.load_state_dict(prior["candidate"])
        optimizer.load_state_dict(prior["optimizer"])
        current = prior["steps"]
        assert current < args.steps
    docs, _ = read_books_cached(root / "nanrouter", comp, r6.vocab, 0)
    # Identical windows and seeds in all experimental arms.
    win = windows(docs, args.span,
                  max(args.train_windows, args.dev_windows, args.test_windows),
                  args.max_tokens)
    key = "r23_span%d_seed%d" % (args.span, args.seed)
    cache = root / ".cache" / key
    train_paths = build_examples("train", docs["train"], win["train"][:args.train_windows],
                                 args.span, model, bridge, adapter, memory, r21, r22,
                                 cache / "train")
    dev_paths = build_examples("dev", docs["dev"], win["dev"][:args.dev_windows],
                               args.span, model, bridge, adapter, memory, r21, r22,
                               cache / "dev")
    sp = comp.sp
    sem_train = semantic_data(sp, args.semantic_train, args.seed + 2, 20, "train_augmented")
    sem_dev = semantic_data(sp, args.semantic_dev, args.seed + 3, 21, "cross_query")
    initial_lm = prediction_metrics(candidate, dev_paths)
    initial_sem = evaluate_semantic(candidate, sem_dev)
    best = copy.deepcopy(candidate.state_dict())
    best_opt = copy.deepcopy(optimizer.state_dict())
    best_step = current
    baseline = initial_lm["r22_bpb"]
    bestkey = float("inf")
    trace = []
    for step in range(current + 1, args.steps + 1):
        candidate.train()
        terms = []
        stats = {}
        if args.mode in ("context", "value", "joint"):
            pick = train_paths[(step * 7 + args.seed) % len(train_paths)]
            item = load_example(pick)
            output, det = candidate(item["base"], item["index"])
            lm_loss = F.cross_entropy(output, item["y"])
            terms.append(lm_loss)
            stats["lm_nll"] = float(lm_loss.detach())
            if args.mode in ("value", "joint"):
                value_loss, positive_counts = multipositive_value_loss(
                    det["scores"], item["index"], item["y"])
                terms.append(args.value_weight * value_loss)
                stats["value_loss"] = float(value_loss.detach())
                stats["span_positives"] = positive_counts
        if args.mode in ("semantic", "joint"):
            q, k, pos = sem_train
            ids = torch.tensor([(step * 11 + args.seed + j * 19) % len(q)
                                for j in range(args.semantic_batch)], dtype=torch.long)
            score = semantic_scores(candidate, q[ids], k[ids])
            sloss = multipositive_semantic_loss(score, pos[ids])
            terms.append(args.semantic_weight * sloss)
            stats["semantic_loss"] = float(sloss.detach())
        loss = sum(terms)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), 2.0)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            lm = prediction_metrics(candidate, dev_paths)
            sem = evaluate_semantic(candidate, sem_dev)
            # Context and value modes are chosen by development NLL, with a
            # non-space penalty. Semantic-only mode is evaluated independently.
            keyval = (sem["nll"] if args.mode == "semantic" else
                      lm["bpb"] + 0.05 * max(0.0, lm["nonspace_nll"] -
                                                 lm["r22_nonspace_nll"]))
            if keyval < bestkey - 1e-8:
                bestkey = keyval
                best = copy.deepcopy(candidate.state_dict())
                best_opt = copy.deepcopy(optimizer.state_dict())
                best_step = step
            trace.append({"step": step, "objective": keyval, "dev_lm": lm,
                          "dev_semantic": sem, "train": stats})
            print(json.dumps({"phase": "DEV", "mode": args.mode,
                              "step": step, "best": best_step,
                              "bpb": lm["bpb"], "r22_bpb": baseline,
                              "paraphrase_accuracy": sem["accuracy"]}), flush=True)
    candidate.load_state_dict(best)
    optimizer.load_state_dict(best_opt)
    candidate.eval()
    chosen_lm = prediction_metrics(candidate, dev_paths)
    chosen_sem = evaluate_semantic(candidate, sem_dev)
    # Save pretest; test labels are not available to the parameter selector.
    dst = root / "reports" / ("r23_%s_candidate.pth" % args.mode)
    torch.save({"parent_sha": parent_sha, "r21_sha": r21_sha,
                "r22_sha": r22_sha, "mode": args.mode,
                "candidate": best, "optimizer": best_opt,
                "steps": best_step, "attempted_steps": args.steps,
                "promoted": False}, dst)
    # Historical PAL test: read only after selecting a frozen dev checkpoint.
    test_paths = build_examples("test", docs["test"], win["test"][:args.test_windows],
                                args.span, model, bridge, adapter, memory, r21, r22,
                                cache / "test")
    sem_test = semantic_data(sp, args.semantic_dev, args.seed + 4, 22, "paraphrase")
    result = {"round": "FA2 R23 64-token contextual / multi-value / contrastive",
              "mode": args.mode, "parent_sha": parent_sha,
              "r21_sha": r21_sha, "r22_sha": r22_sha,
              "train_book_ids": sorted({load_example(x)["book"] for x in train_paths}),
              "dev_book_ids": sorted({load_example(x)["book"] for x in dev_paths}),
              "test_book_ids": sorted({load_example(x)["book"] for x in test_paths}),
              "initial_dev": {"lm": initial_lm, "semantic": initial_sem},
              "selected_dev": {"lm": chosen_lm, "semantic": chosen_sem},
              "historical_test": {"lm": prediction_metrics(candidate, test_paths),
                                  "paraphrase": evaluate_semantic(candidate, sem_test)},
              "best_step": best_step, "started_from_step": current,
              "steps_attempted": args.steps, "trace": trace,
              "parameters_new": sum(p.numel() for p in candidate.parameters()),
              "promoted": False, "original_checkpoint_unchanged": pal.sha(original) == parent_sha,
              "r21_unchanged": pal.sha(r21_file) == r21_sha,
              "r22_unchanged": pal.sha(r22_file) == r22_sha,
              "runtime_seconds": time.monotonic() - t0,
              "limitations": [
                  "Older historical test books were seen in previous PAL rounds",
                  "32K addressable reader receives gradients; R18 backbone still short BPTT",
                  "Paraphrase benchmark uses generated facts, not real multi-hop QA",
                  "One CPU candidate per arm, no validated frontier performance"]}
    report = root / "reports" / ("r23_%s.json" % args.mode)
    report.write_text(json.dumps(result, indent=2) + "\n")
    assert result["original_checkpoint_unchanged"] and result["r21_unchanged"] and result["r22_unchanged"]
    print(json.dumps({"phase": "FINAL", "mode": args.mode,
                      "dev": chosen_lm, "historical": result["historical_test"],
                      "best_step": best_step, "promoted": False}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--mode", choices=("context", "value", "semantic", "joint"), required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--train-windows", type=int, default=24)
    p.add_argument("--dev-windows", type=int, default=8)
    p.add_argument("--test-windows", type=int, default=8)
    p.add_argument("--span", type=int, default=120)
    p.add_argument("--max-tokens", type=int, default=800)
    p.add_argument("--min-query", type=int, default=192)
    p.add_argument("--semantic-train", type=int, default=256)
    p.add_argument("--semantic-dev", type=int, default=64)
    p.add_argument("--semantic-batch", type=int, default=8)
    p.add_argument("--semantic-weight", type=float, default=0.25)
    p.add_argument("--value-weight", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=0.0005)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=23064)
    p.add_argument("--resume-state", default="")
    train(p.parse_args())
