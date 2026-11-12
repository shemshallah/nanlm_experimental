#!/usr/bin/env python3
"""FA2 R23-S: source-verified contiguous-book CPU experiment.

This driver carries the frozen inherited PAL recurrence across *consecutive*
book chunks rather than restarting it at every prediction window, and trains
the NEW R23 contextual reader on the resulting honest causal logits. The R18
decoder, all inherited optimizer moments, frozen R21 gate and frozen R22 adapter
remain unchanged. Only R23 reader parameters have gradients.

This is persistent-state FORWARD training of the 32K-addressed reader, not
32K backpropagation through the frozen recurrent backbone. Do not promote this
candidate automatically or compare directly against the old reset-window BPB.
"""
import argparse
import copy
import json
import math
import pathlib
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import r14_slim_base as pal
from r13_corpus_cache import read_books_cached
from r18_long_memory import R18LongMemory, build_copy_queries, apply_copy_boost
from r21_relevance_novelty import RelevanceNoveltyGate, causal_proposal
from r22_multiscale_memory import R22Residual, hierarchical_proposals
from r23_context64 import ContextualSpanMemory, build_index, multipositive_value_loss
from r23_stream_carry import ContinuousPALState, contiguous_book_picks

ACCEPTED_SHA = "dd040106568786e7fbad2a880f4010635b626b51c3697e2ab08712905a2de91f"


def cached_book(split, book, model, bridge, adapter, r18, r21, r22,
                cache_dir, chunk_span, max_chunks):
    """One strict sequential pass from the beginning of one verified book.

    Frozen logits are generated with recurrent history carried across every
    chunk. Candidate keys/values refer only to the same book's preceding
    *completed* chunks. The causal attention cache has bounded length.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    carry = ContinuousPALState(book.book_id)
    manifest = []
    with torch.no_grad():
        for i, (begin, end, t0, t1) in enumerate(contiguous_book_picks(book, chunk_span)):
            if i >= max_chunks:
                break
            words = torch.as_tensor(book.examples[begin:end].astype(np.int64))[None]
            lex = torch.as_tensor(book.context[begin:end].astype(np.float32))[None]
            tags = torch.as_tensor(book.token_tag[t0:t1].astype(np.int64))[None] - begin
            tokens = torch.as_tensor(book.tokens[t0:t1].astype(np.int64))[None]
            if (tags.min() < 0 or tags.max() >= end - begin):
                raise ValueError("token tags not aligned to word chunks")
            hidden = model.core.hidden(words.reshape(-1, 7),
                                        lex.reshape(-1, 72)).reshape(1, end-begin, -1)
            streamed = carry.forward_chunk(
                model, bridge, adapter, hidden, tokens, tags,
                book_id=book.book_id, token_offset=t0, chunk_tag_offset=begin,
                detach_after=True)[0]
            # Inherited R18 memory and subsequent R21/R22 gates are frozen.
            q18 = build_copy_queries(tokens[0], mem=r18.max_memory, gap=r18.min_gap)
            inherited = apply_copy_boost(r18, streamed, q18)
            delta21, f21, _ = causal_proposal(r18, inherited, tokens[0])
            post21, _ = r21(inherited, delta21, f21)
            delta22, f22, _ = hierarchical_proposals(
                r18, post21, book.tokens, t0)
            post22, _ = r22(post21, delta22, f22)
            # First chunk warms up recurrence; remaining chunks have complete
            # historical key/value spans before their current targets.
            if t0 >= 65:
                memory = build_index(book.tokens, t0, t1-t0)
                memory["query_position_offset"] = t0
                file = cache_dir / ("%03d.pth" % i)
                byte_start = int(book.byte_starts[begin])
                byte_end = (int(book.byte_starts[end]) if end < len(book.byte_starts)
                            else book.raw_len)
                torch.save({"base": post22, "y": tokens[0].clone(),
                            "bytes": byte_end-byte_start,
                            "index": memory, "book": book.book_id,
                            "chunk_index": i, "token_offset": t0}, file)
                manifest.append(str(file))
    if not manifest:
        raise ValueError("no eligible chunks: increase --max-chunks")
    return manifest, carry.snapshot()


def read_item(path):
    return torch.load(path, map_location="cpu", weights_only=False)


@torch.no_grad()
def evaluate(net, paths):
    net.eval()
    total = prior = other = prior_other = 0.0
    all_bytes = nonspace = tokens = 0
    covered = 0
    for path in paths:
        item = read_item(path)
        new, detail = net(item["base"], item["index"])
        tgt = item["y"]
        oldce = F.cross_entropy(item["base"], tgt, reduction="none")
        newce = F.cross_entropy(new, tgt, reduction="none")
        mask = tgt != 32
        total += float(newce.sum())
        prior += float(oldce.sum())
        other += float(newce[mask].sum())
        prior_other += float(oldce[mask].sum())
        all_bytes += item["bytes"]
        nonspace += int(mask.sum())
        tokens += len(tgt)
        covered += int(detail["eligible"].sum())
    return {"bpb": total / all_bytes / math.log(2),
            "frozen_r22_stream_bpb": prior / all_bytes / math.log(2),
            "nonspace_nll": other / max(1,nonspace),
            "frozen_r22_stream_nonspace_nll": prior_other / max(1,nonspace),
            "tokens": tokens, "bytes": all_bytes,
            "eligible_queries": covered, "chunks": len(paths)}


def run(args):
    root = pathlib.Path(args.root).resolve()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    start = time.monotonic()
    checkpoint = root / "checkpoint/r12_active.pal12"
    parent = pal.sha(checkpoint)
    if parent != ACCEPTED_SHA:
        raise RuntimeError("R18 accepted parent hash mismatch")
    r21_file = root / "reports/fa2_2_r21_gate_candidate.pth"
    r22_file = root / "reports/r22_multiscale_candidate.pth"
    r21_sha, r22_sha = pal.sha(r21_file), pal.sha(r22_file)
    state21 = torch.load(r21_file, map_location="cpu", weights_only=False)
    state22 = torch.load(r22_file, map_location="cpu", weights_only=False)
    assert state21["parent_checkpoint_sha256"] == parent
    assert state22["parent_checkpoint_sha256"] == parent
    assert state22["frozen_r21_gate_sha256"] == r21_sha
    assert not state22["accepted_model_promoted"]
    accepted, r6, comp, model, bridge, adapter, opt, _ = pal.load_for_arm(root, checkpoint, True)
    assert "r18_memory" in accepted and len(opt.param_groups) == 4
    r18 = R18LongMemory(model.memory.piece_emb.weight)
    r18.load_state_dict(accepted["r18_memory"])
    r21 = RelevanceNoveltyGate()
    r21.load_state_dict(state21["gate"])
    r22 = R22Residual()
    r22.load_state_dict(state22["model"])
    for frozen in (model, bridge, adapter, r18, r21, r22):
        frozen.eval().requires_grad_(False)
    docs, _ = read_books_cached(root / "nanrouter", comp, r6.vocab, 0)
    if not docs["train"] or not docs["dev"] or not docs["test"]:
        raise RuntimeError("empty NaNROUTER split")
    # Rebuild from source verified book, always starting from its first token.
    # This is not random window shuffling disguised as persistent state.
    cache = root / ".cache" / ("r23_stream_seed%d_span%d" %
                                (args.seed, args.chunk_span))
    train, train_meta = cached_book(
        "train", docs["train"][0], model, bridge, adapter, r18, r21, r22,
        cache / "train", args.chunk_span, args.max_chunks)
    dev, dev_meta = cached_book(
        "dev", docs["dev"][0], model, bridge, adapter, r18, r21, r22,
        cache / "dev", args.chunk_span, args.max_chunks)
    model23 = ContextualSpanMemory(model.memory.piece_emb.weight,
                                   minimum_query_position=192)
    optimizer = torch.optim.AdamW(model23.parameters(), lr=args.lr, weight_decay=1e-4)
    initial = evaluate(model23, dev)
    initial_parent = initial["frozen_r22_stream_bpb"]
    best_model = copy.deepcopy(model23.state_dict())
    best_opt = copy.deepcopy(optimizer.state_dict())
    best_step = 0
    best_obj = initial["bpb"] + 0.05 * max(0.0, initial["nonspace_nll"] -
                                               initial["frozen_r22_stream_nonspace_nll"])
    trace = []
    for step in range(1, args.steps+1):
        model23.train()
        # Caches are in document order; deterministic stride traversal
        # does not turn off or reset the forward-state history used to
        # generate these logits.
        item = read_item(train[(step-1) % len(train)])
        new, detail = model23(item["base"], item["index"])
        lm = F.cross_entropy(new, item["y"])
        value, positives = multipositive_value_loss(
            detail["scores"], item["index"], item["y"])
        loss = lm + args.value_weight * value
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model23.parameters(), 2.0)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model23, dev)
            objective = metrics["bpb"] + 0.05 * max(0.0, metrics["nonspace_nll"] -
                                                        metrics["frozen_r22_stream_nonspace_nll"])
            if objective < best_obj - 1e-9:
                best_obj = objective
                best_model = copy.deepcopy(model23.state_dict())
                best_opt = copy.deepcopy(optimizer.state_dict())
                best_step = step
            trace.append({"step": step, "dev": metrics,
                          "train_loss": float(loss.detach()),
                          "span_positives": positives})
            print(json.dumps({"phase":"DEV_STREAM", "step":step,
                              "bpb":metrics["bpb"],"r22_stream_bpb":initial_parent,
                              "best":best_step}), flush=True)
    model23.load_state_dict(best_model)
    optimizer.load_state_dict(best_opt)
    selected = evaluate(model23, dev)
    assert pal.sha(checkpoint) == parent
    assert pal.sha(r21_file) == r21_sha and pal.sha(r22_file) == r22_sha
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    state = {"parent_sha":parent, "r21_sha":r21_sha, "r22_sha":r22_sha,
             "source":"R23 contiguous PAL frozen-state forward",
             "model":best_model, "optimizer":best_opt, "best_step":best_step,
             "attempted_steps":args.steps, "promoted":False}
    torch.save(state, reports / "r23_stream_candidate.pth")
    # Historical test generated only after selecting the best development
    # checkpoint, from a completely separate book with fresh recurrent state.
    test, test_meta = cached_book(
        "test", docs["test"][0], model, bridge, adapter, r18, r21, r22,
        cache / "test", args.chunk_span, args.max_chunks)
    report = {"round":"FA2.4 R23-S continuous-backbone-forward candidate",
              "checkpoint":parent, "r21_sha":r21_sha, "r22_sha":r22_sha,
              "train_book":docs["train"][0].book_id,
              "dev_book":docs["dev"][0].book_id,
              "test_book":docs["test"][0].book_id,
              "train_carry":train_meta,"dev_carry":dev_meta,"test_carry":test_meta,
              "steps":args.steps,"best_step":best_step,"initial_dev":initial,
              "selected_dev":selected,"historical_test":evaluate(model23,test),
              "trace":trace, "seconds":time.monotonic()-start,
              "promoted":False,"parent_unchanged":True,
              "limitations":["Backbone weights and optimizer frozen, no 32K truncated/full BPTT",
                             "Cached causal logits produced with persistent frozen state",
                             "History uses first book in each split only; test is previously consulted",
                             "The stream path and original reset-window BPB are NOT comparable directly"]}
    (reports / "r23_stream.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"phase":"FINAL_STREAM",
                      "selected_dev":selected,
                      "historical_test":report["historical_test"],
                      "promoted":False}), flush=True)


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--chunk-span",type=int,default=32)
    ap.add_argument("--max-chunks",type=int,default=32)
    ap.add_argument("--steps",type=int,default=100)
    ap.add_argument("--eval-every",type=int,default=20)
    ap.add_argument("--value-weight",type=float,default=0.25)
    ap.add_argument("--lr",type=float,default=0.0005)
    ap.add_argument("--seed",type=int,default=23091)
    ap.add_argument("--threads",type=int,default=2)
    run(ap.parse_args())
