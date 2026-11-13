#!/usr/bin/env python3
"""FA2 R23 non-oracle regression: 32K causal context, value spans, gradients."""
import json
import sys

import torch
from torch.nn import functional as F
from r23_context64 import (ContextualSpanMemory, build_index,
                           multipositive_semantic_loss,
                           multipositive_value_loss, TIERS, HORIZONS)


def run():
    torch.set_num_threads(2)
    torch.manual_seed(23064)
    start, length = 33000, 40
    # A repeated pattern gives honest historical matches at all distances.
    tok = torch.tensor([(17 * t + t // 101) % 197 + 1
                        for t in range(start + length + 4)], dtype=torch.long)
    # Force matching contexts with several different stored continuations.
    tok[300:368] = tok[17000:17068]
    tok[26000:26068] = tok[17000:17068]
    emb = torch.randn(256, 24)
    model = ContextualSpanMemory(emb, projection=32,
                                 minimum_query_position=0, topk=6)
    mem = build_index(tok, start, length, per_band=12)
    y = tok[start:start + length].clone()
    baseline = torch.randn(length, 256)
    original = baseline.clone()
    out, detail = model(baseline, mem)
    check = {}
    check["zero_residual_bit_exact"] = torch.equal(out, original)
    check["all_three_tiers_indexed"] = all(mem["coverage"][x] > 0 for x in TIERS)
    check["far_reads_beyond_8192"] = mem["max_distance"]["far"] >= 8192
    good = mem["mask"]
    source = mem["positions"][good]
    query_positions = (start + torch.arange(length)[:, None]).expand_as(good)[good]
    check["source_precedes_query_by_65"] = bool(((query_positions - source) >= 65).all())
    check["all_value_spans_before_window"] = bool((source + 8 <= start).all())
    check["all_read_distances_bounded"] = bool(((query_positions - source) <= 32768).all())
    check["64_token_query_context"] = tuple(mem["query"].shape) == (length, 64)
    check["8_token_source_values"] = mem["values"].shape[-1] == 8
    # Edit current/future targets without changing earlier read indices.
    edited = tok.clone()
    edited[start + 15:] = torch.randint(1, 198, (len(tok) - start - 15,))
    altered = build_index(edited, start, length, per_band=12)
    check["future_suffix_isolation"] = all(
        torch.equal(mem[name][:15], altered[name][:15])
        for name in ("query", "keys", "values", "positions", "mask", "distance", "tier"))
    with torch.no_grad():
        model.amplitude.fill_(0.35)
    out, detail = model(baseline, mem)
    check["enabled_adapter_changes_logits"] = not torch.equal(out, original)
    ce = F.cross_entropy(out, y)
    span_loss, counts = multipositive_value_loss(detail["scores"], mem, y)
    check["finite_multi_horizon_contrastive"] = bool(torch.isfinite(span_loss))
    check["span_supervision_uses_multiple_horizons"] = (
        set(counts) == {str(x) for x in HORIZONS} and counts["1"] > 0)
    # Supervised paraphrase-like ranking shares the SAME 4..64 encoder;
    # multiple positives and lexical distractors are scored jointly.
    queries = mem["query"][:4]
    facts = mem["keys"][:4, :6]
    sem = (model.encode(queries, True)[:, None, :] *
           model.encode(facts, False)).sum(-1) * model.log_temperature.exp()
    positives = torch.zeros_like(sem, dtype=torch.bool)
    positives[:, 0:2] = True
    semantic = multipositive_semantic_loss(sem, positives)
    check["finite_semantic_multipositive"] = bool(torch.isfinite(semantic))
    (ce + 0.2 * span_loss + 0.2 * semantic).backward()
    check["context_query_gradient"] = bool(model.query_net[-1].weight.grad.abs().sum() > 0)
    check["context_key_gradient"] = bool(model.key_net[-1].weight.grad.abs().sum() > 0)
    check["gate_amplitude_gradient"] = bool(model.amplitude.grad.abs() > 0)
    check["all_gradients_finite"] = all(
        p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    try:
        multipositive_semantic_loss(torch.zeros((2, 3)), torch.zeros((2, 3), dtype=torch.bool))
        check["reject_missing_positive_semantic"] = False
    except ValueError:
        check["reject_missing_positive_semantic"] = True
    assert all(check.values()), json.dumps(check, indent=2)
    print(json.dumps({"checks": check, "passed": len(check),
                      "span_positive_counts": counts,
                      "coverage": mem["coverage"],
                      "max_distance": mem["max_distance"]}, indent=2))


if __name__ == "__main__":
    run()
