#!/usr/bin/env python3
"""Per-domain probe for the r32 domain-token Mirror LM.

Reports, for every held-out domain: CE/BPB/top-1, MoE utilization, effective
pairs, domain->pair routing matrix, per-pair next-byte specialization, and the
domain-token embedding norm. Geometry diagnostics are recorded per MoE layer.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import poincare_frontier as pf  # noqa: E402

EPS = 1e-9


def moe_module(block):
    return block.moe if getattr(block, "moe_additive", False) else block.ff


def counts_summary(counts: torch.Tensor) -> dict:
    n = int(counts.sum().item())
    p = counts.double() / max(n, 1)
    nz = p[p > 0]
    ent = float(-(nz * nz.log()).sum().item()) if nz.numel() else 0.0
    top = torch.sort(counts, descending=True).values
    return {
        "assignments": n,
        "active_pairs": int((counts > 0).sum().item()),
        "effective_pairs": round(math.exp(ent), 3),
        "max_share": round(float(p.max().item()), 6) if n else 0.0,
        "top8_share": round(float(top[:8].sum().item()) / max(n, 1), 6),
        "counts": [int(x) for x in counts.tolist()],
    }


def specialization(hist: torch.Tensor, global_hist: torch.Tensor, vocab: int) -> dict:
    rows = hist.float()
    sums = rows.sum(dim=1, keepdim=True)
    p_pair = (rows + EPS) / (sums + vocab * EPS)
    g = global_hist.float()
    p_global = (g + EPS) / (g.sum() + vocab * EPS)
    kl = (p_pair * (p_pair.log() - p_global.log())).sum(dim=1)
    active = sums.squeeze(1) > 0
    kls = kl[active]
    ranked = []
    for idx in torch.argsort(kl, descending=True)[:5].tolist():
        if sums[idx].item() <= 0:
            continue
        vals, ids = torch.topk(rows[idx], k=5)
        ranked.append({
            "pair": int(idx),
            "assignments": int(sums[idx].item()),
            "kl": round(float(kl[idx].item()), 6),
            "top_bytes": [
                {"byte": int(b), "char": chr(int(b)) if 32 <= int(b) < 127 else None,
                 "share": round(float(v) / max(float(sums[idx].item()), 1), 6)}
                for b, v in zip(ids.tolist(), vals.tolist())
            ],
        })
    return {
        "mean_kl": round(float(kls.mean().item()), 6) if kls.numel() else 0.0,
        "max_kl": round(float(kls.max().item()), 6) if kls.numel() else 0.0,
        "most_specialized": ranked,
    }


def geometry_stats(module) -> dict:
    with torch.no_grad():
        c = module.to_ball(module.centers).float().cpu()
        norms = c.norm(dim=-1)
        pair = torch.cdist(c, c)
        off = pair[~torch.eye(c.shape[0], dtype=torch.bool)]
        return {
            "strength": round(float(module.log_strength.exp().item()), 6),
            "centers_norm_mean": round(float(norms.mean().item()), 6),
            "pair_distance_mean": round(float(off.mean().item()), 6),
            "pair_bias_min": float(module.pair_bias.min().item()),
            "pair_bias_max": float(module.pair_bias.max().item()),
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--context", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    meta = pf.load_meta(args.data_dir)
    names = meta.get("domain_names") or []
    base = int(meta.get("domain_token_base", 257))
    ckpt = torch.load(pf.checkpoint_path(args.run_dir), map_location="cpu", weights_only=False)
    cfg = pf.Config(**ckpt["config"])
    if args.context + 1 > cfg.context:
        raise ValueError(f"context+1 exceeds model max {cfg.context}")
    model = pf.MirrorLM(cfg).to(device)
    pf.load_v25_compatible(model, ckpt["model"])
    model.eval()
    valmem = np.memmap(Path(args.data_dir) / "valid.bin",
                       dtype=meta.get("token_dtype", "<u4"), mode="r")
    lengths = pf.legacy_document_lengths(valmem, meta, "valid")
    domains = meta.get("valid_domains")
    sampler = pf.DocumentWindowSampler(valmem, args.context, lengths, domains=domains,
                                       domain_token_base=base)
    moe_layers = [i for i, b in enumerate(model.blocks) if b.use_moe]
    mods = {i: moe_module(model.blocks[i]) for i in moe_layers}

    per_domain, routing_matrix = [], []
    all_ce = all_n = all_correct = 0
    for d, name in enumerate(names):
        rng = np.random.default_rng(args.seed + d)
        counts = {i: torch.zeros(cfg.pairs, dtype=torch.long) for i in moe_layers}
        hist = {i: torch.zeros(cfg.pairs, cfg.vocab_size, dtype=torch.long) for i in moe_layers}
        ce_sum = correct = total = 0
        try:
            with torch.no_grad():
                for _ in range(args.batches):
                    ids = sampler.domain_windows(d, args.batch, rng, device)
                    x, targets = ids[:, :-3], ids[:, 1:-2]
                    logits, _, metrics = model(x, targets)
                    n = targets.numel()
                    ce_sum += float(metrics["next_ce"].item()) * n
                    correct += int((logits.argmax(dim=-1) == targets).sum().item())
                    total += n
                    flat = targets.reshape(-1).cpu()
                    for layer in moe_layers:
                        mod = mods[layer]
                        routed = mod.last_top_idx.reshape(-1, mod.topk).cpu()
                        counts[layer] += torch.bincount(routed.reshape(-1), minlength=cfg.pairs)
                        for slot in range(mod.topk):
                            bins = torch.bincount(routed[:, slot] * cfg.vocab_size + flat,
                                                  minlength=cfg.pairs * cfg.vocab_size)
                            hist[layer] += bins.reshape(cfg.pairs, cfg.vocab_size)
        except ValueError as exc:
            per_domain.append({"domain": name, "id": d, "error": str(exc)})
            routing_matrix.append([0] * cfg.pairs)
            continue
        global_hist = torch.zeros(cfg.vocab_size, dtype=torch.long)
        for layer in moe_layers:
            global_hist += hist[layer].sum(dim=0) // max(len(moe_layers) * mods[layer].topk, 1)
        union = torch.zeros(cfg.pairs, dtype=torch.long)
        for layer in moe_layers:
            union += counts[layer]
        routing_matrix.append([int(x) for x in union.tolist()])
        entry = {
            "domain": name, "id": d, "positions": total,
            "ce": round(ce_sum / max(total, 1), 6),
            "bpb": round(ce_sum / max(total, 1) / math.log(2), 6),
            "top1": round(correct / max(total, 1), 6),
            "domain_token_id": base + d,
            "domain_token_norm": round(float(model.embed.weight[base + d].norm().item()), 6),
            "union": counts_summary(union),
            "layers": {
                str(layer): {
                    **counts_summary(counts[layer]),
                    "specialization": specialization(hist[layer], global_hist, cfg.vocab_size),
                    "geometry": geometry_stats(mods[layer]),
                }
                for layer in moe_layers
            },
        }
        per_domain.append(entry)
        all_ce += ce_sum; all_n += total; all_correct += correct
        print(f"{name:14} bpb={entry['bpb']:.4f} top1={entry['top1']*100:5.2f}% "
              f"active={entry['union']['active_pairs']}/{cfg.pairs} "
              f"eff={entry['union']['effective_pairs']:.1f}", flush=True)

    result = {
        "run_dir": str(Path(args.run_dir).resolve()),
        "checkpoint_step": int(ckpt["step"]),
        "parameters": sum(p.numel() for p in model.parameters()),
        "config": ckpt["config"], "domains": names,
        "evaluated_positions": all_n,
        "aggregate": {
            "ce": round(all_ce / max(all_n, 1), 6),
            "bpb": round(all_ce / max(all_n, 1) / math.log(2), 6),
            "top1": round(all_correct / max(all_n, 1), 6),
        },
        "per_domain": per_domain,
        "routing_matrix": routing_matrix,
    }
    text = json.dumps(result, indent=2)
    print(text, flush=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
