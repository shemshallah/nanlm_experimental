#!/usr/bin/env python3
"""r31 probe: held-out BPB, per-book BPB, MoE utilization, expert
specialization, and geometry-vs-router diagnostics.

Works with both replacement MoE blocks (R30) and additive MoE blocks (r31).
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


def describe_counts(count_tensor: torch.Tensor) -> dict:
    n = int(count_tensor.sum().item())
    p = count_tensor.double() / max(n, 1)
    nz = p[p > 0]
    entropy = float(-(nz * nz.log()).sum().item()) if nz.numel() else 0.0
    ordered = torch.sort(count_tensor, descending=True).values
    cum = {f"top{k}": round(float(ordered[:k].sum().item()) / max(n, 1), 6)
           for k in (1, 2, 4, 8, 16, 32)}
    return {
        "assignments": n,
        "active_pairs": int((count_tensor > 0).sum().item()),
        "dead_pairs": int((count_tensor == 0).sum().item()),
        "effective_pairs": round(math.exp(entropy), 3),
        "entropy_nats": round(entropy, 6),
        "max_share": round(float(p.max().item()), 6) if n else 0.0,
        "cumulative_share": cum,
        "counts": [int(v) for v in count_tensor.tolist()],
    }


def specialization(hist: torch.Tensor, global_hist: torch.Tensor, vocab: int) -> dict:
    """hist [pairs, vocab] counts, global_hist [vocab] counts over tokens."""
    rows = hist.float()
    row_sums = rows.sum(dim=1, keepdim=True)
    p_pair = (rows + EPS) / (row_sums + vocab * EPS)
    g = global_hist.float()
    p_global = (g + EPS) / (g.sum() + vocab * EPS)
    kl = (p_pair * (p_pair.log() - p_global.log())).sum(dim=1)
    active = row_sums.squeeze(1) > 0
    kl_active = kl[active]
    top_pairs = []
    order = torch.argsort(kl, descending=True)
    for idx in order[:8].tolist():
        if row_sums[idx].item() <= 0:
            continue
        vals, ids = torch.topk(rows[idx], k=min(5, vocab))
        shares = (vals / row_sums[idx].squeeze(0)).tolist()
        top_pairs.append({
            "pair": int(idx),
            "assignments": int(row_sums[idx].item()),
            "kl_to_global": round(float(kl[idx].item()), 6),
            "top_bytes": [
                {"byte": int(b), "char": chr(int(b)) if 32 <= int(b) < 127 else None,
                 "share": round(float(s), 6)}
                for b, s in zip(ids.tolist(), shares)
            ],
        })
    global_top = torch.topk(global_hist, k=min(10, vocab))
    return {
        "mean_kl_to_global": round(float(kl_active.mean().item()), 6) if kl_active.numel() else 0.0,
        "max_kl_to_global": round(float(kl_active.max().item()), 6) if kl_active.numel() else 0.0,
        "min_kl_to_global": round(float(kl_active.min().item()), 6) if kl_active.numel() else 0.0,
        "active_pairs": int(active.sum().item()),
        "most_specialized": top_pairs,
        "global_top_bytes": [
            {"byte": int(b), "char": chr(int(b)) if 32 <= int(b) < 127 else None,
             "share": round(float(v) / max(int(global_hist.sum().item()), 1), 6)}
            for b, v in zip(global_top.indices.tolist(), global_top.values.tolist())
        ],
    }


def geometry_stats(module) -> dict:
    with torch.no_grad():
        c = module.to_ball(module.centers).float().cpu()
        norms = c.norm(dim=-1)
        pairwise = torch.cdist(c, c)
        off = pairwise[~torch.eye(c.shape[0], dtype=torch.bool)]
        try:
            hyperbolic = module.hyperbolic_distance(c, c).float().cpu()
            hoff = hyperbolic[~torch.eye(c.shape[0], dtype=torch.bool)]
            hyp_mean = float(hoff.mean().item())
            hyp_min = float(hoff.min().item())
            hyp_max = float(hoff.max().item())
        except Exception as exc:  # noqa: BLE001
            hyp_mean = hyp_min = hyp_max = None
            print(f"hyperbolic center distance failed: {exc}", file=sys.stderr)
        return {
            "strength": round(float(module.log_strength.exp().item()), 6),
            "centers_norm_mean": round(float(norms.mean().item()), 6),
            "centers_norm_median": round(float(norms.median().item()), 6),
            "centers_norm_min": round(float(norms.min().item()), 6),
            "centers_norm_max": round(float(norms.max().item()), 6),
            "pair_distance_mean": round(float(off.mean().item()), 6),
            "pair_distance_min": round(float(off.min().item()), 6),
            "pair_distance_max": round(float(off.max().item()), 6),
            "hyperbolic_pair_distance_mean": round(hyp_mean, 6) if hyp_mean is not None else None,
            "hyperbolic_pair_distance_min": round(hyp_min, 6) if hyp_min is not None else None,
            "hyperbolic_pair_distance_max": round(hyp_max, 6) if hyp_max is not None else None,
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--split", default="valid", choices=("valid", "test"))
    ap.add_argument("--context", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--batches", type=int, default=32)
    ap.add_argument("--book-windows", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    meta = pf.load_meta(args.data_dir)
    ckpt = torch.load(pf.checkpoint_path(args.run_dir), map_location="cpu", weights_only=False)
    cfg = pf.Config(**ckpt["config"])
    if not 4 <= args.context <= cfg.context:
        raise ValueError(f"context must be within 4..{cfg.context}")
    model = pf.MirrorLM(cfg).to(device)
    pf.load_v25_compatible(model, ckpt["model"])
    model.eval()

    mem = np.memmap(Path(args.data_dir) / (args.split + ".bin"),
                    dtype=meta.get("token_dtype", "<u4"), mode="r")
    lengths = pf.legacy_document_lengths(mem, meta, args.split)
    sampler = pf.DocumentWindowSampler(mem, args.context, lengths)
    rng = np.random.default_rng(args.seed)

    moe_layers = [i for i, block in enumerate(model.blocks) if block.use_moe]
    mods = {i: moe_module(model.blocks[i]) for i in moe_layers}
    counts = {i: torch.zeros(cfg.pairs, dtype=torch.long) for i in moe_layers}
    hist = {i: torch.zeros(cfg.pairs, cfg.vocab_size, dtype=torch.long) for i in moe_layers}
    router_std = {i: 0.0 for i in moe_layers}
    geo_std = {i: 0.0 for i in moe_layers}
    router_abs = {i: 0.0 for i in moe_layers}
    geo_abs = {i: 0.0 for i in moe_layers}
    seen_batches = 0
    ce_sum = 0.0
    correct = 0
    total = 0
    global_hist = torch.zeros(cfg.vocab_size, dtype=torch.long)

    with torch.no_grad():
        for _ in range(args.batches):
            ids = sampler.windows(args.batch, rng, device)
            x = ids[:, :-3]
            targets = ids[:, 1:-2]
            logits, _, metrics = model(x, targets)
            n = targets.numel()
            ce_sum += float(metrics["next_ce"].item()) * n
            correct += int((logits.argmax(dim=-1) == targets).sum().item())
            total += n
            target_flat = targets.reshape(-1).cpu()
            global_hist += torch.bincount(target_flat, minlength=cfg.vocab_size)
            seen_batches += 1
            for layer in moe_layers:
                mod = mods[layer]
                routed = mod.last_top_idx.reshape(-1, mod.topk).cpu()
                counts[layer] += torch.bincount(routed.reshape(-1), minlength=cfg.pairs)
                for slot in range(mod.topk):
                    bins = torch.bincount(routed[:, slot] * cfg.vocab_size + target_flat,
                                          minlength=cfg.pairs * cfg.vocab_size)
                    hist[layer] += bins.reshape(cfg.pairs, cfg.vocab_size)
                rlog = mod.last_router_logits.float().cpu()
                router_std[layer] += float(rlog.std(dim=1).mean().item())
                router_abs[layer] += float(rlog.abs().mean().item())
                if mod.last_geo_term is not None:
                    gterm = mod.last_geo_term.float().cpu()
                    geo_std[layer] += float(gterm.std(dim=1).mean().item())
                    geo_abs[layer] += float(gterm.abs().mean().item())

    def layer_block(layer: int) -> dict:
        mod = mods[layer]
        out = {
            **describe_counts(counts[layer]),
            "pair_bias_min": float(mod.pair_bias.min().item()),
            "pair_bias_max": float(mod.pair_bias.max().item()),
            "specialization": specialization(hist[layer], global_hist, cfg.vocab_size),
            "geometry": geometry_stats(mod),
        }
        out["geometry"].update({
            "router_logit_std_mean": round(router_std[layer] / max(seen_batches, 1), 6),
            "router_logit_abs_mean": round(router_abs[layer] / max(seen_batches, 1), 6),
            "geometry_term_std_mean": round(geo_std[layer] / max(seen_batches, 1), 6),
            "geometry_term_abs_mean": round(geo_abs[layer] / max(seen_batches, 1), 6),
        })
        rstd = out["geometry"]["router_logit_std_mean"]
        gstd = out["geometry"]["geometry_term_std_mean"]
        out["geometry"]["geo_to_router_std_ratio"] = round(gstd / rstd, 6) if rstd > 0 else None
        return out

    union = torch.zeros(cfg.pairs, dtype=torch.long)
    for layer in moe_layers:
        union += counts[layer]

    # Per-book held-out BPB.
    manifest_path = Path(args.data_dir) / "split_manifest.json"
    names = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        names = [b["path"] for b in manifest.get(args.split + "_books", [])]
    offsets = np.cumsum([0] + list(lengths))
    per_book = []
    for idx, length in enumerate(lengths):
        start = int(offsets[idx])
        if length <= args.context + 3:
            continue
        book_rng = np.random.default_rng(args.seed + idx)
        book_ce = 0.0
        book_n = 0
        with torch.no_grad():
            for _ in range(args.book_windows):
                i = int(book_rng.integers(start, start + length - (args.context + 3) + 1))
                ids = torch.from_numpy(
                    np.asarray(mem[i:i + args.context + 3]).astype(np.int64)
                ).unsqueeze(0).to(device)
                _, _, metrics = model(ids[:, :-3], ids[:, 1:-2])
                book_ce += float(metrics["next_ce"].item()) * args.context
                book_n += args.context
        per_book.append({
            "book": names[idx] if idx < len(names) else f"book_{idx}",
            "positions": book_n,
            "ce": round(book_ce / max(book_n, 1), 6),
            "bpb": round(book_ce / max(book_n, 1) / math.log(2), 6),
        })

    result = {
        "run_dir": str(Path(args.run_dir).resolve()),
        "checkpoint_step": int(ckpt["step"]),
        "parameters": sum(p.numel() for p in model.parameters()),
        "config": ckpt["config"],
        "evaluated_positions": total,
        "ce": round(ce_sum / max(total, 1), 6),
        "bpb": round(ce_sum / max(total, 1) / math.log(2), 6),
        "top1": round(correct / max(total, 1), 6),
        "moe_layers": moe_layers,
        "layers": {str(layer): layer_block(layer) for layer in moe_layers},
        "union": describe_counts(union),
        "per_book": per_book,
    }
    text = json.dumps(result, indent=2)
    print(text, flush=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
