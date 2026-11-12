#!/usr/bin/env python3
"""nanlmv3 v2 — RoPE, Pre-LN/RMSNorm/SwiGLU, learnable hyperbolic routing,
two mirror-pair MoE layers, and a hashed n-gram memory.

Text mode only.  The 1-4 branch gate remains a separate canary run.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from head_geometry import TorchHeadGeometry
from hybrid_poincare import PairMoE
from poincare_geometry import PoincareTriheadLayer, POINCARE_OUT_DIM


# ---------------------------------------------------------------------------
# Trunk
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class RoPE(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, T, D)
        t = torch.arange(x.shape[-2], device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.h = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.rope = RoPE(self.dh)
        self.drop = dropout

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q = q.view(b, t, self.h, self.dh).transpose(1, 2)
        k = k.view(b, t, self.h, self.dh).transpose(1, 2)
        v = v.view(b, t, self.h, self.dh).transpose(1, 2)
        q = self.rope(q)
        k = self.rope(k)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.drop if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        return self.out(y)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, hidden: int, dropout: float):
        super().__init__()
        self.n1 = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, dropout)
        self.n2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, hidden, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.n1(x)))
        x = x + self.mlp(self.n2(x))
        return x


class Trunk(nn.Module):
    def __init__(self, d_model: int, layers: int, heads: int, swiglu: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(d_model, heads, swiglu, dropout) for _ in range(layers)]
        )
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Hashed n-gram memory
# ---------------------------------------------------------------------------


class HashedMemory:
    """1M-entry top-1 4-gram memory with collision keys."""

    def __init__(self, size: int = 1 << 20, order: int = 4, min_count: int = 5):
        self.size = size
        self.order = order
        self.min_count = min_count
        self.keys = np.full(size, -1, dtype=np.int64)
        self.next_tok = np.zeros(size, dtype=np.int64)
        self.counts = np.zeros(size, dtype=np.int32)

    @staticmethod
    def _hash(ctx: np.ndarray) -> int:
        h = 1469598103934665603
        for v in ctx:
            h ^= int(v) + 0x9E3779B97F4A7C15
            h = (h * 1099511628211) & 0x7FFFFFFFFFFFFFFF
        return h

    def build(self, ids: np.ndarray, max_tokens: int = 20_000_000):
        n = min(len(ids) - self.order, max_tokens)
        keys = self.keys
        nxt = self.next_tok
        cnt = self.counts
        size = self.size
        for i in range(n):
            ctx = ids[i : i + self.order]
            h = self._hash(ctx)
            b = h % size
            if keys[b] == -1:
                keys[b] = h
                nxt[b] = ids[i + self.order]
                cnt[b] = 1
            elif keys[b] == h:
                if nxt[b] == ids[i + self.order]:
                    cnt[b] += 1
                elif cnt[b] > 0:
                    cnt[b] -= 1
            # collision: leave the incumbent; count-min style
        self.keys = keys
        self.next_tok = nxt
        self.counts = cnt

    def bias(self, context: np.ndarray, vocab: int, alpha: float) -> np.ndarray:
        h = self._hash(context[-self.order :])
        b = h % self.size
        out = np.zeros(vocab, dtype=np.float32)
        if self.keys[b] == h and self.counts[b] > self.min_count:
            out[self.next_tok[b]] = alpha * math.log(float(self.counts[b]))
        return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class UnifiedV2LM(nn.Module):
    def __init__(self, vocab: int, ctx: int, d_model: int = 256, layers: int = 8,
                 heads: int = 8, swiglu: int = 1024, d_hidden: int = 512,
                 n_pairs: int = 128, topk: int = 8, n_branches: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, d_model)
        self.trunk = Trunk(d_model, layers, heads, swiglu, dropout)
        self.attn_score = nn.Linear(d_model, 1)
        self.ctx_proj = nn.Linear(3 * d_model, d_model)
        self.branch_proj = nn.Linear(d_model, 8)
        # Learnable centroids, radius 0.6, angles spread.
        cent = torch.zeros(n_branches, 8)
        for i in range(n_branches):
            angle = 2.0 * math.pi * i / n_branches
            cent[i, 0] = 0.6 * math.cos(angle)
            cent[i, 1] = 0.6 * math.sin(angle)
        self.centroids = nn.Parameter(cent)
        self.log_curvature = nn.Parameter(torch.zeros(()))  # curvature -1 at init
        self.branch_temp = nn.Parameter(torch.ones(()))
        self.router = nn.Linear(d_model + n_branches, d_model)

        self.head_proj = nn.Linear(d_model, 255)
        self.geom = TorchHeadGeometry()
        self.pgeom = PoincareTriheadLayer()
        geom_dim = 129 + POINCARE_OUT_DIM
        self.geom_bias1 = nn.Linear(geom_dim, n_pairs)
        self.geom_bias2 = nn.Linear(geom_dim, n_pairs)

        self.moe1 = PairMoE(d_model, d_hidden, n_pairs, topk)
        self.moe2 = PairMoE(d_model, d_hidden, n_pairs, topk)
        self.res1 = nn.Linear(d_hidden, d_model)
        self.res2 = nn.Linear(d_hidden, d_model)
        self.n1 = RMSNorm(d_model)
        self.n2 = RMSNorm(d_model)
        self.prev_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.memory_alpha = nn.Parameter(torch.tensor(0.1))
        self.apply(self._init)
        nn.init.normal_(self.tok.weight, mean=0.0, std=0.02)
        for g in (self.geom_bias1, self.geom_bias2):
            nn.init.zeros_(g.weight)
            nn.init.zeros_(g.bias)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def hyperbolic_branch(self, ctx_vec):
        u = torch.tanh(self.branch_proj(ctx_vec)) * 0.95
        c = self.centroids
        u2 = (u * u).sum(-1, keepdim=True)
        c2 = (c * c).sum(-1)
        diff2 = ((u[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        denom = torch.clamp((1 - u2) * (1 - c2)[None, :], min=1e-6)
        dist = torch.acosh(torch.clamp(1 + 2 * diff2 / denom, min=1.0 + 1e-6, max=1e12))
        curv = -torch.exp(self.log_curvature)
        dist = dist * torch.sqrt(-curv)
        return F.softmax(-dist * self.branch_temp, dim=-1)

    def forward(self, tokens):
        b, t = tokens.shape
        x = self.tok(tokens)
        x = self.trunk(x)
        scores = self.attn_score(self.drop(x)).squeeze(-1)
        attn = F.softmax(scores, dim=-1)
        attended = torch.einsum("bt,btd->bd", attn, x)
        last = x[:, -1, :]
        last2 = x[:, -2, :] if t >= 2 else last
        ctx_vec = F.relu(self.ctx_proj(torch.cat([last, last2, attended], dim=-1)))
        branch = self.hyperbolic_branch(ctx_vec)
        routed = F.relu(self.router(torch.cat([ctx_vec, branch], dim=-1)))
        credits = self.head_proj(ctx_vec)
        gfeat = self.geom(credits)
        pfeat, paux = self.pgeom(credits)
        g = torch.cat([gfeat, pfeat], dim=-1)
        gb1 = self.geom_bias1(g)
        gb2 = self.geom_bias2(g)
        h1, lb1 = self.moe1(routed, gb1)
        z1 = self.n1(routed + self.res1(self.drop(h1)))
        h2, lb2 = self.moe2(z1, gb2)
        z2 = self.n2(z1 + self.res2(self.drop(h2)))
        logits = z2 @ self.tok.weight.T
        prev_logits = self.prev_proj(z2) @ self.tok.weight.T
        aux = {
            "load_balance": lb1 + lb2,
            "mirror_loss": paux["mirror_loss"],
            "geom_load": paux["load_balance"],
            "curv_reg": paux["curv_reg"],
            "branch_mean": branch.mean(dim=0),
            "centroids": self.centroids.detach(),
            "curvature": (-torch.exp(self.log_curvature)).detach(),
            "branch_temp": self.branch_temp.detach(),
        }
        return logits, prev_logits, aux


# ---------------------------------------------------------------------------
# Data / training
# ---------------------------------------------------------------------------


def text_data(corpus, tokenizer, ctx, train_fraction, max_bytes):
    """Encode in bounded chunks to avoid a huge Python-int list."""
    parts: list[np.ndarray] = []
    total = 0
    with open(corpus, "rb") as f:
        while True:
            chunk = f.read(8_000_000)
            if not chunk:
                break
            if max_bytes > 0:
                remaining = max_bytes - total
                if remaining <= 0:
                    break
                chunk = chunk[:remaining]
            text = chunk.decode("utf-8", errors="ignore")
            arr = tokenizer.encode(text)
            if len(arr):
                parts.append(np.asarray(arr, dtype=np.int32))
            total += len(chunk)
            if max_bytes > 0 and total >= max_bytes:
                break
    if not parts:
        raise SystemExit("corpus produced no tokens")
    ids = np.concatenate(parts)
    if len(ids) < ctx + 64:
        raise SystemExit("corpus too small")
    split = int(len(ids) * train_fraction)
    split = max(ctx + 32, min(split, len(ids) - 64))
    return ids, split


def batch_at(ids, split, ctx, batch, rng, train):
    lo = ctx
    hi = split if train else len(ids) - 1
    if not train:
        lo = max(split, ctx)
    pos = rng.integers(lo, hi, size=batch)
    return batch_positions(ids, pos, ctx)


def batch_positions(ids, positions, ctx):
    pos = np.asarray(positions, dtype=np.int64)
    xs = np.stack([ids[p - ctx : p] for p in pos]).astype(np.int64)
    ys = np.array([ids[p] for p in pos], dtype=np.int64)
    prev_pos = pos - ctx - 1
    valid = prev_pos >= 0
    yp = np.zeros(len(pos), dtype=np.int64)
    yp[valid] = ids[prev_pos[valid]]
    pm = valid.astype(np.float32)
    return torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(yp), torch.from_numpy(pm)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--bpe-model", default="")
    p.add_argument("--tokenizer", choices=["bpe", "bytes"], default="bpe")
    p.add_argument("--out", default="/kaggle/working/v2")
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--swiglu", type=int, default=1024)
    p.add_argument("--d-hidden", type=int, default=512)
    p.add_argument("--pairs", type=int, default=128)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--branches", type=int, default=32)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--max-seconds", type=float, default=0.0)
    p.add_argument("--lr-horizon", choices=["auto", "steps", "time"], default="auto")
    p.add_argument("--min-lr-frac", type=float, default=0.0)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--ema-start-frac", type=float, default=0.5)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--train-fraction", type=float, default=0.95)
    p.add_argument("--max-bytes", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--prev-weight", type=float, default=0.1)
    p.add_argument("--recluster-every", type=int, default=1000)
    p.add_argument("--memory-size", type=int, default=1 << 20)
    p.add_argument("--memory-tokens", type=int, default=20_000_000)
    p.add_argument("--no-memory", action="store_true")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--amp-init-scale", type=float, default=1024.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=3)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", init_scale=args.amp_init_scale) if amp else None
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    if args.tokenizer == "bpe":
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(args.bpe_model)
        vocab = tok.get_vocab_size()

        class Tok:
            def encode(self, s):
                return np.asarray(tok.encode(s).ids, dtype=np.int32)
    else:
        vocab = 256

        class Tok:
            def encode(self, s):
                return np.frombuffer(s.encode("utf-8", errors="replace"), dtype=np.uint8).astype(np.int32)

    ids, split = text_data(args.corpus, Tok(), args.ctx, args.train_fraction, args.max_bytes)
    print(f"[data] tokenizer={args.tokenizer} vocab={vocab} tokens={len(ids)} train={split} ctx={args.ctx}", flush=True)

    memory = None
    if not args.no_memory:
        t0 = time.time()
        memory = HashedMemory(args.memory_size, order=4)
        memory.build(ids[:split], max_tokens=args.memory_tokens)
        print(f"[memory] built in {time.time() - t0:.1f}s", flush=True)

    model = UnifiedV2LM(
        vocab, args.ctx, args.d_model, args.layers, args.heads, args.swiglu,
        args.d_hidden, args.pairs, args.topk, args.branches, args.dropout,
    ).to(device)
    print(f"[model] params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema_model = copy.deepcopy(model) if not args.no_ema else None
    if ema_model is not None:
        for p_ in ema_model.parameters():
            p_.requires_grad_(False)
        ema_model.eval()
    ema_start = int(args.steps * args.ema_start_frac) if args.steps > 0 else 0
    rng = np.random.default_rng(args.seed)
    eval_rng = np.random.default_rng(12345)
    val_lo = max(split, args.ctx)
    val_hi = len(ids) - 1
    val_positions = eval_rng.integers(val_lo, val_hi, size=args.eval_batches * args.batch_size)
    init_centroids = model.centroids.detach().clone()
    t0 = time.time()
    best = float("inf")
    best_rec = {}
    history = []

    def mem_bias_np(context):
        if memory is None:
            return None
        return memory.bias(context, vocab, 1.0)

    def memory_tensor(xb):
        if memory is None:
            return None
        mb = np.zeros((xb.shape[0], vocab), dtype=np.float32)
        for j in range(xb.shape[0]):
            b = mem_bias_np(xb[j].cpu().numpy())
            if b is not None:
                mb[j] = b
        return torch.from_numpy(mb).to(xb.device)

    def eval_text(net):
        net.eval()
        total = correct = top5 = 0
        prev_c = prev_t = 0
        loss_sum = 0.0
        with torch.no_grad():
            for i in range(0, len(val_positions), args.batch_size):
                pos = val_positions[i : i + args.batch_size]
                xb, yb, ypb, pmb = batch_positions(ids, pos, args.ctx)
                xb, yb = xb.to(device), yb.to(device)
                logits, plogits, aux = net(xb)
                mbt = memory_tensor(xb)
                if mbt is not None:
                    logits = logits + torch.clamp(net.memory_alpha, 0.0, 0.5) * mbt
                loss_sum += float(F.cross_entropy(logits, yb, reduction="sum"))
                correct += int((logits.argmax(-1) == yb).sum())
                t5 = logits.topk(5, dim=-1).indices
                top5 += int((t5 == yb.unsqueeze(-1)).any(-1).sum())
                total += yb.numel()
        model.train()
        return {
            "loss": loss_sum / max(1, total),
            "bits_per_token": (loss_sum / max(1, total)) / math.log(2),
            "accuracy": correct / max(1, total),
            "top5": top5 / max(1, total),
            "tokens": total,
        }

    for step in range(args.steps):
        if args.max_seconds > 0 and (time.time() - t0) >= args.max_seconds:
            print(f"[stop] max-seconds reached at step {step}", flush=True)
            break
        xb, yb, ypb, pmb = batch_at(ids, split, args.ctx, args.batch_size, rng, True)
        xb, yb, ypb, pmb = xb.to(device), yb.to(device), ypb.to(device), pmb.to(device)
        if step < 200:
            lr = args.lr * min(1.0, (step + 1) / 200)
        else:
            use_time = args.lr_horizon == "time" or (args.lr_horizon == "auto" and args.max_seconds > 0)
            if use_time:
                prog = min(1.0, (time.time() - t0) / max(1.0, args.max_seconds))
            else:
                prog = (step - 200) / max(1, args.steps - 200)
            frac = args.min_lr_frac + (1.0 - args.min_lr_frac) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
            lr = args.lr * frac
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            logits, plogits, aux = model(xb)
            mbt = memory_tensor(xb)
            if mbt is not None:
                logits = logits + model.memory_alpha * mbt
            loss = F.cross_entropy(logits, yb)
            mask = pmb.bool()
            if mask.any():
                loss = loss + args.prev_weight * F.cross_entropy(plogits[mask], ypb[mask])
            loss = loss + 0.01 * aux["load_balance"]
            loss = loss + 0.05 * aux["mirror_loss"]
            loss = loss + 0.001 * aux["geom_load"]
            loss = loss + 1e-4 * aux["curv_reg"]
        if scaler is not None:
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() < scale_before:
                print(f"[amp-skip] step {step} scale {scale_before:.1f}->{scaler.get_scale():.1f}", flush=True)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if ema_model is not None and (step + 1) >= ema_start:
            with torch.no_grad():
                d = args.ema_decay
                for e, p_ in zip(ema_model.parameters(), model.parameters()):
                    e.mul_(d).add_(p_, alpha=1.0 - d)
        if (step + 1) % args.recluster_every == 0:
            model.pgeom.recluster()
            if ema_model is not None:
                with torch.no_grad():
                    for eb, mb in zip(ema_model.buffers(), model.buffers()):
                        if eb.shape == mb.shape and eb.dtype == mb.dtype:
                            eb.copy_(mb)
        if (step + 1) % 50 == 0:
            print(f"step {step + 1:6d}/{args.steps} loss={float(loss):.4f} lr={lr:.2e}", flush=True)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            ev = eval_text(model)
            ev_ema = eval_text(ema_model) if ema_model is not None else None
            history.append({"step": step + 1, "model": ev, "ema": ev_ema})
            print(
                f"  EVAL {step + 1}: model loss={ev['loss']:.4f} bits={ev['bits_per_token']:.3f} "
                f"top1={ev['accuracy'] * 100:.2f}% top5={ev['top5'] * 100:.2f}%",
                flush=True,
            )
            if ev_ema is not None:
                print(
                    f"        ema   loss={ev_ema['loss']:.4f} bits={ev_ema['bits_per_token']:.3f} "
                    f"top1={ev_ema['accuracy'] * 100:.2f}% top5={ev_ema['top5'] * 100:.2f}%",
                    flush=True,
                )
            choose = ev_ema if (ev_ema is not None and ev_ema["loss"] < ev["loss"]) else ev
            source = "ema" if choose is ev_ema else "model"
            if choose["loss"] < best:
                best = choose["loss"]
                best_rec = {"step": step + 1, "source": source, **choose}
                torch.save(
                    {"model": (ema_model if source == "ema" else model).state_dict(),
                     "args": vars(args), "step": step + 1, "metrics": choose,
                     "source": source},
                    out / "best.pt",
                )
    final = eval_text(model)
    final_ema = eval_text(ema_model) if ema_model is not None else None
    torch.save(
        {"model": model.state_dict(), "args": vars(args),
         "final": final, "best": best_rec},
        out / "final.pt",
    )
    if ema_model is not None:
        torch.save(
            {"model": ema_model.state_dict(), "args": vars(args),
             "final": final_ema, "best": best_rec},
            out / "ema.pt",
        )
    (out / "history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")
    centroid_delta = float((model.centroids.detach() - init_centroids).norm())
    result = {
        "params": sum(p.numel() for p in model.parameters()),
        "steps": args.steps,
        "tokens": len(ids),
        "train_tokens": int(split),
        "val_tokens": int(len(ids) - split),
        "best": best_rec,
        "final": final,
        "final_ema": final_ema,
        "elapsed": time.time() - t0,
        "centroid_delta": centroid_delta,
        "curvature": float((-torch.exp(model.log_curvature)).detach()),
        "branch_temp": float(model.branch_temp.detach()),
        "memory_alpha": float(model.memory_alpha.detach()),
        "memory_enabled": memory is not None,
        "artifacts": {
            "best_weights": str(out / "best.pt"),
            "final_weights": str(out / "final.pt"),
            "result": str(out / "result.json"),
            "history": str(out / "history.json"),
            "config": str(out / "config.json"),
        },
        "config": vars(args),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
