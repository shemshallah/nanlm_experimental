#!/usr/bin/env python3
"""Measure optimizer-step throughput for the r31 arms on the available GPU."""
from __future__ import annotations

import gc
import json
import shutil
import sys
import time
from pathlib import Path

import torch

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
CTX = 256


def main() -> int:
    code_hits = sorted(INPUT.rglob("probe_r31.py"))
    if not code_hits:
        print("r31 code not mounted", file=sys.stderr)
        return 2
    code_dir = WORK / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    for name in ("poincare_frontier.py", "probe_r31.py", "gen_r31.py"):
        shutil.copy2(code_hits[0].parent / name, code_dir / name)
    sys.path.insert(0, str(code_dir))
    import poincare_frontier as pf

    ngpu = torch.cuda.device_count()
    print(f"gpu_count={ngpu} devices={[torch.cuda.get_device_name(i) for i in range(ngpu)]}", flush=True)
    configs = {
        "plain21": dict(dim=576, layers=4, heads=24, pairs=128, topk_pairs=2,
                        expert_hidden=100, moe_every=5, hyperbolic_strength=0.0,
                        router_mode="linear", moe_additive=False, context=CTX),
        "hybrid151": dict(dim=544, layers=4, heads=16, pairs=128, topk_pairs=2,
                          expert_hidden=235, moe_every=2, hyperbolic_strength=1.0,
                          router_mode="both", moe_additive=True, context=CTX),
        "control21": dict(dim=192, layers=4, heads=16, pairs=128, topk_pairs=2,
                          expert_hidden=100, moe_every=2, hyperbolic_strength=1.0,
                          router_mode="both", moe_additive=False, context=CTX),
    }
    results = {}
    for name, kw in configs.items():
        results[name] = {}
        for batch in (8, 16, 32):
            torch.cuda.empty_cache()
            gc.collect()
            cfg = pf.Config(vocab_size=257, **kw)
            model = pf.MirrorLM(cfg).cuda()
            params = sum(p.numel() for p in model.parameters())
            optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
            ids = torch.randint(0, 257, (batch, CTX + 3), device="cuda")
            model.train()
            try:
                for _ in range(2):
                    optim.zero_grad(set_to_none=True)
                    loss = model.loss_with_future(ids)["loss"]
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.time()
                steps = 5
                for _ in range(steps):
                    optim.zero_grad(set_to_none=True)
                    loss = model.loss_with_future(ids)["loss"]
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
                torch.cuda.synchronize()
                elapsed = time.time() - started
                results[name][str(batch)] = {
                    "params": params,
                    "seconds_per_step": round(elapsed / steps, 4),
                    "tokens_per_second": round(batch * CTX * steps / elapsed, 1),
                    "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                }
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                results[name][str(batch)] = {"params": params, "oom": True}
                torch.cuda.empty_cache()
            finally:
                del model, optim, ids
                torch.cuda.empty_cache()
                gc.collect()
            print(name, batch, results[name][str(batch)], flush=True)
    print("BENCH\n" + json.dumps(results, indent=2), flush=True)
    (WORK / "bench.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
