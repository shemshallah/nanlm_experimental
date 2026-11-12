# r31 — compute-matched MoE and geometry routing

r31 is the experimental branch that fixes the two main flaws in the R30
parameter-matched ablation:

1. **Compute matching.** The hybrid is a dense trunk plus an additive sparse
   MoE branch, so total parameters can grow while active MAC/token stays near
   the dense baseline.
2. **Geometry as a router, not a penalty.** The router can be `linear`, `geo`,
   or `both`; the geometry strength is a learnable scalar initialized at 1.0.

It also logs what previous runs did not: per-layer MoE utilization, per-pair
next-byte specialization, centers norm, pairwise center distance, and
geometry-term variance vs router-logit variance. Evaluation is 65,536 held-out
positions plus per-book BPB.

## What changed in `code/poincare_frontier.py`

- `Config` additions:
  - `router_mode: 'linear' | 'geo' | 'both'`
  - `learnable_strength: bool`
  - `moe_additive: bool`
- `MirrorPairMoE.forward`:
  - `scores = router(x)` for linear
  - `scores = strength * -hyperbolic_distance(...)` for geo
  - `scores = router(x) + strength * -hyperbolic_distance(...)` for both
  - `strength = exp(log_strength)`, initialized to `hyperbolic_strength`
  - stores `last_router_logits`, `last_geo_term`, `last_strength`,
    `last_top_idx` for probes only (routing unchanged)
- `Block`:
  - `moe_additive=False` preserves the R30 behavior (MoE replaces SwiGLU)
  - `moe_additive=True` keeps the dense SwiGLU and adds a separate MoE branch
  - `future_logits` handles the additive case correctly
- CLI: `--router-mode {linear,geo,both}`, `--moe-additive`
- `load_v25_compatible` tolerates missing `.log_strength` / `.moe.*` on old
  checkpoints.

## Model arms used in the latest run

| Arm | Config | Params | Active MAC/token (est.) |
|---|---|---:|---:|
| Plain | dim 576, 4 layers, 24 heads, SwiGLU, no MoE | 21,388,032 | ~21.4M |
| Hybrid | dim 544, 4 layers, 16 heads, 128 pairs, top-2, expert 235, additive MoE | 151,005,954 | ~21.1M |
| Hybrid geo-only | same as hybrid, `--router-mode geo` | 151,005,954 | ~21.1M |

The dense trunk of the hybrid alone is 19,085,696 parameters. The additive MoE
branch adds two layers of 128 experts with `f=235`; top-2 makes the active
expert MAC contribution ~2.0M/token, plus ~1.0M for dense router/geo projections.

## How to run

Data: a prepared whole-book byte corpus (`train.bin`, `valid.bin`, `meta.json`,
`split_manifest.json`) from 43 Project Gutenberg train books and 5 held-out
books. The Kaggle datasets used here are `shem42/r30-poincare-nanrouter-books`
(data) and `shem42/r31-code` (this code). Both are private by default; create
your own data dataset with the same layout if needed.

Train (hybrid both, 220 steps × batch 32 × context 256):

```sh
python code/poincare_frontier.py train \
  --data-dir DATA --run-dir run/hybrid_both \
  --steps 220 --context 256 --batch 32 --grad-accum 1 \
  --future2 0.3 --future3 0.15 --future-mode rollout \
  --lr 3e-4 --eval-every 100 --eval-batches 2 --save-every 110 --seed 1337 \
  --dim 544 --layers 4 --heads 16 --pairs 128 --topk 2 \
  --expert-hidden 235 --moe-every 2 --geo 1.0 --balance 0.01 \
  --moe-additive --router-mode both
```

Plain baseline:

```sh
python code/poincare_frontier.py train \
  --data-dir DATA --run-dir run/plain \
  --steps 220 --context 256 --batch 32 --grad-accum 1 \
  --future2 0.3 --future3 0.15 --future-mode rollout \
  --lr 3e-4 --eval-every 100 --eval-batches 2 --save-every 110 --seed 1337 \
  --dim 576 --layers 4 --heads 24 --pairs 128 --topk 2 \
  --expert-hidden 100 --moe-every 5 --geo 0.0 --balance 0.0 \
  --router-mode linear
```

Probe (65,536 positions, specialization, geometry, per-book BPB):

```sh
python code/probe_r31.py --data-dir DATA --run-dir run/hybrid_both \
  --context 256 --batch 8 --batches 32 --book-windows 4 \
  --out results/hybrid151_both.probe.json
```

Generate 200 bytes sampled at temp 0.8 / top-k 30 with repetition metrics:

```sh
python code/gen_r31.py --run-dir run/hybrid_both \
  --prompt-file datasets/r31-code/heldout_prompt.txt \
  --max-new-tokens 200 --temperature 0.8 --topk-sample 30 \
  --out results/hybrid151_both.gen.json
```

## Important engineering note

The 151M additive MoE **must use the default `grouped` dispatch** on T4. The
`gather` path OOMs at batch 32 even though the model fits with grouped dispatch
(~5.25 GiB peak in the benchmark). The first 10-minute attempt used gather and
failed with a 14.5 GiB OOM; the grouped run completed.

## Results

See [`EXPERIMENTS.md`](EXPERIMENTS.md) and
[`docs/R31_10MIN_REPORT.md`](docs/R31_10MIN_REPORT.md). Raw artifacts are in
[`results/`](results/), including the full per-pair 128-bin routing counts in
the probe JSONs.
