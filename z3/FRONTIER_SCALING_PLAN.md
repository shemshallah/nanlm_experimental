# Frontier scaling plan

Goal: the largest model that actually trains on Kaggle, a 6 GB GitHub-runner
CPU test twin, and a growth path for combinatorial `C(256,2+3)` objects rather
than a one-off dense parameter count.

## 1. Presets

| Preset | Total params | Target hardware | Context | Batch | Purpose |
|---|---:|---|---:|---:|---|
| `cpu-6gb` | **38M** | GitHub runner, 6 GB RAM | 384 | 1 | CI, routing/tokenizer/MTP regression |
| `kaggle-mid` | **149M** | 1× T4 16 GB | 512–1024 | 1–2 | fast iteration, ablations |
| `kaggle-max` | **310M** | 1× T4 16 GB | 512–1024 | 1 | main Kaggle run |
| `kaggle-stretch` | 400–500M | 2× T4 / A100 | 512–1024 | 1 | only after memory probe + DP/DDP fix |

The old 6h byte checkpoint was **115M**. The 149M/310M presets are not
regressions; they add per-lag MTP, hierarchical context routers, HD codebooks,
associative-memory experts, and HD time rails.

Measured on Kaggle 1×T4: the **312,069,396-parameter** max preset completed a
5-step probe at batch 1 / ctx 512 in **17.9 s** without OOM. A 16-layer
**322,609,557-parameter** probe completed 3 steps in 18.4 s. The **500,266,485
parameter / 7-MoE-layer** probe completed 2 steps in **32.9 s** without OOM.
That is the current single-T4 ceiling we have actually measured, not guessed.

Measured on the GitHub 6 GB CPU runner: the 38M preset is actually
**39,616,285 params**, completed 5 steps in **21.3 s** with **3.28 GB peak
RSS**, zero non-finite skips. The CI gate is enforced at 5.5 GB.

### Memory rule of thumb

AdamW mixed precision needs roughly **16 bytes/parameter**:

- weights (4) + grads (4) + exp_avg (4) + exp_avg_sq (4).

So 149M ≈ 2.4 GB of optimizer/model state, 310M ≈ 5.0 GB, 500M ≈ 8.0 GB.
The rest is activations, logits, and allocator fragmentation. On a 15 GB T4,
310M at batch 1 / ctx 512 is the first target; ctx 1024 is the next probe.

## 2. Exact commands

### Kaggle max (310M)

```sh
python z3/train_v3.py \
  --ctx 512 --d-model 384 --n-layers 12 --n-heads 12 --n-kv-heads 4 \
  --swiglu 1536 --window 512 --global-every 4 \
  --moe-layers 4 --n-experts 32 --topk 2 --shared-experts 2 \
  --expert-hidden 1536 --no-use-memory \
  --mtp 3 --mtp-block --mtp-block-experts 8 --mtp-block-hidden 1024 \
  --mtp-aux-layers 3,7 --mtp-weight 0.5 \
  --moe-hierarchical --moe-context-router --moe-route-feedback \
  --moe-route-to-attention --moe-hyperdimensional --moe-associative-memory \
  --mtp-hd-time --mtp-branch-norm --mtp-lr-scale 0.5 \
  --batch-size 1 --steps 5000 --eval-every 250 --eval-batches 4 \
  --lr 2e-4 --warmup-steps 100 --lr-horizon wsd --lr-total-steps 20000 \
  --save-every 250 --best-every 250 \
  --corpus data/fineweb_v4c_train.bin --val-corpus data/fineweb_v4c_valid.bin \
  --tokenizer bin --vocab-size 64060 --device cuda --amp \
  --out runs/kaggle_max
```

Start with a 5-step probe and then a 300-step stability run. Do not launch
5,000 steps until MTP loss is finite and the peak memory is recorded.

### CPU 6 GB (38M)

```sh
python z3/train_v3.py \
  --ctx 384 --d-model 256 --n-layers 6 --n-heads 8 --n-kv-heads 2 \
  --swiglu 768 --window 384 --global-every 4 \
  --moe-layers 2 --n-experts 8 --topk 2 --shared-experts 1 \
  --expert-hidden 768 --no-use-memory \
  --mtp 3 --mtp-block --mtp-block-experts 4 --mtp-block-hidden 512 \
  --mtp-aux-layers 2,4 --mtp-weight 0.5 \
  --moe-hd-dim 192 --moe-memory-keys 8 \
  --batch-size 1 --steps 100 --eval-every 25 --eval-batches 1 \
  --lr 1e-4 --warmup-steps 10 --no-ema --device cpu --threads 4 \
  --corpus data/fineweb_v4c_train.bin --val-corpus data/fineweb_v4c_valid.bin \
  --tokenizer bin --vocab-size 64060 --out runs/cpu_6gb
```

CI gate: peak RSS below 5.5 GB, finite loss, at least one router update, and a
checkpoint that reloads.

## 3. `C(256,2+3)` growth path

The diamond engine already has the exact combinatorial indexing:

| Objects | Count |
|---|---:|
| vertices | 256 |
| pairs `C(256,2)` | 32,640 |
| triples `C(256,3)` | 2,763,520 |
| total pairs + triples | 2,796,160 |

Do **not** make every object a full FFN expert. At 1M params each that is
2.8 trillion parameters. Store an HD signature and a small adapter per object;
route top-k objects with factorized scores.

### Object codebook

- 256 learned vertex hypervectors `V ∈ R^(256×d_hd)`.
- Pair code `p_ij = bind(V_i, V_j)` or a low-rank learned pair code.
- Triple code `t_ijk = bind(V_i, bind(V_j, V_k))`.
- Router query = bundle(token hidden, attention context, task tag).
- Score by HD similarity, then top-k objects.
- Each selected object applies a shared low-rank transform plus an object gate.
- Active compute is `O(topk · rank · d_model)`, independent of object count.

Storage at rank 16:

| V | pairs | triples | code params | fp32 storage |
|---:|---:|---:|---:|---:|
| 32 | 496 | 4,960 | 87K | 0.35 MB |
| 64 | 2,016 | 41,664 | 699K | 2.8 MB |
| 128 | 8,128 | 341,376 | 5.6M | 22 MB |
| 256 | 32,640 | 2,763,520 | 44.7M | 179 MB |

This is why the growth path is affordable: it grows the **addressable
combinatorial memory**, not the active FFN width.

### Growth stages

1. `V=32` CPU: establish object router, growth tests, checkpoint migration.
2. `V=64` Kaggle mid: pair/triple HD binding, object-gated low-rank experts.
3. `V=128` Kaggle max: add objects with zero-init gates; replay old objects.
4. `V=256` frontier: full pair/triple address space, sparse top-k active.
5. Tetrahedra `C(256,4)` only as a sparse critic; never dense.

When adding vertices, new pair/triple codes are initialized by binding existing
vertex codes. New object gates start at zero. Old object weights remain valid;
the checkpoint carries `vertex_count` and object-count metadata.

### Why this matters for math/translation

- Pair objects encode `(symbol, operator)` / `(subject, relation)`.
- Triple objects encode `(subject, relation, object)` and `(lhs, op, rhs)`.
- HD binding composes expressions; permutation handles future positions.
- Router context can select algebra/geometry/proof/translation objects without
  forcing all math into one expert set.

## 4. Frontier gates

Every stage must report **bits/byte**, not raw token CE.

| Gate | Target |
|---|---|
| tokenizer v4c | all bytes round-trip, ≥4.5 bytes/token |
| bit/byte layer | exact round-trip |
| 300-step stability | zero non-finite steps; MTP finite |
| 1,000-step v4c | no NaN, loss decreasing, routing utilization logged |
| FineWeb valid | ≤1.30 bits/byte for frontier; ≤1.41 matched baseline first |
| WikiText-103 test | ≤1.55 bits/byte frontier; ≤1.69 first gate |
| predicted MTP | <1.50 normalized; acceptance >67% |
| math translation | exact-match / verifier pass on held-out synthetic + real math |
| CPU CI | peak RSS <5.5 GB, reload works |

The old 0.9538 train loss was **nats/byte**; the old valid score was 1.4148
bits/byte. Do not compare either to v4 token CE (≈7–10 nats/token).

## 5. Execution order

1. Finish v4c byte-complete corpus; re-baseline tokenizer.
2. Kaggle 310M memory probe → 300-step stability → 1,000-step gate.
3. CPU 38M CI run on GitHub runner.
4. Add `C(V,2+3)` object codebook at V=32/64; test growth and routing.
5. Scale V and context; only then 5k–20k-step runs.
6. Math/translation curriculum with tagged synthetic + real data.
7. Frontier run on larger hardware when Kaggle budget allows.

## 6. Current risks

- MTP branch still became non-finite near step 392 before the latest
  branch-norm + MTP-LR fixes; re-test before any long run.
- Single-GPU T4 is the practical limit. Two-GPU DataParallel needs an
  aux-metadata-safe wrapper; DDP is preferable.
- The 310M preset may need ctx 512 and batch 1; ctx 1024 is a separate probe.
- The object codebook must not become a dense 2.8M-way softmax. Use factorized
  scores, candidate sampling, and top-k.
