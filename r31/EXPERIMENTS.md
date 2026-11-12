# Experiment log

All runs are byte-level next-byte models on Project Gutenberg books. “BPB” is
bits per byte on held-out books; lower is better. Unless noted, evaluation is
65,536 held-out positions and the train/valid split is at book boundaries.

---

## 2026-11-12 — r31 10-minute compute-matched pilot (latest)

**Question:** does the 151M-total / ~21M-active hybrid beat or tie the 21M
dense model when **active compute**, not parameter count, is matched? And does
geometry work when it is the router rather than a penalty?

**Hardware:** 2× Tesla T4 (14.56 GiB each). **Wall time:** ~8 minutes.
**Kernel:** https://www.kaggle.com/code/shem42/r31-10min-v2
**Plain reference:** https://www.kaggle.com/code/shem42/r31-10min

### Protocol

- Corpus: 43 train books / 5 held-out books, 11,079,654 train bytes.
- 220 optimizer steps × batch 32 × context 256 = **1,802,240 positions per arm**.
- Seed 1337, identical sampler stream, fresh initialization.
- Eval: 65,536 held-out positions; per-book BPB with 4 windows/book.
- Balance loss 0.01 (down from 0.05). Geometry strength learnable from 1.0.

| Arm | Total params | Active MAC/token (est.) | Config |
|---|---:|---:|---|
| Plain | 21,388,032 | ~21.4M | dim 576, 4 layers, 24 heads, SwiGLU, no MoE |
| Hybrid both | 151,005,954 | ~21.1M | dim 544, 4 layers, 16 heads, 128 pairs, top-2, expert 235, additive MoE, linear+geo router |
| Hybrid geo-only | 151,005,954 | ~21.1M | same, `--router-mode geo` |

### Main result

| Arm | CE (nats/byte) | BPB | Top-1 |
|---|---:|---:|---:|
| Plain | 1.706585 | 2.462082 | 49.55% |
| Hybrid both | 1.706278 | **2.461639** | 49.22% |
| Hybrid geo-only | 1.724591 | 2.488059 | 48.94% |

- **Hybrid − plain = −0.000443 BPB. A tie.**
- **Geo-only − both = +0.026420 BPB.** Geometry as the sole router is worse at
  this budget.
- The earlier parameter-matched run had hybrid −0.2708 BPB. Matching active
  compute removed that loss. The 151M model uses 7.06× the total parameters at
  essentially the same active MAC/token (+/−1.2%).

Training-eval note: on the 32K-position validation sample used during training,
plain was slightly ahead at step 200 (CE 1.75098 vs 1.76188); the 65K probe has
the hybrid ahead by 0.0004. This is a tie within noise, not a demonstrated win.

Per-book BPB:

| Book | Plain | Hybrid both | Hybrid geo |
|---|---:|---:|---:|
| The Touchstone | 2.5507 | 2.5506 | 2.5855 |
| Miss Billy Married | 2.4593 | 2.4578 | 2.5306 |
| Pictures from Italy | 2.4065 | 2.4334 | 2.4314 |
| Ozma of Oz | 2.4266 | 2.4422 | 2.4727 |
| The Wallet of Kai Lung | 2.3498 | 2.3460 | 2.3850 |

### Router and geometry diagnostics

| Arm / layer | Strength | Centers norm (mean) | Pair distance | Router std | Geo-term std | Geo/router |
|---|---:|---:|---:|---:|---:|---:|
| both, layer 1 | 0.977 | 0.464 | 0.655 | 0.647 | 0.119 | 0.183 |
| both, layer 3 | 0.971 | 0.471 | 0.664 | 0.681 | 0.105 | 0.154 |
| geo-only, layer 1 | 1.044 | 0.453 | 0.642 | — | 0.205 | — |
| geo-only, layer 3 | 1.020 | 0.465 | 0.656 | — | 0.164 | — |

- Strength moved from 1.0, so it learned.
- Centers did not collapse to the origin: mean norm 0.45–0.47, min 0.375.
- In `both`, geometry variance is only 15–18% of router variance: linear still
  dominates. In `geo`, geometry is the router and is 2–3× larger, but BPB is
  worse.
- Conclusion for next-byte: geometry-only does not beat linear+geometry. This
  supports moving geometry to a hierarchical/retrieval task rather than adding
  it as a next-byte penalty.

### MoE utilization and specialization (balance 0.01)

| Metric | Hybrid both | Hybrid geo |
|---|---:|---:|
| Active pairs (union) | **128 / 128** | 127 / 128 |
| Effective pairs (union) | 108.4 | 90.7 |
| Layer 1 effective | 87.7 | 57.9 |
| Layer 3 effective | 101.1 | 75.7 |
| Max pair share | 5.78% | 8.54% |
| Mean KL to global | 0.876 | 0.904 |
| Max KL | 4.35 | 4.03 |

Lowering balance 0.05 → 0.01 increased concentration (effective 118.4 → 108.4,
max share 2.16% → 5.78%) without killing pairs. Meaningful specialists:

| Pair | Layer | Assignments | Top next bytes |
|---|---:|---:|---|
| 118 | 1 | 454 | **80.8% newline** (byte 10), 9.7% byte 226, 2.6% quote/period/space |
| 121 | 3 | 283 | **79.5% byte 128** (UTF-8 continuation), 6.7% `l`, 3.5% `h` |
| 0 | 3 | 591 | 13.0% `I`, 11.3% `T`, 9.5% byte 226, 9.0% `A`, 8.0% `H` |
| 21 | 3 | 634 | **55.0% newline, 33.0% space** |
| 104 | 1 | 120 | **90% `h`** |

Some very high-KL pairs have 1–4 assignments and are noise; the counts above
are the meaningful ones. Specialization is emerging at 1.8M positions.

### Generation

200 bytes from a held-out book prompt, sampled at temperature 0.8 / top-k 30
(no greedy decoding in this experiment):

- **Hybrid both:** ` the from in the repeced from their open through the large of the consident of Mermor, while first be my when he eness as should climped and this dire any the swarm of the brokes, “Opewiy somether,`
- **Hybrid geo:** ` the opperning do, Holded and them undipht and of thee had men for for the torriate. Bearing the buelt hour to a once officience midding and tootal, and he had never being when enward daxey her door,`
- **Plain:** ` the old to liearing, the concessared in thought the direct of the gring of the peet streation. I would of the remating as to the but head and wife, and we somed, and up an lay lold of will her mothin`

4-gram repetition: hybrid both **9.74%**, hybrid geo **11.17%**, plain
**15.74%**. Longest repeated substring: 8 / 6 / 8 bytes. No arm is fluent at
1.8M positions; hybrid is less repetitive than plain. BPB is the metric.

### Caveats

- 220 steps and 1.8M positions is a pilot. The target was 5k–10k steps and
  10M+ positions. The 151M hybrid benchmarks at ~4.3k tok/s, so 10M positions
  takes ~39 minutes for that arm on T4.
- A 50-minute compute-matched kernel was launched and then deleted to respect a
  10-minute limit. The 10M-position comparison remains to be run.
- Plain was measured in a separate kernel run (v1) with the same data, seed,
  steps, batch and context. The hybrid's first attempt in v1 OOM'd; the v2
  grouped-dispatch run is the one compared here.
- Single seed, 65,536 eval positions, 4 windows/book for per-book BPB. No
  formal confidence intervals; the 0.0004 BPB difference is not a win.

### Engineering failure/fix

The first 10-minute attempt (`r31-10min`) used `--dispatch-mode gather` for the
151M additive MoE. It OOM'd at batch 32 (14.53 GiB in use) in both router arms.
The 5-step benchmark used the default `grouped` dispatch and fit in ~5.25 GiB.
Rerunning with `grouped` completed normally. **Use grouped dispatch for the
151M/top-2 model on T4.**

### Artifacts

- `results/r31_10min_v2_summary.json` — full two-arm summary
- `results/hybrid151_both.probe.json`, `results/hybrid151_geo.probe.json` —
  full per-pair 128-bin counts, specialization, geometry
- `results/hybrid151_both.gen.json`, `results/hybrid151_geo.gen.json`
- `results/r31_10min_v1_summary.json` — plain reference + first hybrid OOM
- `results/plain.probe.json`, `results/plain.gen.json`, `results/plain.train.log`
- `results/bench.json` — throughput/memory benchmark
- `docs/R31_10MIN_REPORT.md`

---

## 2026-11-12 — r31 throughput benchmark

Kernel: https://www.kaggle.com/code/shem42/r31-bench

T4, 5 timed optimizer steps per cell, context 256, grouped dispatch:

| Model | Batch | s/step | Tokens/s | Peak VRAM |
|---|---:|---:|---:|---:|
| Plain 21M | 8 | 0.1052 | 19,466 | 0.80 GiB |
| Plain 21M | 16 | 0.2095 | 19,551 | 1.33 GiB |
| Plain 21M | 32 | 0.4042 | 20,267 | 2.39 GiB |
| Hybrid 151M | 8 | 1.4056 | 1,457 | 3.18 GiB |
| Hybrid 151M | 16 | 1.6062 | 2,550 | 3.88 GiB |
| Hybrid 151M | 32 | 1.9156 | 4,277 | 5.25 GiB |
| Control 21M / 2M active | 8 | 0.4162 | 4,920 | 0.58 GiB |
| Control 21M / 2M active | 16 | 0.4342 | 9,434 | 0.78 GiB |
| Control 21M / 2M active | 32 | 0.4968 | 16,490 | 1.19 GiB |

The benchmark used grouped dispatch. The 151M model is 4.7× slower than plain
at batch 32 despite similar active MAC, due to 7× total parameters and Adam
state traffic.

---

## 2026-11-12 — R30 parameter-matched ablation (negative result)

Kernel: https://www.kaggle.com/code/shem42/r30-ablation-gpu

- 600 steps × batch 8 × context 256 = 1.23M positions.
- Plain 21.39M vs hybrid 21.41M total (parameter-matched, but hybrid active
  compute only ~2M/token).
- 128 pairs, top-2, balance 0.05.

| Arm | BPB | Top-1 |
|---|---:|---:|
| Hybrid | 2.6082 | 47.00% |
| Plain | 2.3374 | 51.70% |
| Hybrid geo0 | 2.6060 | 46.71% |

- Hybrid − plain = **+0.2708 BPB** (hybrid worse).
- Hybrid − geo0 = **+0.0022 BPB** (geometry flat).
- MoE utilization was 128/128 active, effective 118.4, max share 2.16% —
  balanced but perhaps too uniform.
- Generation: hybrid greedy collapsed to “the said the said…”, sampled output
  was word salad; plain was also word salad.
- This run motivated the move to compute matching, lower balance, and
  geometry-only routing.

Full report: `docs/R30_ABLATION_REPORT.md`.

---

## 2026-11-11 — R30 fresh 5–10 minute run

Kernel: https://www.kaggle.com/code/shem42/r30-poincare-gpu

- Source-only R30 ZIP; no canonical step-2801 checkpoint, so fresh init.
- 32 pairs, 6 layers, dim 192, 1,110 steps in 420 s on one T4.
- 4,096-position held-out eval: CE 1.4366, BPB 2.0725, top-1 57.45%.
- Local 2,048-position recheck: CE 1.4566, BPB 2.1015, top-1 56.15%.
- Generation from “The night was” was not fluent.
- The high top-1 did **not** imply fluent generation; the 4K eval was noisy.
  This motivated the 65K eval in r31.
