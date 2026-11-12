# R31 10-minute pilot: compute-matched hybrid vs dense, router modes, specialization

The 50-minute run was deleted to respect the 10-minute limit. This is the
largest valid run that fits: **~8 minutes wall, 2× T4**, 220 steps × batch 32 ×
context 256 = **1,802,240 positions per arm**. It is a directional pilot, not
the 10M-position target.

- Both/geo kernel: https://www.kaggle.com/code/shem42/r31-10min-v2
- Plain kernel: https://www.kaggle.com/code/shem42/r31-10min
- Data: 43 train / 5 held-out books, 11,079,654 train bytes, same seed/data order
- Eval: 65,536 held-out positions, per-book BPB, 200-byte sampled generation

## 1. Compute-matched comparison

| Arm | Total params | Active MAC/token | CE | BPB | Top-1 |
|---|---:|---:|---:|---:|---:|
| Plain dense | 21,388,032 | ~21.4M | 1.706585 | **2.462082** | 49.55% |
| Hybrid 151M | 151,005,954 | ~21.1M | 1.706278 | **2.461639** | 49.22% |
| Hybrid, geometry-only | 151,005,954 | ~21.1M | 1.724591 | **2.488059** | 48.94% |

**Hybrid − plain = −0.000443 BPB.** At matched active compute, the 151M
hybrid **ties** the 21M dense model. The previous −0.2708 BPB loss was the
parameter-matched control, exactly as predicted. The hybrid has 7.06× the
total parameters at essentially the same active MAC/token (+/-1.2%).

Caveat: 220 steps is very short. The training-eval sample (32K positions)
had plain slightly ahead (CE 1.75098 vs 1.76188); the 65K probe has the hybrid
ahead by 0.0004. This is a tie within noise, not a win yet.

Per-book BPB:

| Held-out book | Plain | Hybrid both | Hybrid geo |
|---|---:|---:|---:|
| The Touchstone | 2.5507 | 2.5506 | 2.5855 |
| Miss Billy Married | 2.4593 | 2.4578 | 2.5306 |
| Pictures from Italy | 2.4065 | 2.4334 | 2.4314 |
| Ozma of Oz | 2.4266 | 2.4422 | 2.4727 |
| The Wallet of Kai Lung | 2.3498 | 2.3460 | 2.3850 |

## 2. Geometry as the router

| Arm | Router | Strength (learned) | Centers norm | Pair distance | Router std | Geo-term std | Geo/router |
|---|---|---:|---:|---:|---:|---:|---:|
| both, layer 1 | linear + geo | 0.977 | 0.464 | 0.655 | 0.647 | 0.119 | 0.183 |
| both, layer 3 | linear + geo | 0.971 | 0.471 | 0.664 | 0.681 | 0.105 | 0.154 |
| geo-only, layer 1 | geometry | 1.044 | 0.453 | 0.642 | — | 0.205 | — |
| geo-only, layer 3 | geometry | 1.020 | 0.465 | 0.656 | — | 0.164 | — |

- Strength moved from 1.0, so it is learnable and not stuck.
- Centers are **not** collapsed to the origin (mean norm 0.45–0.47, min 0.375).
- In the both arm, geometry variance is 15–18% of router variance: linear still
  dominates. In the geo-only arm, geometry is the router and its variance is
  2–3× larger.
- **Geometry-only is 0.0264 BPB worse than both** at this budget. Geometry as
  the sole router is not yet better than linear+geometry for next-byte.

## 3. MoE specialization (balance 0.01)

128 pairs, top-2. Union across both MoE layers:

| Metric | Hybrid both | Hybrid geo |
|---|---:|---:|
| Active pairs | **128 / 128** | 127 / 128 |
| Effective pairs | 108.4 | 90.7 |
| Layer 1 effective | 87.7 | 57.9 |
| Layer 3 effective | 101.1 | 75.7 |
| Max pair share | 5.78% | 8.54% |
| Mean KL to global | 0.876 | 0.904 |
| Max KL | 4.35 | 4.03 |

Lowering balance 0.05 → 0.01 increased concentration (effective 118.4 → 108.4,
max share 2.16% → 5.78%) without killing pairs. Real specialists emerged:

- Pair 118 (layer 1, 454 assignments): **80.8% newline** (byte 10), 9.7% byte
  226, 2.6% quote/period/space.
- Pair 121 (layer 3, 283 assignments): **79.5% byte 128** (UTF-8 continuation
  prefix), 6.7% `l`.
- Pair 0 (layer 3, 591 assignments): 13.0% `I`, 11.3% `T`, 9.5% byte 226,
  9.0% `A`, 8.0% `H` — a capital-letter/punctuation pair.
- Pair 21 (layer 3, 634 assignments): **55.0% newline, 33.0% space**.
- Pair 104 (layer 1, 120 assignments): **90% `h`** (small sample).

Some very high-KL pairs have only 1–4 assignments and are noise; the counts
above are the meaningful ones. Specialization is emerging at 1.8M positions,
not fully formed.

## 4. Generation (200 bytes, held-out prompt, temp 0.8 / top-k 30)

- **Hybrid both:** ` the from in the repeced from their open through the large of the consident of Mermor, while first be my when he eness as should climped and this dire any the swarm of the brokes, “Opewiy somether,`
- **Hybrid geo:** ` the opperning do, Holded and them undipht and of thee had men for for the torriate. Bearing the buelt hour to a once officience midding and tootal, and he had never being when enward daxey her door,`
- **Plain:** ` the old to liearing, the concessared in thought the direct of the gring of the peet streation. I would of the remating as to the but head and wife, and we somed, and up an lay lold of will her mothin`

4-gram repetition: hybrid both **9.74%**, hybrid geo **11.17%**, plain **15.74%**.
Longest repeated substring: 8 chars both, 6 geo, 8 plain. No arm is fluent at
1.8M positions; hybrid is less repetitive than plain. BPB, not fluency, is the
metric at this stage.

## What this changes

- The compute-matched comparison is the right one. At 1.8M positions the
  151M/21M-active hybrid ties the 21M dense baseline; the earlier 0.27 BPB loss
  was a protocol artifact. This is the first positive signal for the hybrid.
- Geometry-only is slightly worse than linear+geometry at this budget. The
  next-byte task still does not reward Poincaré geometry as the sole router.
- MoE is not collapsed and is specializing (newlines, capitals, UTF-8
  prefixes), but 220 steps is too short to conclude.
- The decisive run is still 10M positions. The 151M hybrid needs ~39 min for
  that on T4 (benchmark: 4.3k tok/s). At 10M positions, re-test plain vs hybrid,
  add the linear-only arm, and keep 65K+ eval.
- If geometry-only still ties/loses at 10M, pivot it to hierarchical routing,
  long-context retrieval, or memory in the ball — not next-byte.
