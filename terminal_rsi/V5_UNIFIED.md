# Terminal-RSI v5 unified graph

One graph for CPU / T4 / Blackwell. Blackwell is an execution profile, not a
different model.  The inherited v103 core can be warm-started from the verified
2 BPB parent; the 100M upgrade branch carries the hierarchy, concept memory,
unit heads and RSI target.

## Model

- width 704, 12 layers, 11 heads, context 256, two recursive passes
- ~108.5M total parameters (v103 core 2.78M + v5 upgrade branch)
- fused 8x GLU (`gate_value`) in every block
- 256-expert **rank-one** top-2 MoE in layers 1, 4, 7, 10 (~1.04M each)
- adaptive-depth head that blends pass 1 and pass 2; one-pass generation path
- zero-initialized low-rank **fast-weight** output head (`rank=8`) for
  weight-space RSI branches
- hierarchy bus at width 176: byte -> subword -> word -> sentence,
  sentence -> word -> subword -> byte, lateral word/subword -> byte logits
- boundary-aware pooling: predicted subword/word/sentence boundaries gate the
  fixed causal windows `(4, 16, 64)`
- **paired concept memory**: 65,536 = `2 * 32^3` = `2*C(256,2) + 256`
  ordered pair concepts; bottom/subword state routes and writes, top/sentence
  state reads through a zero gate
- real unit heads: BPE-token 4,096, whitespace-word 4,096, sentence-onset 1,024
- full 256-way **distillation** from the inherited v103 parent into the 100M
  branch; the parent is detached
- unit-head and boundary losses are causal; no future byte enters a label

## 32^3 closure

```text
C(256,2) = 32,640 unordered mirror pairs
2 * 32,640 + 256 = 65,536 = 256^2 = 2 * 32^3
bottom 32^3 = 32,640 pairs + 128 diagonal points
top    32^3 = 32,640 mirrors + the other 128 diagonal points
```

## Milestone M1 (current): unit heads must lower held-out BPB

Unit heads are not allowed to remain standalone classifiers.  The current
milestone is:

1. **current-token / current-word heads** trained on the prefix (the unit
   containing the next byte, not only the next unit)
2. a **zero-gated unit→byte decoder** from those heads into the final byte
   logits
3. a direct unit-byte CE that trains the decoder even while the mixing gate is
   closed
4. held-out BPB reported with the unit decoder **on and off**, plus the unit
   gate value; success means `BPB_on < BPB_off`
5. `no_unit_decoder` ablation and a receipt field with the delta

Standalone unit top-k remains a diagnostic, not the milestone.

## RSI v5

- parallel replicas: neutral + K candidate clones, one CUDA stream per clone
- action space adds three hierarchy gate deltas (bottom-up, top-down, lateral)
  to the v4 LR/clip/weight-decay/residual-gate/refine/temperature/batch space
- branch training includes the fast-weight head, so accepted branches persist
  weight-space deltas
- paired probe shards; shard positions 128 gives SE ~4e-5
- default `min_gain=0.001` (the old 0.01 rejected every precise branch)
- hierarchy gates start at zero; the first auxiliary byte-CE step gives them a
  gradient, so the loops can open without a hand-written schedule

## Generation

- bytes are always sampled; words/sentences are latent
- `hierarchical_generate.generate_bytes` supports:
  - `word_vocab` + `word_onset_weight`: at a predicted word boundary, the word
    head rescoring the first byte of the next word
  - `fast=True`: one-pass KV-cache prefill + per-byte cached step, with a
    rolling hierarchy buffer (approximate one-pass path; default is the exact
    two-pass path)
- KV cache stores per-layer K/V and rolling basis/state buffers

## Speed

- fused GLU removes one kernel launch per block
- `--compile-upgrade` tries `torch.compile` on the upgrade branch; the helper
  removes the repo root from `sys.path` during the first trace so the local
  `z3/` notebook directory cannot shadow torch-dynamo's `z3` dependency
- eager fallback on any compile error
- KV-cache generation is one pass; the two-pass path remains the reference

## Ablation

`--ablation` variants: `full`, `no_loops`, `no_bottom_up`, `no_top_down`,
`no_lateral`, `no_pair`, `no_units`, `no_boundary_pooling`.

`python -m terminal_rsi.run_v5_ablation --output-dir ... --legacy-checkpoint ...`
runs short parent-warm-start variants and writes `ABLATION.md` +
`ablation_summary.json`.

## Key flags

| flag | purpose |
|---|---|
| `--legacy-checkpoint` + `--allow-parent-mismatch` | warm-start the v103 core |
| `--legacy-core-lr-scale 0.1` | conservative parent fine-tune |
| `--unit-loss-weight 0.1` | token/word/sentence heads |
| `--distill-weight 0.5` | full parent KL |
| `--distill-samples 8` | teacher samples per microbatch |
| `--moe-loss-weight 0.01` | load balancing |
| `--boundary-loss-weight 0.02` | boundary heads |
| `--compile-upgrade` | optional torch.compile |
| `--fast-generation` | one-pass KV-cache generator |
| `--generation-word-onset-weight 0.3` | word-boundary onset prior |
| `--rsi-v4-shard-positions 128` | precise RSI signal |
| `--rsi-v4-min-gain 0.001` | accept measured gains |

## Test results (short probes)

- `test_v5_unified.py`: **11/11 pass**, including per-sample 256-way parent
  distillation normalization and KV-cache equivalence against a one-pass full
  sequence.
- `test_blackwell_preflight.py`: 10/10 pass.
- `test_rsi_v4.py`: 13/13 pass.
- `test_hierarchical_bus.py`: 9/9 pass.

GPU short probes:

- 60 s full-v5 smoke: 5 updates, **72.78 GiB allocated / 93.89 GiB reserved**,
  fast cached generation `"social\nmit chair"`.
- 45 s hierarchical + unit eval: **1.808 BPB**, byte top1 60.9%, top5 88.3%.
- `torch.compile`: succeeds; reserved VRAM **67.3 GiB**; no short-horizon
  speedup yet.
- ablation `full` / `no_loops` / `no_pair`: identical within 2e-5 BPB after
  four updates because the hierarchy gates are still ~10^-3. A meaningful
  ablation needs a longer distillation run or a gate-warmup schedule.
- unit-head DEV metrics are still near random after five updates
  (token top5 9.4%, word 1.6%); the 4k-way heads need a real training budget.

Full receipt: `terminal_rsi/reports/blackwell_v5_short_tests/RUN_SUMMARY.md`.

## Reliability fixes and five-minute receipts (2026-11-18)

Root causes found by tracing the full backward graph and RSI branch path:

- **Word-prefix prior `log(0)` backward**: `torch.where(mask, probabilities.log(), -30)`
  evaluated `log(0)` in the unselected branch. With a zero gate the upstream
  gradient is `0`, and `LogBackward0` produced `0/0 = NaN`, poisoning every
  upgrade parameter. The prior now logs a strictly positive tensor.
- **Prior calibration**: the raw scatter counts were unnormalized and used a
  `-30` floor, producing ±30 logits. A tiny gate (0.054) then cost **0.217 BPB**
  after RSI-v4 branch training. The prior is now normalized, mixed with 5%
  uniform, and saturates at a **0.05 trust region**.
- **Pair lattice address collision**: the top 128 special points used
  `32768 + (left - 128)`, overlapping 128 mirrored-pair rows. They now start at
  `65408`; `pair_addresses()` covers all 65,536 rows exactly once
  (`test_pair_addresses_cover_entire_bank`).
- **Detached MoE load balance**: `last_load_balance` was computed under
  `no_grad`, so `--moe-loss-weight` had no gradient. The router-balance term is
  now differentiable and includes the top/bottom hierarchy MoEs.
- **RSI deepcopy crash**: the live MoE balance tensor broke
  `copy.deepcopy(model)` in branch capture. `SparseRankOneMoE.__getstate__`
  drops it; clones rebuild it on the next forward.
- Fast-adapter output is NaN-guarded, clamped to ±10, and reset when its
  weights become non-finite.

### Five-minute full-v5 smoke (RSI off, parent frozen)

`terminal_rsi_v5_final5c`, 2M pilot, batch 1024 / microbatch 256:

| Metric | Value |
|---|---:|
| Updates / non-finite skips | **41 / 0** |
| Parent DEV BPB | 2.453140 → **2.450072** (**−0.003067**) |
| Unit→byte decoder on/off | **+0.000442** (helps) |
| Word-prefix prior on/off | **+0.000114** (helps) |
| Byte top-5 / token top-5 / word top-5 | 87.1% / 89.0% / 40.4% |
| Unit-head DEV word top-1 / top-5 | 21.7% / 37.0% |
| Peak VRAM | **27.38 GiB allocated / 36.37 GiB reserved** |
| Byte-driven generation, prompt `"The "` | `"tone in the hoof"` |

### RSI-v4 branch fit

`terminal_rsi_v5_v4fit2`, short probe (120 s training, 4 cycles/tick):

| Metric | Value |
|---|---:|
| Updates / cycles / branch updates | 3 / 12 / **120** |
| Accepted interventions | 1 |
| DEV BPB | 2.497848 → **2.497619** (**−0.000229**) |
| Unit→byte decoder on/off | −0.000592 |
| Word-prefix prior on/off | **+0.000021** (bounded; was −0.217) |
| Peak VRAM | 29.62 GiB allocated / 51.85 GiB reserved |

The v4 run shares the GPU with another training job; the low-memory recipe
(batch 1024 / microbatch 256) is the default for concurrent smokes.
