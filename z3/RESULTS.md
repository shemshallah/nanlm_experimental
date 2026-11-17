# v3 measured results

These are short plumbing/smoke results. They validate the code path, not model
quality and not frontier-scale training.

## Integrated architecture

- **Params:** ~316.4M for the `train_v3.py` default (64,060 vocab);
  ~316.4M for the `V3Config` dataclass default (64,000 vocab).  The
  historical 286.5M figure came from an earlier flag set.
- **Active/token:** roughly 70–80M (estimate from routed top-2 + shared experts,
  not a measured hardware counter).
- **Feedback bus:** token-concept residual, route/concept/hidden feedback,
  object depth carry, memory→MTP, and MTP→MTP; all gates start closed
  (`tanh(0)=0`) and are reported in `aux["feedback"]`.
- **Long context:** RoPE cache covers 8,192 positions. `rope_scaling="auto"`
  selects standard RoPE at 1k and YaRN above the 1k original window; `ntk` is
  also available explicitly.
- **MoE dispatch:** grouped, row-chunked top-k dispatch in
  `moe_vectorized.py`; `v3_model.SparseMoE` no longer loops over experts and
  no longer materializes per-token expert weights.
- **Memory:** optional learned memory cross-attention plus a sparse hashed
  4-gram bias. The n-gram store is built from the training split only.

## Test results

All commands were run from the repository root.

```text
z3/test_rope_scaling.py       17 tests  OK  (includes 8k finite forward/backward)
z3/test_moe_vectorized.py     12 tests  OK  (output/gradient parity vs loop)
z3/test_memory_attention.py   17 tests  OK  (causality, collisions, gradients)
z3/test_eval_v3.py             6 tests  OK
z3/test_train_v3_smoke.py      2 tests  OK
z3/test_v3_integration.py      8 tests  OK
```

## MoE memory check

The first T4 attempt failed before any training step. The old dispatch gathered
`(tokens × topk, d_model, expert_hidden)` weights; for the notebook profile
(4,096 tokens, 32 experts, top-2, hidden 1,024) that is about **8.6 GB per
projection**. It is replaced with grouped, row-chunked dispatch.

Measured on CPU for the same MoE shape (4×1,024 tokens, d_model 256, 32 routed
experts, top-2, 2 shared, hidden 1,024, no grad):

```text
output        (4, 1024, 256), finite
wall          10.5 s
peak RSS      351 MB (MoE-only)
```

A full 128.6M-parameter v3 forward with this profile (batch 1, 256 tokens)
also ran finite and peaked at 1.09 GB RSS, but the complete T4 step has not
been rerun here.

The integration tests verify that YaRN is selected at `ctx=8192`, NTK builds,
the model uses `VectorizedSparseMoE`, and the memory path backpropagates into
its learnable n-gram scale.

## Kaggle 10-minute v3 GPU run

Measured on a Kaggle T4 with the v3 trainer, mixed 16-domain byte/token corpus
(`r32-multidomain-16domains`), 1,024 context, 112,375,040 parameters:

```text
steps              1,092
wall               603.8 s
tokens seen        4,472,832
eval before        8.1894 bits/token, top-1 0.60%, top-5 1.58%
best (step 1050)   3.3723 bits/token, top-1 38.75%, top-5 68.41%
final              3.3723 bits/token, top-1 38.72%, top-5 68.40%
MTP loss           2.843
```

The run used `--lr-horizon time` with `--max-seconds 600`; the cosine schedule
therefore reached ~8e-11 by step 1,092. That is why it flattened after roughly
1,000 steps. `--lr-horizon auto` now means **step-based**, so a preempted or
short run resumes on the same curve. `--lr-horizon time` is explicit opt-in.

This is a 1,000-step smoke, not a 1M-step run and not a frontier-scale result.
The step-1050 checkpoint is the best checkpoint from this run.

## GQA 10-minute run

Same 16-domain corpus and 1,000-step step-based cosine, with 8 query heads and
2 KV heads:

```text
params             111,588,608  (0.79M fewer than MHA)
steps              1,000
wall               558.2 s
tokens seen        4,096,000
eval before        8.2416 bits/token, top-1 0.10%, top-5 0.93%
final              3.5041 bits/token, top-1 36.06%, top-5 67.19%
MTP loss           2.905
```

GQA is 0.13 bits/token worse than the MHA 1,000-step run at this scale and
step budget. That is an expected capacity reduction, not evidence that GQA is
better for quality; it is an inference/memory-efficiency feature.

## Gated attention + WSD 10-minute run

Same 16-domain corpus and 1,000-step budget, with the sigmoid attention gate
and warmup-stable-decay schedule (`--attn-gate --lr-horizon wsd`):

```text
params             112,899,328
steps              1,000
wall               570.7 s
tokens seen        4,096,000
eval before        8.1654 bits/token, top-1 0.38%, top-5 1.99%
final              2.9823 bits/token, top-1 45.30%, top-5 73.41%
MTP loss           2.652
LR at end          5.14e-08 (WSD decay)
```

This is the best measured 1,000-step result so far: **0.39 bits/token better**
and **6.55 top-1 points better** than the 1,000-step MHA/cosine run. It is
still a short smoke, not a long-run or frontier-scale claim.

## CPU random smoke

Tiny `--random-smoke` run, 3 steps, 2 MTP heads, CPU:

```text
params       328,576
tokens       3,072
CE           5.5640
total loss   7.3244
MTP loss     5.7358
load balance 3.3836
router z     5.7580
wall         8.09 s
```

## CPU memory text smoke

Tiny byte-tokenizer run on a repeated 64 KB synthetic corpus, 3 steps,
`--use-memory --memory-size 4096`, fixed validation windows:

```text
params            169,666
memory entries        43
memory windows    20,000
memory collisions      0
eval before       7.9771 bits/token, top-1 2.38%, top-5 6.75%
final             7.8693 bits/token, top-1 3.57%, top-5 10.71%
wall                 1.13 s
```

This is a plumbing test. The tiny validation set (252 tokens) and three steps
are not evidence of a useful language model.

## CPU real-corpus byte smoke

A 2,000,000-byte prefix of `nanlmv3/data/corpus_100m.txt`, byte tokenizer,
2 steps, tiny 152,192-parameter model:

```text
eval before   8.0413 bits/token, top-1 0.39%, top-5 0.98%
final         7.9760 bits/token, top-1 0.39%, top-5 1.57%
wall          1.97 s
```

Again, this only verifies the real-corpus/data/trainer path. It is not a
learning result.

## GitHub 115M CPU wiring smoke (2026-11-17)

Run: <https://github.com/shemshallah/nanlm_experimental/actions/runs/44019140042>
Commit: `f309e6a`

This is a one-step wiring smoke, not a quality run.  It builds a tiny
document-disjoint corpus, constructs the fixed `115m` preset, runs one CPU
optimizer step at `ctx=8`, evaluates the three held-out synthetic domains, and
validates the artifacts.

| Metric | Value |
|---|---:|
| Parameters | 115,238,237 |
| Steps | 1 |
| Context / batch | 8 / 1 |
| Non-finite skips | 0 |
| Macro BPB | 4.0381 |
| Macro BPB 95% interval | 3.9725–4.1248 |
| MoE layout | 4 branches × 3 experts, top-2, no dead branch |
| Validation | `ok=true`, 0 errors, 2 warnings |

The two warnings are expected at initialization and are not failures:

- `feedback_path_opened`: every learned feedback gate is
  `tanh(0) * progress = 0`; the preset ramps `progress` over 500 steps, so a
  one-step run cannot open the bus.
- `object_gate_open`: the object gate starts at `sigmoid(-4) = 0.017986`,
  below the 0.05 promotion threshold.

The earlier failed dispatch (`44018696524`, commit `8d19da5`) was an import
path issue fixed by adding `z3` to `PYTHONPATH` in `f309e6a`.  A companion
38M CPU 6 GB run (`44018683034`) completed 5 steps at **3.787 GB peak RSS**
with **40,918,706 parameters** and zero non-finite skips.

Do not read the 4.04 BPB as a model result: the test set is three short
synthetic documents and the model has seen one optimizer step.  Use a GPU or
self-hosted runner for the 10k–20k-step run in `TRAIN_TEST_115M.md`.

### 300-step CPU feedback stability gate (2026-11-17)

Run: <https://github.com/shemshallah/nanlm_experimental/actions/runs/44141483908>
Preset: `115m` · 300 steps · `ctx=8` · batch 1 · feedback warmup 50 ·
`best_every=50`

| Metric | Value |
|---|---:|
| Parameters | 115,238,237 |
| Steps completed | 300 |
| Non-finite skips | 0 |
| Training wall | 273.0 s |
| Best eval | step 250, loss 2.6850 |
| Best eval BPB | 0.8747 (synthetic template corpus) |
| Final eval BPB | 0.9312 (synthetic template corpus) |
| Held-out macro BPB | 0.6820 (300 synthetic tokens) |
| Validation | `ok=true`, 0 errors, 1 warning |

The feedback bus opened: memory gate **0.00693**, route-hidden gate
**-0.00704**, route-concept gate **-0.00108**, token gate **+0.000323**,
MTP-carry gate **-5.2e-06**, object-depth gate **-2.1e-05**, progress
**1.0**.  The only validation warning is the object gate:
**0.01803 < 0.05**, effectively still closed at 300 steps.

The sub-1.0 BPB is **not** a language-model result.  The corpus is the tiny
24-document-per-domain template built by the CI smoke; 300 steps fit that
template.  Real 115M references are 1.944 BPB on FineWeb-Edu (byte/Blackwell)
and 2.663 BPB on WikiText-103 (subword).  This gate only proves that the
115M optimizer, feedback bus, checkpoint, evaluation, and validation paths
work on a CPU runner.

## Not yet measured

- No 30-minute T4/Colab v3 run has been completed for this integration.
- No long v3 training run, long-context quality evaluation, or scaling run.
- GQA has only a 1,000-step comparison against MHA; no scale comparison.
- The attention gate and WSD schedule have one 1,000-step comparison, not a
  long-run or multi-seed result.
- No throughput comparison against the old Python expert loop.
