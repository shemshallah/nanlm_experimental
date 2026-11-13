# v3 measured results

These are short plumbing/smoke results. They validate the code path, not model
quality and not frontier-scale training.

## Integrated architecture

- **Params:** 286,517,376 total for the default `V3Config`.
- **Active/token:** roughly 70–80M (estimate from routed top-2 + shared experts,
  not a measured hardware counter).
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
z3/test_v3_integration.py      7 tests  OK
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

## Not yet measured

- No 30-minute T4/Colab v3 run has been completed for this integration.
- No long v3 training run, long-context quality evaluation, or scaling run.
- GQA has shape/gradient tests, but no quality comparison against MHA at scale.
- No throughput comparison against the old Python expert loop.
