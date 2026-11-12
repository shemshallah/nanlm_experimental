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
- **MoE dispatch:** gathered top-k weights + batched einsums in
  `moe_vectorized.py`; `v3_model.SparseMoE` no longer loops over experts.
- **Memory:** optional learned memory cross-attention plus a sparse hashed
  4-gram bias. The n-gram store is built from the training split only.

## Test results

All commands were run from the repository root.

```text
z3/test_rope_scaling.py       17 tests  OK  (includes 8k finite forward/backward)
z3/test_moe_vectorized.py      9 tests  OK  (output/gradient parity vs loop)
z3/test_memory_attention.py   17 tests  OK  (causality, collisions, gradients)
z3/test_eval_v3.py             6 tests  OK
z3/test_train_v3_smoke.py      2 tests  OK
z3/test_v3_integration.py      4 tests  OK  (1 optional default-param test)
```

The integration tests verify that YaRN is selected at `ctx=8192`, NTK builds,
the model uses `VectorizedSparseMoE`, and the memory path backpropagates into
its learnable n-gram scale.

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

## Not yet measured

- No 30-minute T4/Colab v3 run has been completed for this integration.
- No long v3 training run, long-context quality evaluation, or scaling run.
- No throughput comparison against the old Python expert loop.
