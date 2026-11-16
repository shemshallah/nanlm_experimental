# FA2 R61 — Bounded CPU index for causal geometric successor memory

Justin Howard-Stanley · Independent Researcher · Santa Cruz, California, USA
2026-11-16 · GitHub master · FA2 only

R54–R60 compare each query geometric edge against all historic edges in every
window. That dense comparison scales with T * horizon * query_edges *
history_edges and is wasteful for large causal memories.

R61 adds an **opt-in CPU indexed engine** for all existing retrieval modes:

- exact pair identity (R54/R55)
- exact pair + observed byte (R59)
- exact pair + observed byte bigram (R59)
- shared 256-point endpoint + observed byte (R60)
- hierarchical exact pair/byte, then shared endpoint/byte (R60)

The index uses deques for exact compound keys and for each endpoint/byte,
with strictly ordered eviction. Memory is bounded to O(horizon * edges).
For each query at position t, the index contains only historical input
positions u<t, and every retrieved successor x[u+1] is already observed
among x[0..t]. It never reads future target x[t+1].

R60's exact-match priority, 1/4-confidence endpoint fallback, wrong-query
address/byte controls, and wrong-successor controls are preserved.
The vectorized engine remains the **default** and is untouched for GPU.

## Numerical parity

CPU unit tests compare histograms, edge coverage, and hit-position counts
with the original vectorized oracle for multiple horizons, two batches,
all five key modes, both successor controls, address/byte permutations,
degenerate starts, and long windows. Tolerance is 1e-6 on the floating
histogram (exact bit-for-bit floating accumulation is not claimed).
Model-level nonzero-rail logit comparison is within 2e-5. Strict state
reload retains the same single-model configuration.

## Real-model continuation

Pass `--retrieval-engine indexed` to the existing R60 checkpoint-backed
training command, alongside `--joint-content --joint-key-mode hierarchical`.
The driver checks before/after logits before training and saves the engine
setting in the model configuration. R61 does NOT introduce new trainable
parameters or alter any inherited weights or optimizer moments.

The critical remaining empirical gate is unchanged: obtain the exact R58
step3185 full model+Adam checkpoint, SHA-256
`63cde9aa1875d9f5e3db78e92678b63057550584752cf20ffe8859c890895dc7`,
and run on SHA-verified NaNROUTER book-disjoint TRAIN/DEV. All next/+2/+3,
non-space, top1/top5, and BPB comparisons must beat the predeclared isolated
controls before a candidate can be promoted or independent TEST opened.

The complete R58 archive exists in ChatGPT Library; the current CPU
execution environment is unavailable and the GitHub runner does not have
the checkpoint. Synthetic parity is NOT a real-book predictive gain.
