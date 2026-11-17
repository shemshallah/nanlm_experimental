# Hierarchical feedback audit: Terminal-RSI vs Diamond

## Implemented hierarchy in the 100M upgrade branch

The unified 100M model now contains `CausalHierarchy` in
`terminal_rsi/hierarchical_bus.py`:

1. **Bottom-up sweep** `byte -> subword -> word -> sentence` via causal
   windows `(4, 16, 64)` bytes, vectorized with `cumsum`.
2. **Top-down sweep** `sentence -> word -> subword -> byte`, each edge a
   separate zero-initialized gated residual.
3. **Lateral priors** `subword -> byte logits` and `word -> byte logits`,
   independent zero-gated paths.
4. **Boundary heads** for subword/word/sentence boundaries, trained from
   observed bytes only (`. ! ?`, whitespace/punctuation, case/digit changes).
5. **Byte-driven generation** in `terminal_rsi/hierarchical_generate.py`:
   sample one byte, append it to the causal context, re-run the bus.  Words,
   sentences and paragraphs are latent synthesized states, never sampled
   tokens.

The gates start at zero but the lateral/read projections start small and
nonzero, so the first auxiliary byte-CE step gives the gates a gradient.  In
the 60 s Blackwell probe the gates moved to bottom-up `0.0042`, top-down
`-0.0047`, lateral `±0.005` while the byte BPB fell from 8.188 to 5.37.

The old cycles (`byte->byte` recursive passes, v103 online feedback, v4
controller) remain.  The requested `word->byte->subword->word...` chain is a
directed cycle; the implementation is its causal DAG equivalent above.

## 32³ pair/palindrome geometry — the count closes

The apparent mismatch was treating the diagonal as 256 points in *one* lattice.
The object is the full ordered 256×256 pair matrix:

```text
C(256,2) = 32,640 unordered mirror pairs
2 * C(256,2) + 256 = 65,536 = 256² = 2 * 32,768 = 2 * 32³
```

So the intended **32³ + 32³ paired point cloud** is exact:

- **bottom 32³ lattice**: 32,640 mirror pairs + 128 palindromic points
- **top 32³ lattice**: the 32,640 mirrored pairs + the other 128 palindromic points
- total: 65,536 concepts = the full ordered pair matrix

The 128 "missing" points are the per-lattice half of the 256-point diagonal.
The 256 bottom specialists write concepts into the bottom lattice; the 256 top
specialists read the mirrored concepts from the top lattice and project them
back into the word/sentence states. The 256 extension heads are the top-2 MoE
experts over those concept slots.

This is deliberately **not** implemented yet. The next geometry step is a
65,536-entry, low-dimensional pair-concept bank with a top-2 router, not a
dense 256-expert MoE at width 704. The bank is cheap (~1M parameters at
concept_dim=16); the router and zero-gated read/write path are the only new
graph edges.


## What Diamond actually implements

From `diamond/diamond_engine.py`:

- `N_PAIR = 32,640 = C(256,2)`; `N_TRIPLE = 2,763,520 = C(256,3)`.
- `pair_bank` / `triple_bank` are **4-dimensional concept embeddings**, not
  full pair matrices. A router selects top-3 of 256 tokenizer feature slots,
  then gathers/hashes a pair and a triple.
- `MoE(256, dim, topk=2)` is a **256-expert** top-2 MoE. The "256 extension
  heads" are these 256 experts, not a separate 256-way output head.
- A 256-wide tokenizer bus pools bit/byte, subword, token, word and sentence
  states, adds meta/concept states, runs two micro-transformer blocks, and
  feeds back to meta.
- Meta has temporal slots `+0/+1/+2/+3`; the +0 state and future states are
  projected back into byte/word/sentence towers through per-layer gates.
- Feedback gates initialize near zero (`-8` logits), so loops start closed.
- There is **no 32³ paired point-cloud implementation** in the current engine.
  `32³ = 32,768 ≈ 32,640 + 128`; a 32³ lattice is a plausible geometric index
  for the pair bank with 128 padding/special points, but it is not in the code.

### Diamond's own loop ablation

The 10-minute WikiText ablation found full loops **worse** than no loops:

| Variant | eval nats | byte bits | sent top-5 |
|---|---:|---:|---:|
| no tokenizer/self loops | **6.5332** | **4.1862** | 56.35% |
| full self-referential loops | 6.5800 | 4.2430 | 56.05% |
| closed meta/concept loops | 6.6360 | 4.3237 | 56.05% |
| meta-cross output feedback | 6.8472 | 4.6450 | 56.35% |

So loops are not automatically useful. They must be gate-zero, ablated, and
measured against a no-loop control.

## What is applicable to the unified Terminal-RSI graph

Keep the v103 core exact at gate 0 and put the hierarchy in the 100M upgrade
branch:

1. **Boundary states**: causal pooling from byte hidden states into subword,
   word and sentence states using boundary heads/latent boundaries.
2. **Bottom-up sweep**: byte → subword → word → sentence.
3. **Top-down sweep**: sentence → word → subword → byte. Each edge is a
   separate zero-initialized gated residual, so it is a DAG per timestep and
   cannot leak future bytes.
4. **Lateral priors**: word→byte and subword→byte as independent gated priors
   into byte logits, not part of a cycle.
5. **Pair-concept memory**: use the two 32³ lattices as a 65,536-entry paired
   concept bank with a small concept dimension. The 256 bottom specialists
   write concepts into the bottom lattice; the 256 top specialists read the
   mirrored concepts from the top lattice. Full 256-expert dense MoEs at width
   704 are too large; use a low-rank, top-k pair router instead.
6. **Temporal meta slots**: +0/+1/+2/+3 for training heads; at generation only
   +0 conditions the next byte, so the output remains byte-driven.
7. **Generation**: sample/argmax a byte; update boundary states; use the
   top-down word/sentence priors to rescore the next byte distribution. Words,
   sentences and paragraphs are latent synthesized states; they are never
   sampled as tokens. This is the "byte-driven, layer-synthesized" contract.

All loop gates must start at zero and be evaluated against a no-loop ablation.
Diamond's result is a warning, not a template to copy blindly.

## 95%+ target

The current 4.46 BPB run has byte top-1 21.9% / top-5 53.1%. A 95% byte top-5
target requires a much stronger byte distribution than the current 10-minute
fresh model. The 2 BPB trained v103 parent is the only existing checkpoint in
that class; a from-scratch 100M model needs many more optimizer steps, not just
more loops. Hierarchical loops can improve calibration and context, but they
are not a substitute for the training budget.

## Parallel candidate replicas — implemented

`terminal_rsi/train_verbatim.py` now runs the neutral branch and each
candidate on cloned model/optimizer pairs:

- clone the current model and optimizer state;
- apply one candidate action per clone;
- train the same precomputed branch batches in separate CUDA streams;
- evaluate probe shards on all clones concurrently;
- restore the winning clone into the main model;
- update the critic/policy/replay from every branch outcome;
- reset the main sampler/counters to the pre-tick values.

A 45 s probe with **3 clones / 2 candidates** used 65.8 GiB allocated and
94.1 GiB reserved, with no OOM.  A 60 s probe with 2 clones / 1 candidate
moved the hierarchy gates and ran 16 RSI cycles (32 branch updates, 13
accepted) alongside 5 main updates.  The memory limit is dominated by the
reserved allocator cache, not the clones; `PYTORCH_CUDA_ALLOC_CONF=
expandable_segments:True` is set by the smoke runner.

