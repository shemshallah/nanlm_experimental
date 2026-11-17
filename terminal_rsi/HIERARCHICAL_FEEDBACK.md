# Hierarchical feedback audit: Terminal-RSI vs Diamond

## Current Terminal-RSI loops (what actually exists)

Today there is **no word→byte or subword→word generation loop**. The existing
loops are:

1. **Byte→byte recursive passes** in `TerminalRSI.forward`: two passes over the
   12-layer upgrade backbone, with previous-byte surprise and a learned critic
   feeding the next pass.
2. **v103 online feedback** in `V103ParityGPUFullSystem`: online bit counts,
   slot state, feedback state, verified byte/terminal outcomes.
3. **v4 controller loop**: branch proposal → train one step → probe shards →
   accept/reject → critic/world-model/policy update. This is a *training-policy*
   loop, not a language-level hierarchy.
4. **Hierarchical metrics**: byte/subword/token/word/sentence evaluation only.
   These are not wired back into generation.

So the requested sequence `word→byte→subword→word→subword→byte` is **not**
implemented. It is also a directed cycle; it must be built as a causal DAG per
timestep, not one circular chain.

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
5. **Pair-concept memory**: use a 32,640-entry pair bank with a small concept
   dimension (as Diamond does) as a sparse lookup over hierarchical feature
   slots. Full 256-expert dense MoEs at width 704 are too large; use low-rank
   expert factors or a small top-k pair router.
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

## Parallel candidate replicas

The current v4 candidate loop is sequential. The card at the 2048/microbatch
1024 profile uses ~63 GiB allocated / 66 GiB reserved, leaving room for branch
replicas because main training and RSI evaluation do not overlap.

Proposed implementation:

- At each RSI tick, snapshot the base model+optimizer.
- Create K branch replicas (model + optimizer) and load the base state.
- Apply one candidate action per replica.
- Train `horizon` branch steps on the same precomputed branch batches, each
  replica on its own CUDA stream.
- Evaluate each replica's probe shards in parallel.
- Select the best branch; copy that replica's model/optimizer state back into
  the main model; update critic/policy/replay with all branch outcomes.
- Keep the main sampler/counters reset to the pre-tick values.

This is not implemented yet. It requires a clone-local trainer because the
current `train_updates` closure is bound to the main model/optimizer. It is the
next code change for both throughput and RSI quality.
