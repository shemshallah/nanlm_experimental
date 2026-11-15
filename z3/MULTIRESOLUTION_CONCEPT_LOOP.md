# Multi-resolution concept loop

This is the architecture we actually want:

```text
bytes -> subwords -> words -> sentences -> concepts
   ^        |           |          |           |
   |        v           v          v           v
   +---- aligned spans / concept abstraction ---+
                              |
                    C(256,2) + C(256,3) objects
                              |
                 top-k object routing per layer
                              |
             MoE router + attention + generation
                              |
                    composition tokenizer
                              |
                     next token / span
```

The current model has the pieces, but they are not wired into that loop.
This document separates what exists from what is disconnected.

## What exists today

- `z3/layered_tokenizer.py`:
  - bit and byte levels
  - 64k v4 BPE subword level
  - token/special level
  - word spans with a 16k vocabulary
  - sentence spans (512-char cap)
  - math spans
- `z3/tokenizers/v4_64k/tokenizer_complete.json`: byte-complete 64,060 vocab.
- `z3/combinatorial_objects.py`:
  - exact C(V,2) and C(V,3) IDs
  - growable vertex codebook
  - pair/triple hypervector binding
  - sparse `CombinatorialMemory` with top vertices, top pairs, top triples
- `z3/moe_vectorized.py`:
  - hierarchical branch -> expert routing
  - context-conditioned branch/expert logits
  - HD expert/branch codebook
  - associative-memory experts
- `z3/v3_model.py`:
  - attention -> routing -> attention feedback through all layers
  - MTP branch with HD `<now>`/permutation time rails
- `z3/train_v3.py`:
  - v4c bin training and Wiki eval
  - per-resume LR/WSD controls.

## Disconnects

### D1. The tokenizer hierarchy is not in the training stream

The v4c corpus is a flat uint16 subword stream. It has no sidecar for:

- token -> byte spans
- subword -> word spans
- word -> sentence spans
- math spans
- composition boundaries

So the model cannot use the word/sentence/math layers during training.
The layered tokenizer is currently an encoder/evaluation artifact, not a
training input.

**Severity: critical.** This is the main missing wiring.

### D2. The 32,640 pair objects are disconnected from the tokenizer

`C(256,2) = 32,640`, but the model default uses `comb_vertices=64`
(2,016 pairs, 41,664 triples) and only top-8 pairs per token.

More importantly, the 256 vertices are **learned latent vertices**, not
byte/subword/word/sentence concepts. The pair/triple memory is not driven by
the tokenizer hierarchy.

**Severity: critical.** We need 256 concept vertices seeded from the
tokenizer levels, with exact C(256,2) addressing.

### D3. Concept memory is applied once, after all blocks

`V3LM.forward` runs all transformer blocks, then adds
`self.comb_memory(x)` once before the output norm.

The MoE routers never see:

- the object-memory summary
- word/sentence state
- math/composition state

So there is no multi-layer object feedback. The object memory is a
post-hoc residual, not part of the routing loop.

**Severity: high.**

### D4. The router context is attention-only

`route_ctx` is updated from `attn_out.mean(1)` and the MoE route summary.
It has no explicit:

- word boundary
- sentence boundary
- math span
- composition-token state

The router can infer some of this from the hidden state, but it is not
guaranteed, probeable, or aligned.

**Severity: high.**

### D5. The word/sentence layers are not aligned to v4c token IDs

The v4c builder discards document/span alignment. There is no
token-indexed word/sentence mask, so adding them later requires rebuilding
the corpus or re-tokenizing raw documents.

**Severity: high.** This is why the sidecar must be built now, before more
training.

### D6. MTP uses time permutations but no composition state

MTP has HD `<now>` and permutation future rails, but its branch does not see:

- the current word/sentence span
- the composition state
- pair/triple object state

It can predict future subwords, but not future word/sentence/composition
structure.

**Severity: medium-high.**

### D7. There is no incremental/compositional generation path

Training/evaluation is causal-LM scoring. There is no generation loop that:

1. emits a subword,
2. updates byte/word/sentence state,
3. runs concept abstraction,
4. updates the object router,
5. emits the next subword or composition token.

The tokenizer is not reversible **incrementally** at word/sentence level.
`LayeredTokenizer.decode_token_ids` decodes a completed sequence, not a
streaming sequence with composition.

**Severity: critical for generation.**

### D8. No composition/grammar tokenizer layer

We have byte/subword/token/word/sentence/math, but no layer that maps a
composition such as:

```text
("x", "+", "3") -> "x + 3"
("solve", eq) -> equation
(word, word) -> phrase
```

This layer needs its own vocabulary and generation rules. It does not exist.

**Severity: high for mathematical translation.**

### D9. Object IDs are not part of the model vocabulary

The 32,640 pair IDs and 2.76M triple IDs are an internal memory space, not
generation tokens. The decoder cannot emit or reference an object ID, and
there is no copy/pointer path from object state back into token generation.

**Severity: high.**

### D10. v4c corpus and layer tuning are on different data scopes

The layered tokenizer was tuned on a 5MB FineWeb sample, while v4c training
uses the 300MB FineWeb byte shard. Word/sentence vocab statistics are not
recomputed on the full training shard.

**Severity: medium.**

## Target architecture

### A. Aligned sidecar schema

For every v4c token bin, store parallel token-indexed arrays:

```text
token_ids        uint16 [N]
byte_start       uint32 [N]
byte_len         uint16 [N]
word_id          uint16 [N]   # 0 = no word boundary; 1..K word IDs
sentence_id      uint16 [N]   # span ID or packed start/length
math_kind        uint8  [N]   # 0 prose, 1 number, 2 operator, ...
composition_id   uint32 [N]   # optional learned composition token
```

Because v4c is built from byte bins, the builder can retain exact UTF-8
offsets instead of re-tokenizing and guessing.

### B. 256-vertex concept seed

Use exactly 256 vertices:

```text
vertex[0..255] = byte-symbol concept seeds
```

Initialize each from the byte embedding/hypervector, then let them train.
All C(256,2)=32,640 pair objects are addressable. C(256,3) is sparse.

For word/sentence concepts:

```text
word_concept   = bundle(vertex[byte] for bytes in word span)
sentence_concept = bundle(word concepts)
```

The pair/triple memory is then a compositional store over the same 256
vertices, not a separate learned latent universe.

### C. Per-layer concept feedback

At each MoE block `l`:

```text
concept_ctx = ConceptEncoder(byte, subword, word, sentence, math, composition)
object_ctx  = ObjectMemory(concept_ctx)       # top pairs/triples
router_ctx  = RouteNorm(route_ctx + W_c concept_ctx + W_o object_ctx)
moe_out     = MoE(hidden, context=router_ctx)
route_ctx   = RouteNorm(route_ctx + W_a attn_summary + W_r route_summary)
```

Object memory must run **inside the layer loop**, not once after all blocks.
The router sees object/concept state at every layer.

### D. Two-way multi-resolution loop

- Low layers route on subword/byte objects.
- Middle layers route on word/pair objects.
- High layers route on sentence/triple objects.
- High-level concept state feeds back down through `route_ctx` and
  `route_to_attention`, so attention at every layer is conditioned on the
  current word/sentence/composition state.

This is the multi-layer feedback loop.

### E. Composition tokenizer and generation

Add a composition layer with a small structural vocabulary:

```text
<word_start> <word_end> <sentence_end>
<math> </math> <eq> </eq>
<compose> <decompose> <copy>
```

Generation loop:

```text
sample subword
 -> update byte/subword state
 -> update word/sentence spans
 -> concept abstraction
 -> object memory top pairs/triples
 -> next router context
 -> sample next subword or composition token
```

For math translation, add constrained decoding/verifier hooks:
`<eq>` expression, SymPy/LaTeX validity, number/operator copy.

### F. Losses

- main CE
- MTP CE
- word-boundary/sentence-boundary auxiliary CE
- composition-token CE
- object-routing load balance and diversity
- copy/constraint loss for math/translation

## Build order

1. **Sidecar builder** for v4c: token↔byte↔word↔sentence↔math alignment.
2. **256 vertex seed + C(256,2) object codebook**; verify exact IDs and
   all-pair recall.
3. **Concept encoder** with span pooling and role/position binding.
4. **Per-layer object memory** conditioned on concept state; feed router and
   attention.
5. **Composition vocabulary/head** and incremental tokenizer state.
6. **Generation loop** with constrained math eval.
7. **A/B gates**: CE, word/sentence probes, object routing, Wiki BPB,
   MTP per-byte, composition exact-match.

## Implementation status

Implemented in this integration:

- `SpanSidecar` token↔byte↔word↔sentence↔math alignment
- `SpanSidecar.from_token_ids` bridge for v4c bin batches
- multi-row sidecar collation
- `ConceptEncoder` word/sentence pooling and concept vectors
- shared 256-vertex / `C(256,2)` / `C(256,3)` object memory
- deterministic 256-vertex byte-concept seed
- per-layer concept → object → route context → attention → MoE feedback
- concept state fed into MTP branch context
- composition event head and auxiliary CE
- sidecar-aware trainer and evaluator
- tokenizer-side incremental composition decoder
- `CompositionVocabulary` structural layer
- exact pair-object IDs exposed during generation
- autoregressive `generate()` with model/tokenizer composition and object readouts

Validated in 313M concept runs:

- 512 context: composition loss 0.384, no non-finite steps
- 1024 context: composition loss 0.291, no non-finite steps
- 2048 context with top-4 objects: composition loss 0.199 → 0.164,
  no non-finite steps, ≈1.94 BPB after 925 steps
- MTP branch beats the fallback on both correct-t1 and wrong-t1 subsets;
  the remaining correct/wrong loss difference is t+2 selection hardness,
  not a branch failure

Still pending, and should not be pretended otherwise:

- mapping the 256 deterministic seeds to actual byte values through the
  tokenizer's byte embedding (the current seed is a deterministic HD codebook)
- a trained full-corpus word/sentence vocabulary on the 300MB v4c shard
- long-run validation beyond 10-minute sessions
- feeding generated object IDs back as an explicit token/pointer stream
  (they are currently readouts, not vocabulary tokens)

## Frontier disconnects to avoid

- Do not compute word/sentence features only at evaluation.
- Do not apply object memory after the block stack.
- Do not let object IDs remain a hidden internal space if generation needs
  them.
- Do not compare 64k-subword top-5 with 256-byte top-5 as if they were the
  same task.
- Do not add more experts without a concept/tokenizer routing signal.
