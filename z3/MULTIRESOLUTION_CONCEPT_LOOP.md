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

The active concept path now wires the pieces into that loop.  This document
keeps the remaining gaps explicit instead of treating the diagram as a claim.

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
  - per-resume LR/WSD controls
  - optional multi-corpus domain mixture with per-corpus task ids
- `z3/tune_word_vocab_bins.py`:
  - streaming word-vocabulary retune from aligned token bins.

## Disconnects

### D1. The tokenizer hierarchy is in the training stream (active path)

The v4c corpus is still a flat uint16 subword stream, but `SpanSidecar`
now reconstructs token -> byte spans, word spans, sentence spans, math spans,
and composition events for every training/evaluation window.  `--sidecar` is
on by default for the concept path.

The remaining limitation is CPU overhead for random windows, not a missing
signal.  Repeated eval/overfit windows use an LRU; a persistent random-window
cache is still pending.

**Severity: resolved for signal wiring; performance remains a gap.**

### D2. The 32,640 pair objects are connected to the concept loop

The active concept path uses exactly 256 vertices, seeds them from
deterministic byte-bit hypervectors, and addresses `C(256,2)=32,640` pairs and
`C(256,3)=2,763,520` triples.  Word/sentence/math concepts are pooled into the
query; the object memory is still sparse top-k rather than a dense object
vocabulary.

**Severity: resolved for addressing and seeding; hard object-token
generation remains partial.**

### D3. Concept memory runs inside every block (active path)

The active concept loop runs per block, feeds attention and every MoE router,
and carries concept/object state across depth.  The legacy post-hoc
`comb_memory` remains available as a separate optional path for old
checkpoints, but it is not the multi-resolution route.

**Severity: resolved for the active concept path.**

### D4. The router context is attention-only (addressed in the active path)

`route_ctx` is still updated from the attention summary and the MoE route
summary, but the active path now conditions concept pooling, object retrieval,
and the pointer query on that route state, then feeds the resulting multi-label
word/sentence/math/composition concept back into every router.  This closes the
loop in both directions.

**Severity: resolved in the active path; explicit boundary probes remain a
validation gap.**

### D5. Alignment is reconstructed from token IDs (active path)

`SpanSidecar.from_token_ids` rebuilds byte spans from the byte-complete
subword table and aligns word/sentence/math spans to the existing token
window.  It does not require rebuilding the v4c corpus.

The remaining limitation is that chunk boundaries can split a UTF-8 code
point or word during the full-corpus word-vocabulary tuner; the training
sidecar itself uses exact token-byte spans.

**Severity: resolved for training; tuner boundary handling is approximate.**

### D6. MTP is multi-resolution (active path)

MTP has HD `<now>` and permutation future rails, and its branch context now
includes the route plus final concept/object state.  Shared heads additionally
supervise future composition events and future object pointers.

**Severity: resolved for structure conditioning; future structure quality is
not yet benchmarked.**

### D7. Incremental generation exists, with a tokenizer limitation

`V3LM.generate` now:

1. emits a subword,
2. feeds generated IDs through the incremental composition decoder,
3. rebuilds the aligned sidecar and concept/object state,
4. feeds predicted composition events back into the concept state,
5. reports model/tokenizer/object/composition readouts.

The remaining limitation is that `LayeredTokenizer.decode_token_ids` decodes
a completed sequence rather than exposing a fully reversible streaming
word/sentence state machine.  The event-level incremental decoder is the
current bridge.

**Severity: active path implemented; tokenizer streaming is partial.**

### D8. Composition layer (structural layer implemented; grammar rules pending)

The active path now has `CompositionVocabulary`, `CompositionTokenizer`, and a
multi-label composition head aligned to token/byte spans.  It can emit and
consume boundaries such as `<word_end>`, `<sentence_end>`, `<math_end>`, and
`<compose>`.

What remains is a learned grammar/rewrite layer that maps a composition such as
`("x", "+", "3") -> "x + 3"` or abstracts a phrase into a reusable structural
token.  The structural vocabulary exists; the grammar rules do not.

**Severity: high for mathematical translation.**

### D9. Object IDs are not part of the model vocabulary

The 32,640 pair IDs and 2.76M triple IDs are still not text-vocabulary tokens.
The pointer head predicts the two vertices of a pair, generation reports the
resulting ID, and the hard pointer at token `t-1` now conditions object
retrieval at token `t` through a straight-through stream.  The remaining gap is
an explicit sampled object-ID vocabulary entry, not a recurrent pointer input.

**Severity: partial; hard recurrent pointer exists, object vocabulary does
not.**

### D10. Full-corpus layer tuning is scripted but not applied

The layered tokenizer was tuned on a 5MB FineWeb sample, while v4c training
uses the 300MB FineWeb shard.  `z3/tune_word_vocab_bins.py` streams the v4c
bins and can write a full-corpus word artifact, but it has not been run here.

**Severity: medium; tooling exists, data run pending.**

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
- `ConceptEncoder` hierarchical subword → word → sentence pooling
- shared 256-vertex / `C(256,2)` / `C(256,3)` object memory
- deterministic 256-vertex byte-concept seed
- per-layer concept → object → route context → attention → MoE feedback
- bidirectional route → concept/object/pointer conditioning
- causal hard pointer stream with straight-through gradients
- causal object-state recurrence along the token sequence
- gated recurrent concept carry across all blocks
- persistent task/domain embedding through concept state and every router
- concept state fed into MTP branch context
- MTP future-composition BCE through the shared structural head
- MTP future-object-pointer CE through the shared vertex codebook
- future composition/pointer losses reported by the evaluator
- bits/byte telemetry from the tokenizer byte table
- multi-label composition event head and BCE supervision
- per-event composition threshold calibration for generation
- stateful composition grammar filtering during generation
- multi-label composition events consumed by the concept encoder
- sidecar-aware trainer and evaluator
- tokenizer-side incremental composition decoder
- `CompositionVocabulary` structural layer
- `CompositionTokenizer` encode/decode layer with token/byte alignment
- exact pair/triple object IDs exposed during generation
- learned object-pointer head over the 256 vertex pair space
- pointer distribution fed back into the object stream
- pointer pairs constrained to distinct vertices
- generation streams the reconstructed sidecar back into the concept loop
- generation feeds predicted composition events into the next concept state
- adaptive top-k object candidates for 2048 context memory
- autoregressive `generate()` with model/tokenizer composition and object readouts
- task/domain id conditioning during generation
- LRU reuse of aligned sidecars for repeated eval/overfit windows
- streaming `tune_word_vocab_bins.py` for full-corpus word-vocabulary
  retuning directly from v4c token bins

Validated in 313M concept runs:

- 512 context: composition loss 0.384, no non-finite steps
- 1024 context: composition loss 0.291, no non-finite steps
- 2048 context with top-4 objects: composition loss 0.199 → 0.126,
  no non-finite steps, ≈1.86 BPB after 2,314 steps
- MTP branch beats the fallback on both correct-t1 and wrong-t1 subsets
  (7.38 vs 9.98 correct, 7.26 vs 8.55 wrong);
  the remaining correct/wrong loss difference is t+2 selection hardness,
  not a branch failure

Still pending, and should not be pretended otherwise:

- mapping the 256 deterministic seeds to actual byte values through the
  tokenizer's byte embedding (the current seed is a deterministic HD codebook)
- running the new full-corpus word-vocabulary tuner on the 300MB v4c shard
  (the script exists; no corpus is present on this phone)
- long-run validation beyond 10-minute sessions
- feeding generated object IDs back as an explicit vocabulary token stream
  (the soft pointer now conditions the object stream, but the pair/triple ID
  is not yet a sampled vocabulary token)
- a persistent precomputed sidecar cache for the random training stream
  (the current cache only helps repeated eval/overfit windows)

## Frontier disconnects to avoid

- Do not compute word/sentence features only at evaluation.
- Do not apply object memory after the block stack.
- Do not let object IDs remain a hidden internal space if generation needs
  them.
- Do not compare 64k-subword top-5 with 256-byte top-5 as if they were the
  same task.
- Do not add more experts without a concept/tokenizer routing signal.
