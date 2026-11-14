# FA1 R7.35 — causal historical values and isolated writer validation

**Status: unpromoted research continuation.** FA1 remains one inherited model,
one unified optimizer, on `master`. The accepted step-14,416 R7.13 ancestor
and FA2 are unchanged. R7.35 uses a new *parameter group*, not a replacement
network or new training lineage.

## Diagnosed disconnections

1. R7.29/30 retrieved distant evidence into a context **query**, but the
   factorized top-8 triangle **values** came only from learned vertex/address
   embeddings. Correct older-book content was not explicitly stored in those
   sparse values.
2. Sending written values into the ordinary triangle-to-pair path would bypass
   the apex. R7.35 keeps the old direct path intact and adds only a separate
   sparse value read into the apex, then uses the existing apex -> pair ->
   point descent and word/conditional-byte decoders.
3. Zero-initialized new writer/apex gates provide exact checkpoint migration,
   but leaving them near zero throughout training deprives their internal
   attention and value weights of useful gradients. A recorded candidate-only
   warm start opens newly attached gates **after** proving baseline identity.
   Resuming a trained R7.35 checkpoint preserves its learned gates by default.
4. The prior matched-history intervention changed queries and values together.
   R7.35 can now replace only four far modal **values** from another book while
   keeping the query, inherited hidden state, geometric addresses, recent and
   far features, all other modal inputs and the gold target exactly fixed.
   A separate value-only word and byte contrast/ablation is scored on DEV.

Each query attends to the causal far-memory modal IDs with explicit eligibility
masks; all-masked rows write exactly zero. The full 2,763,520 face space is
never materialized. Only the top eight selected triangle addresses receive
the history writer's values. No future target label participates in selecting
or writing memory.

## Reproducible CPU experiment

Parent is `FA1_R733_V2_GEOLR96_UNPROMOTED.pt`, SHA-256
`6d50629767c75bda1e4c61aabdb3fd46bfe56aacc4288336eee992e5dd040924`,
with 399 original model entries, 17 optimizer groups, and its trained moments.
R7.35 appends apex feedback (group 18) and the history writer (group 19), with
strict load/save continuity. No inherited optimizer moment is reset.

Source is the SHA-documented NaNROUTER prose-books release subset (12 TRAIN,
12 DEV, 14 untouched TEST books); `prepare_document_corpus.py --presplit`
preserves its original source assignment. Because far routing requires a
32,768-word preceding history, just **four** DEV books yield eligible long
positions (96 selected DEV targets). The other eight are not implicitly
promoted to long-history evidence. This is a debugging benchmark, not the
required 12+ age-eligible independent-book promotion study.

Three candidate-only stages were executed, retaining the selected development
state and its optimizer history at each transition:

| Stage | CPU updates executed | DEV targets | Selected update | Word NLL before -> after | Conditional spelling BPB before -> after |
|---|---:|---:|---:|---|---|
| Zero-gate writer pilot | 24 | 72 | 16 | 4.604120 -> 4.600262 | 5.197818 -> 5.000693 |
| Explicit gate warm start | 72 | 96 | 1 | 4.863899 -> 4.862853 | 5.028177 -> 5.001301 |
| Value-only donor contrast | 48 | 96 | 1 | 4.862853 -> 4.862622 | 5.001301 -> 4.998825 |

The first stage uses fewer target positions and **must not** be directly
compared with later NLLs. Stage-two and stage-three use the same deterministic
96 DEV positions. Across stages, 144 exploratory updates were *executed*,
but only 18 selected candidate updates (16+1+1) remain after development
selection. No claim is made that all 144 updates were retained.

### Final conditional-byte and word diagnostics (96 DEV positions)

| Metric | Parent for final stage | Selected R7.35 candidate |
|---|---:|---:|
| Full mixture word NLL, nats/word ↓ | 4.862853 | 4.862622 |
| Conditional spelling bits/byte ↓ | 5.001301 | 4.998825 |
| Conditional spelling byte top-1 ↑ | 5.45% | 5.72% |
| Conditional spelling byte top-5 ↑ | 44.69% | 44.96% |
| Geometry word gain, nats/word ↑ | 0.001147 | 0.001169 |
| Writer-off word gain ↑ | 0.0000325 | 0.0000269 |
| Writer-off branch-word gain ↑ | 0.000278 | 0.000700 |
| Value-only wrong-book word advantage ↑ | 0.000100 | 0.000107 |
| Writer-off literal-byte gain, nats/byte ↑ | -0.0000358 | **-0.0000401** |
| Correct versus foreign older-history word advantage ↑ | -0.000687 | **-0.000565** |

The isolated writer word gain has a four-book bootstrap 95% CI of
`[-0.0000598,+0.0000856]`; it is **not established**. The byte writer gain has
a negative four-book bootstrap interval, `[-0.0000536,-0.0000260]`. The
positive value-only wrong-book advantage does not repair that byte regression
or the overall negative correct-history advantage.

The original 32K-word trunk's top-1/top-5 are 17.71%/44.79% on these targets;
the full near/far/Diamond *mixture* does not expose an all-word normalized
ranking, so full-mixture top-1/top-5 are **unavailable**, not inferred from
target-only NLL. First-byte top-1/top-5 are 6.25%/41.67%. Conditional spelling
BPB is not raw contiguous-text BPB. TEST books remain sealed.

## Decision and recovery

**No promotion.** The final selected, checkpoint-resumable candidate is
`FA1_R735_VALUE_ONLY_UNPROMOTED.pt`, SHA-256
`c4c6c2b196dd3d79d032d2e957c15669d8b8a804f6e6fed53515b28d2d81c6be`,
size 265,921,872 bytes; it retains all 19 groups and is marked `promoted=False`.
The source checkpoint was hashed again after each round and remained intact.

Next controlled study: choose at least 12 genuinely age-eligible new DEV books,
enlarge TRAIN book diversity, and separately target writer-only byte gradients
before opening the sealed TEST set. Correct-history and geometry ablations must
pass their book-level lower confidence bounds together; a spelling-only win
does not promote the model.

Run tests with:

```bash
PYTHONPATH=fa1r7.12/source python -m unittest discover \
  -s fa1r7.12/source -p 'test_fa1_r735_*.py' -q
```

