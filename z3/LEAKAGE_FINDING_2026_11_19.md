# Sidecar leakage finding (2026-11-19)

The "too good" numbers are not generalization. The concept sidecar is built
from the **entire context window**, and `_segment_pool` averages word and
sentence states over all tokens in the window. At position `t`, the word
and sentence summaries therefore contain **future tokens** from the same
word/sentence. `global_ctx = concept.mean(dim=1)` is also a bidirectional
summary.

This is not a subtle effect. It is a direct future-token channel.

## External FineWeb-Edu check

Setup:

- Source: `HuggingFaceFW/fineweb-edu` **10BT shard 013**, not used in the
  training build
- Split: **article-level** 20% test, 1,986 articles
- 2,000,000 tokens, whole articles, document-framed
- 13-gram overlap against a 2M-token train sample: **0.0**
- VRAM cap: 37 GB; peak 2.18 GB (coexists with the other training job)

| Mode | BPB | Top-1 | Top-5 |
|---|---:|---:|---:|
| Full sidecar (leaky) | **0.1076** | 93.23% | 96.54% |
| `compute_concept=False` (concept removed) | **6.8406** | 1.35% | 2.54% |

The full-sidecar number is ~0.51 bits/token, which is not a plausible
causal LM result. The concept-disabled number is worse than a uniform
model because removing the concept path puts the trunk out of
distribution; it is a break-glass bound, not a fair causal evaluation.

## Root cause

`concept_loop._segment_pool`:

- pools `word` over all tokens sharing a word id;
- pools `sent` over all tokens sharing a sentence id;
- the ids come from `SpanSidecar.from_token_ids` for the whole window.

Then `ConceptEncoder.forward` returns `concept.mean(dim=1)` as the global
context. Both paths see the future. Training and evaluation both used the
full window, so the model learned to exploit it.

Generation uses an incremental sidecar decoder, so it cannot see the
future. That is why generation is word salad even though the reported BPB
was low.

## Consequence

All metrics that used the full sidecar are invalid as causal LM numbers,
including:

- PTB 0.7850 (30-minute 84M)
- per-domain table 0.036–0.400
- earlier 20-minute/40-minute/55-minute BPBs

They measure a non-causal model with a future-token side channel.

## Fix

1. Replace `_segment_pool` with **prefix-only pooling**: at position `t`,
   average only `j <= t` in the same word/sentence.
2. Make `global_ctx` causal (e.g. last prefix state or causal running
   mean).
3. Retrain from scratch with the causal sidecar.
4. Evaluate with the same causal sidecar and the external article-level
   FineWeb test; report PTB and per-domain numbers again.

A causal-pooling evaluation of the current weights is the next diagnostic.
It will overstate the loss because the weights were trained with future
pooling, but it will show how much of the score was the leak.

## Status

The external FineWeb test is built and the leak-free/leaky bounds are
measured. The causal-pooling evaluation and retrain need a fresh Molab
token; the current `sb-64fe…` session now returns 401.

The 84M 30-minute checkpoint and the per-domain table are still on
Kaggle (`shem42/z4-broad30m-84m`), but those metrics are leaky and should
not be used for universality claims.
