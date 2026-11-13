# Frontier fix plan: tokenizer, context, and MTP

This is a design/research plan, not a training claim. All Kaggle runs are
stopped. The next shard build was cancelled so we do not keep generating data
for an architecture that is not ready.

## What we measured

FineWeb-Edu 6h run (`v3_fineweb_6h`):

```text
params            112,899,328
steps             50,000
tokens/bytes      204,800,000 bytes
wall              18,788 s (5.22 h)
train CE          0.9538 nats/byte = 1.376 bits/byte
FineWeb valid     1.4148 bits/byte, top-1 70.38%, top-5 90.93%
WikiText-103 test 1.6969 bits/byte, top-1 68.55%, top-5 87.28%
```

MTP-only repair, 500 steps, trunk frozen:

```text
CE before         0.98063
CE after          0.98063
MTP before        1.54843
MTP after         1.54856
```

Conclusion: the frozen trunk does not expose t+2 information in a
linearly usable way. Training only the existing MTP head does not fix MTP.
The head is not the bottleneck by itself; the MTP branch/objective is.

## Why 273 vocabulary is not enough

Current V3 is byte-level plus EOS 256. That is good for robustness and exact
bits/byte accounting, but it is not frontier-grade tokenization:

- 1 byte per token makes sequences ~4x longer than a 4-byte-per-token BPE.
- 1024 byte context is only ~200–300 English words.
- The embedding/output matrix is tiny, so most capacity is in the trunk.
- Byte-level MTP at t+2 predicts only two characters; there is little useful
  future information and the MTP objective is poorly conditioned.
- The router/lm head cannot exploit subword regularities cheaply.

### Tokenizer v4 proposal

1. Train a **64k–128k BPE** on a deduplicated mix:
   - FineWeb-Edu
   - Gutenberg/book prose
   - code and math
   - multilingual sample
   - Wikipedia/document boundaries
2. Use byte fallback for every UTF-8 byte; no unknown token except explicit
   specials.
3. Special tokens:
   - `<pad>`, `<bos>`, `<eos>`, `<doc>`
   - `<now>` (the now-slot)
   - `<future_1>`, `<future_2>`, … if a slot token is used
   - chat/tool tokens only if the training objective needs them
4. Split digits into individual digits (or fixed digit groups). This is a
   quality win for arithmetic and code.
5. Normalize whitespace/Unicode deterministically; preserve exact bytes for
   round-trip.
6. Measure **bytes/token** and **bits/byte** (BPB), not just bits/token.
   Target: ~3.8–4.5 bytes/token on English web text.

Vocab sizing tradeoff for this 112.9M model:

- 64k vocab, d_model 256, tied embeddings: ~16.4M embedding params.
- 128k vocab: ~32.8M; this is a large fraction of a 113M model.
- If 128k is required, use a **factorized/low-rank embedding** or increase
  d_model/scale. Do not bolt 128k onto the current model and call it frontier.

The current 273-vocab checkpoint cannot be reused with a new tokenizer. A new
model/embedding must be trained, or the old byte embedding can initialize a
new subword embedding by composing byte vectors.

## Context: “frontier-grade” means more than setting ctx=8192

To reach 32k–128k context we need:

- **GQA or MLA** with a real KV-cache budget; current GQA is parameter-only
  because the training path expands K/V.
- **FlashAttention / fused local attention**. The current local window builds a
  dense T×T mask; at 8k that is expensive, at 32k impossible.
- **Sliding-window local + global layers** (already present) with a properly
  sized window (1k–4k), not 512.
- **RoPE base/YaRN/NTK** already present; add long-context data and evaluate
  RULER/LongPPL, not only next-byte CE.
- **Activation checkpointing and sequence parallelism** for long sequences.
- A100/H100-class hardware for 32k+; a 16GB T4 cannot train a 113M dense model
  at 32k with useful batch size.

## MTP: ranked hypotheses and fixes

### H1. Frozen features lack t+2 information (most likely)

Evidence: MTP-only 500 steps left held-out MTP at 1.5486 while CE stayed
0.9806. The head can fit training batches but not generalize.

Fix: add a small **MTP adapter branch** and freeze the trunk:

```text
h = final_norm(trunk(x))
a = h + W_up(silu(W_down(rmsnorm(h))))      # low-rank, e.g. 256 -> 64 -> 256
logits_t+2 = mtp_head(a)
```

Train only the adapter, final norm (optional), and MTP head for 500–2,000
steps. Main logits still use the frozen trunk, so CE must stay ≈0.98. If MTP
drops while CE stays fixed, the features are usable and the adapter is the fix.

### H2. Random/separate MTP head is a bad prior

Current MTP head is a fresh `Linear(d_model, vocab)`. The main head is tied to
the token embedding. A fresh 256×273 matrix has a poor initialization.

Fixes, cheapest first:

- initialize MTP head from the tied token embedding / main head;
- tie MTP head weights to the token embedding;
- use `logits_mtp = main_logits + low_rank_correction(h)` so MTP starts at the
  main model’s distribution;
- train the correction with a small LR.

### H3. Byte-level t+2 is the wrong horizon/objective

At byte level, t+2 is two characters. That is often function-word/local
continuation, and the information is not a useful “future plan.”

Test a horizon sweep:

- byte: t+2, t+4, t+8;
- after tokenizer v4: subword t+2/t+4/t+8;
- word/sentence-boundary prediction;
- latent MTP: predict a future hidden state plus a token CE, not only hard
  token CE.

Use a shared adapter and per-horizon heads; report each horizon separately.

### H4. The “now slot” / head 255

Important: in `head_geometry.py`, `VOID_HEAD = 255` and `TRAINABLE_HEADS = 255`
means heads 0–254 are trainable triads and head 255 is the **void/global**
head. V3LM does not currently use that geometry at all. So “head 255” is not
an output-vocabulary head and cannot directly be a token predictor.

If we want a now-slot, implement it explicitly:

- add a learned `now_slot` vector (or a `<now>` token embedding);
- condition the MTP adapter/head on `[h_t ; now_slot]`;
- use a **copy gate**: `p = gate * one_hot(x_t) + (1-gate) * mtp_head(h)`,
  where gate is produced by the now slot. This gives the model a local-copy
  prior and a null/no-op path for unpredictable positions;
- or reserve a slot-255 **output class** in a small auxiliary head that
  predicts “same/now/future” rather than the full vocabulary.

Do not repurpose a byte value 255 as `<now>`: 255 is a valid byte and the
current tokenizer already uses 0–255. Add a new special ID above the byte
range.

### H5. MTP loss/schedule interferes with CE

Current MTP weight is fixed at 0.3. Try:

- ramp MTP weight from 0.02 to 0.15 over 1k steps;
- separate MTP LR (lower than trunk);
- detach the trunk for the first adapter experiment;
- mask MTP to high-information positions (document/sentence boundaries,
  not every byte);
- normalize by valid MTP positions correctly and verify the shift with a
  synthetic sequence where t+2 is known.

### H6. Trunk must adapt, but gently

If H1/H2 fail, unfreeze only:

- final RMSNorm;
- last transformer block;
- a LoRA adapter on the last 1–2 blocks.

Gate: CE must stay within 0.01 nats of 0.9806; if CE drifts more, reject.

## Experiment sequence (small runs only)

All runs: 500 steps first, new FineWeb-Edu shard, fixed eval, report CE/MTP
separately. No 6h runs until a 500-step gate passes.

| ID | Change | Pass condition |
|---|---|---|
| T0 | tokenizer v4 64k, no model run yet | ≥3.8 bytes/token, 100% byte fallback |
| M1 | MTP adapter + tied/init head, trunk frozen | MTP −0.05 vs 1.548; CE ±0.01 |
| M2 | M1 + final norm unfrozen | same |
| M3 | M1 + last-block LoRA | same |
| M4 | now-slot/copy gate | MTP −0.05; no CE drift |
| M5 | horizon sweep 2/4/8 (byte then BPE) | best horizon by held-out MTP |
| C1 | 8k context with GQA/MLA + FlashAttention | finite, memory fits A100 |
| C2 | 32k context, activation checkpointing | finite, long-context eval |

## Hardware reality

- T4 16GB: good for 1k–2k context, 112M, 2-GPU throughput; not frontier
  context.
- A100 40/80GB: needed for 8k–32k training with a reasonable batch.
- Frontier tokenizer/context/model work needs a larger data and compute budget
  than the current Kaggle quota.

## Immediate decisions

1. Stop all runs. **Done.**
2. Do not train another 6h cycle on the 273-byte tokenizer.
3. Build tokenizer v4 first; tokenizer quality is a hard gate.
4. Fix MTP with the M1 adapter test before any next full cycle.
5. Build a new FineWeb-Edu shard only after tokenizer/MTP decisions; the
   previous “next shard” build was cancelled.
