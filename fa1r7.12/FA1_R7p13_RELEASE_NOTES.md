# FA1 R7.13 — Hierarchical context router (experimental)

**Codename:** FA1. PAL R7.12 remains the accepted full model, at joint step 14,416.
**Accepted parent SHA-256:** `54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d`.
**All five preexisting optimizer histories are preserved.** The directed reader remains an experimental R7.11 overlay, while the R7.12 LM gate is zero-for-promotion and not integrated into the accepted model. The FA1 R7.13 hierarchy is an additional small, reproducible, auxiliary router overlay.

## Architecture

One trained rank-64 directed writer; an 8×768 historical segmentation (6,144 old tokens), query-dependent top-2 routing to two transient matrices, and a weighted probability readout. The inherited rank-64 recurrent matrix is unchanged. Routing uses only the historical key/query, not the answer candidate or current/future target. The router was trained for 192 steps on lexical historical segment labels; chosen on historical development routing CE. The rank-64 writer and original language model were frozen.

## 512-query lexical historical benchmark (32-way hard negatives)

| Metric | Development | Historical test |
|---|---:|---:|
| Fixed last-768 directed-memory CE ↓ | 3.8381 | 4.0098 |
| FA1 routed top-2 correct memory CE ↓ | 3.2642 | 3.4273 |
| FA1 correct-memory top-1 ↑ | 21.09% | 18.95% |
| FA1 wrong-memory CE ↓ | 3.6084 | 3.7743 |
| FA1 reversed-order CE ↓ | 4.5903 | 4.4475 |
| Route top-2 source hit ↑ | 100% | 100% |

**Interpretation:** the query key is an exact unique token in older history, so source discovery is easy and the 100% hit rate is not a semantic or multihop claim. The new reader nonetheless improves learned key→value readout over the same frozen last-768 reader. Historical test repeatedly consulted; no independent book-level heldout guarantee.

## Adversarial routing stress

- Synthetic conflicting earlier key→different-value pair (115 events): top-1 selects latest true source only **4.35%**, despite top-2 finding that source. This exposes an essential revision/recency limitation.
- Query replaced by a nearest embedding neighbor absent from history (128 events): top-2 source hit **62.5%**; this is *not* synonym paraphrasing.
- These diagnostic limitations are recorded, not tuned away using the historical test.

## Historical LM bridge (512 positions)

| Condition | Dev NLL ↓ | Test NLL ↓ |
|---|---:|---:|
| Parent LM (no FA1 head) | 5.17199 | 5.63917 |
| FA1 hierarchical, correct history | 5.17245 | 5.63387 |
| FA1 hierarchical, unrelated history | 5.16456 | 5.59616 |

Correct history loses to unrelated history on **both** splits. Thus: **NO LANGUAGE-MODEL PROMOTION, NO FLUENCY OR FRONTIER SCALING CLAIM.** The accepted checkpoint and original five optimizers remain unchanged.

## GitHub mirror

Text sources, reports and old release provenance are merged at:
https://github.com/shemshallah/nanlm_experimental/tree/master/fa1r7.12

The GitHub connector does not offer upload of large binary release attachments, so this full ZIP (including checkpoint and training corpus) is distributed in ChatGPT Library, not as a complete GitHub-hosted model.

## Next falsifiable experiments

1. Context-conditioned revision routing with hard negatives sharing keys and contradicting values, not just unique lexical keys.
2. Real paraphrase/semantic distractors from document-disjoint NaNROUTER books; preserve sealed test and hash exact books.
3. Migrate from 6,144 to 8k/32k/128k via sparse addressable chunk summaries, not a single expanded rank-64 average.
4. End-to-end joint optimization requires correct-memory > unrelated on development and improved LM NLL, along with stable regression guards.