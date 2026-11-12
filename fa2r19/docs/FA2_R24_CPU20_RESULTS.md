# FA2 R24 — measured unified CPU20 continuation (nonpromoted)

**Date:** 2026-11-12. **Disposition:** UNPROMOTED experimental candidate, original R18 checkpoint protected.

## Execution provenance

A local CPU reproduction of the `master` FA2 R24 **single integrated PAL + R18/R21/R22/R23 + signed-mirror MoE architecture** was trained for 20 actual AdamW optimizer updates on source-verified NaNROUTER Gutenberg books from the original full R19 release.

- Parent accepted R18 SHA-256: `dd040106568786e7fbad2a880f4010635b626b51c3697e2ab08712905a2de91f`
- R21 candidate source SHA-256: `df6a71caae7c508784e1cc4ee07f5d7baf4df09b35a4da6b0ce85734536054f9`
- R22 candidate source SHA-256: `d004988717113e557d61d84b75a6987df83d281ea117d8939829394580b92c5b`
- Train Gutenberg IDs: 240, 784; development IDs: 866, 970; historical-test IDs: 1063, 710
- Configuration: word-aligned chunk span 12, unroll 2, max 16 chunks per book, CPU threads 2, step seed 24024
- Four expert SwiGLU FFNs, 384-wide each, top-2 sparse routing; recurrent state 96D, token embeddings 64D
- The locally replayed runner used **256-token local copy history** and trained LM CE + multi-positive value loss + MoE balance; it **did not include optional synthetic semantic auxiliary training**. The experimental source/archive, not current master CLI defaults, is authoritative for exact replication.
- Single AdamW with nine groups, 123 optimizer states; R18/R21/R22 moment histories migrated.
- One saved experimental candidate at step 20, not four or five independent models and not an accepted checkpoint.

## Actual held-out measurements (not blind)

| Metric | Initial unified model | 20-update model | Change |
|---|---:|---:|---:|
| Development bits/byte ↓ (2,227 bytes) | 3.770809 | **3.577809** | −0.193000 |
| Development nonspace NLL ↓ | 5.058157 | **4.809186** | −0.248971 |
| Historical-test bits/byte ↓ (2,249 bytes) | 3.180594 | **3.155316** | −0.025279 |
| Historical-test nonspace NLL ↓ | 5.145396 | **5.112242** | −0.033154 |

Historical test books have been examined by earlier PAL research. This is **not a fresh blind test** and cannot establish frontier generalization. Development is only two short sampled books; do not extrapolate a scaling law from these numbers.

### History-specific memory gates

- At development step 20, correct-vs-unrelated-document advantage: **+0.00000794 nats/token** over 1,173 donor-evaluable tokens. Near zero; not a validated semantic or historical-memory gain.
- Correct-vs-shuffled-values advantage: **−0.00000080 nats/token**. Slightly unfavorable.
- Real evaluation reached a maximum of **709 tokens** (development) and **643 tokens** (historical test), **not 8K or 32K**.
- Separate synthetic index regression (no model-prediction claim) reached **31,158-token** source distance; key/value causality and future-suffix invariance passed.

**Conclusion:** Joint trunk training improves small-sample next-token loss. We have **not** demonstrated that the new R23 retrieval contributes useful correct-document information. Freeze promotion until independent memory-use and new-book tests pass.

## Verified recovery

Local reloading of the saved candidate reproduced both development and historical-test scores exactly. Its one optimizer contains all nine groups and 123 Adam states. Accepted checkpoint source digest remained unchanged. The cumulative recovery ZIP was tested with a full CRC check:

- Artifact: `FA2_R24_UNIFIED_CPU20_RECOVERY.zip` (79,595,799 bytes)
- ZIP SHA-256: `2ba52e0a9cb3afd62ae9edda283660209253b99bfb6b93ae422ce7dc73afb58e`
- The ZIP includes all local replay Python source, original R21/R22 adapter inputs, step-20 unified candidate checkpoint, metrics, logs, and a SHA-256 manifest. **It does not duplicate the original full R19 tokenizer/books/checkpoint**; those remain required to run the original PAL initializer.

The GitHub source workflow tests architecture safety and checkpoint lineage. It is **not** evidence of this 20-step training, which was executed locally, or an assertion that the local replay source is byte-identical to `master` implementation.

## Next promotion requirements

1. Achieve clear, statistically meaningful positive correct-vs-unrelated-document and correct-vs-shuffled-value benefit on dev and newly held-out authors.
2. Actually evaluate the 2K, 8K and 32K distance bands with observed retrieved slots, not just an addressable index.
3. Exercise semantic paraphrase/multi-hop tasks, nonspace regression and byte-normalized comparison against a strong local baseline.
4. Continue **the same candidate checkpoint and optimizer state**, not average independent branches, and retain the accepted R18 parent separately.
