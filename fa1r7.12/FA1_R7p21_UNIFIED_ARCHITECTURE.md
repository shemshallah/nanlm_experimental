# FA1 R7.21 — One unified R7 model and five continuous optimizer histories

**Date:** 2026-11-12  
**Branch policy:** `shemshallah/nanlm_experimental` `master` only; stay within existing `fa1r7.12/` source directory. No new architectural branch, no second model lineage, no FA2 integration.  
**Current model state:** unified R7.21 experimental checkpoint built; accepted R7.13 parent remains at 14,416 updates and is **not promoted**.  
**Separate models:** none; the inherited near/far adapters are now registered as layers of the same R7 `torch.nn.Module`, with one R7.21 candidate state and a single inherited collection of five optimizer histories.

## Unified R7 model

```
           R7.7 trunk (96-d shared embeddings and hierarchical word head)
            + R7 rank-64 S+/S- mirror recurrent pathway
            + R7.11 directed pair writer
            + R7.12 calibration parameters (disabled in the parity path)
                                 │
             ONE causal prediction and hidden-state stream
                                 │
       ┌─────────────────────────┴─────────────────────────┐
       │                                                   │
 R7.16 8K/32K past (8 weighted n-gram channels)       R7.18 32K–128K past
                                                        (4 weighted n-gram channels)
       │                                                   │
 near trust gate                                        far residual gate
       └─────────────────────────┬─────────────────────────┘
                      R7.20 packed exact 128K index
                          causal, CPU-efficient
                                 │
                     ONE probability mixture output
```

All learned parameters are in one `UnifiedR7` model `state_dict` (95 named tensors). All optimizer moments, including near and far groups, live in the inherited **fifth** R7 optimizer; original groups 1–5 are preserved. The two newest groups are appended to the existing fifth optimizer, not given separate optimizer objects.

## Candidate migration and integrity

| Item | Value |
|---|---|
| Accepted R7.13 checkpoint SHA-256 | `54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d` |
| Accepted original update step | 14,416 |
| Near R7.16 optimizer moments | 6 parameters, exact original Adam moments |
| Far R7.18 optimizer moments | 6 parameters, exact original Adam moments |
| Original optimizer histories | 5, existing groups and moments unchanged |
| Unified model tensor count | 95 |
| Candidate checkpoint size | 94,805,735 bytes |
| Candidate SHA-256 | `7d2ff0c17430075e3fc6ddeb02f61bf447c07eed027ae77afac0506737734b1f` |
| Independently reloadable? | Yes, with corpus tree/vocab and SentencePiece tokenizer assets; no parent or historical overlay files required |
| Promoted to accepted? | **No** |

### Real checkpoint numerical parity

R7.20 frozen near/far runtime vs `UnifiedR7` candidate loaded from its single checkpoint, 1,024 consecutive historical test positions starting at 134,000, batch 16:

| Output | Maximum absolute NLL error |
|---|---:|
| Inherited baseline R7 model | 0.0 nats/word |
| Near R7.16 mixture | 0.0 nats/word |
| Far R7.18 mixture via packed index | 0.0 nats/word |

Additional tests: one near+far joint Adam step with *both* gradient paths nonzero; inherited trunk and old optimizer moments exactly unchanged; saved one-step state recovered identically; original checkpoint digest unchanged. A separate recovery directory containing corpus/tokenizer links but **no accepted checkpoint or overlay files** successfully loaded the unified candidate with all 95 tensors and five optimizers.

This is parity on a repeatedly consulted **historical test split**; it does **not** measure new-book capability or semantic retrieval.

## 96-step real CPU training pilot (rejected)

On 256 historical train positions and 128 historical development positions, R7.21 jointly optimized the near/far gates with one continuing fifth AdamW optimizer, a small learning rate (1e-4 for each new group), and a causal shuffled-history contrastive penalty. The old trunk, calibration parameters, and their optimizer histories remained frozen. All selections used only the development split.

| Development metric | Initial unified state | Update 96 | Decision |
|---|---:|---:|---|
| Correct-history NLL (nats/word; lower better) | 5.451543 | 5.453338 | Regressed |
| Correct advantage over far-shuffled history | +0.032360 | +0.027225 | Regressed |
| Correct advantage over independent train-split donor | +0.013982 | +0.006506 | Regressed |

**No update was selected.** Both baseline and optimizer state in the published unified checkpoint remain at the original migrated candidate; 96 attempted steps are **not** added to accepted training count.

## Canonical continuation and restoration

```bash
export PYTHONPATH="$(pwd)/fa1r7.12/source:$(pwd)/fa1-full/source"
python fa1r7.12/source/r7_current.py \
  --root /path/to/FA1_full \
  --checkpoint /path/to/r7_unified_candidate.pt \
  --split test --start 134000 --positions 128 --mode auto
```

For the one-time conversion from R7.13 and pinned R7.16/R7.18 adapters:

```bash
python fa1r7.12/source/fa1_r721_unified.py \
  --mode bootstrap --root /path/to/FA1_full \
  --near /path/to/fa1_r716_225.pt --far /path/to/fa1_r718_full_candidate.pt \
  --output /path/to/r7_unified_candidate.pt
```

Later loads require only the one unified `.pt`, tokenizer and corpus-tree metadata. The candidate SHA-256 must be checked before `torch.load` (PyTorch checkpoint files should come from trusted sources). The research source ZIP excludes all parent weights, corpus database files, and the large candidate `.pt` and instead carries its digest and reproduction scripts. The full R7.21 candidate is stored as a separate single recoverable checkpoint in the user's Library.

## Next gates within the SAME R7 architecture

1. Restore the one unified R7.21 checkpoint on a CPU worker and import checksum-pinned, independent NaNROUTER whole books with strict cross-document boundaries and overlap screening.
2. Evaluate long-context gains, order-shuffled and independent-donor controls, book-level bootstrap confidence intervals, and genuine paraphrase/conflict/multi-hop tasks **before** any model-weight promotion.
3. Optimize one R7 model jointly on much larger diverse training corpora with checkpointed fifth-optimizer state; keep original optimizer moments, use rolling CPU-safe checkpoints, and reject regression. No alternate model branch or FA2 architecture.
4. Consider capacity expansion of the **same** R7 weights (rank 64→128 and hidden 96→192) by identity-preserving migration, with matched loss and CPU checks. This is *not implemented* in R7.21.

**Decision: ONE FA1 R7 model, accepted weights at 14,416, R7.21 consolidated candidate staged; no promotion, no separate branches.**