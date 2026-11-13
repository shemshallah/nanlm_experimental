# FA2 R28 — four-chunk recurrent BPTT CPU pilot (experimental)

**Date:** 2026-11-12 Pacific. **Status:** source and measured CPU candidate; **R27 remains the recovery/accepted experimentation reference**. No new training model lineage or Git branch.

## Architecture

R28 adds `MirrorRecurrentResidual` **inside the existing PAL subword GRU recurrence**. Its identity-initialized amplitude is zero, so at construction its logits are bitwise identical to inherited R27. Each recurrence uses a mirror of the 96D recurrent vector, a rank-64 gated SwiLU/SwiGLU-style correction, and a per-channel gate. It shares the original decoder and every inherited R18–R27 component. This is NOT an external next-token ensemble.

Training backpropagates once through **four consecutive 12-word chunks** before detachment (rather than the preceding two chunks); document recurrent state, previous tokens, and causal attention K/V remain continuous across optimizer updates. The exact token count per step varies with the lossless subword tokenizer. There is no claim of full 132K-token gradient propagation.

The existing R24 inference wrapper is additionally guarded against attempting R25 cache features before query position 3 or R26/R27 high-order features before query position 6. This repairs the crash seen when replaying a document from token zero; for older positions the prior source crashed rather than supplying valid predictions.

## CPU experiment and selection

Restored the verified R27 archive from PAL R19 + R27, without loading intermediate R24/R25/R26 binaries. Used Gutenberg/NaNROUTER training books 240 and 784 and historical development books 866 and 970, 4 consecutive 12-word chunks/update, CPU PyTorch, 16 R28-only Adam updates. Only the new R28 tensors had gradients; original tensors, all **144 inherited optimizer states**, and the 12 original stream snapshots were checked for exact equality.

| Metric (lower is better) | Inherited R27 | R28 step 16 | Improvement |
|---|---:|---:|---:|
| Development BPB | 2.9491770687 | 2.9479519785 | 0.0012250902 |
| Development nonspace NLL | 4.7152275672 | 4.7145940504 | 0.0006335169 |
| Dev book 866 BPB | 2.9887978909 | 2.9876119691 | 0.0011859218 |
| Dev book 970 BPB | 2.9068037944 | 2.9055368149 | 0.0012669795 |

Measured 1,132 bytes / 638 target subword tokens over the two dev books. Changes are small and on historical development material, so this result does **not** substantiate frontier performance or independent generalization. The first two-update single-book pilot covered only 320 bytes and improved by about 0.0000052 BPB. The 16-update trial is the better diagnostic; its selected step 16 is not a release promotion.

## Checkpoint and causal audits

- Immutable R27 ancestor: SHA-256 `0f4cdb8ce41b62472b6c8408296b496dfb3e801b388289453c0cdcc720f5b3f8`.
- R28 candidate: one unified checkpoint with **13 Adam parameter groups**, **153 Adam states** (144 inherited + nine new), and **195 model state tensors**. Original R27 tensors/optimizer histories and 12 stream records are retained.
- Independent local R28 cold restoration passed from R19 + R27 + this R28 trial checkpoint.
- Synthetic test passed: exact identity at zero amplitude; no future-label leakage; book isolation; nonzero gradient from fourth 12-token chunk into first-chunk state (48-token controlled example).
- `r28_cpu_continue.py` is a small experimental training driver. This first checkpoint does not contain enough transient R28 train-stream snapshots to reproduce an exact same-stream continuation. The original 12 inherited R27 snapshots are preserved, but additional continuation metadata will be required for bit-identical full training resume. Do not claim otherwise.

## Next acceptance gates

1. A substantially wider and author-disjoint sealed set; report confidence intervals and matched controls.
2. Continuous 33K/65K/132K forward replay with R27 and R28 using identical text; guard against losing R27's remote-history benefit.
3. Attention-gradient diagnostics and a matched 2-vs-4-vs-8 chunk BPTT ablation with the same token budget.
4. Semantic paraphrase and hard-negative retrieval against the existing lexical memory.
5. Exact continuation snapshots, strict package manifest, and immutable SHA audit before a new latest recovery archive replaces R27.

The original R19 and latest R27 recovery archives remain the recovery source of record. No promotion or Library cleanup is authorized by these limited measurements.

## Compact exact recovery overlay

`r28_pack_overlay.py` exported **497,239 bytes** of R28 weights, the R28 Adam group with all nine new Adam states, parent digest, and selected development metadata. This eliminates redundant R27 tensors from the experimental artifact. The full R28 candidate had occupied **129,492,981 bytes**; the overlay is about **99.62% smaller**.

The overlay SHA-256 is `e29155c5bde6d33237f65d13b662e8ccebef3cd8adf89b075390993e20673fd2`.

An independent cold restore using **original PAL R19 + original FA2 R27 recovery + this R28 overlay** produced a unified model whose **195 state tensors and all 153 optimizer states exactly equal** the full R28 candidate. This is exact tensor/Adam equivalence, not only matching loss. The R28 overlay still lacks transient R28 training-cursor snapshots: use it as an experimental model recovery and not an exact same-stream training restart. It is not an accepted R27 replacement.

```bash
# After extracting PAL R19, then FA2 R27 and the R28 overlay package:
PYTHONPATH=source python source/r28_restore_unified.py --root .
PYTHONPATH=source python source/test_r28_recurrent.py
```

## Frozen historical-book follow-up (no updates)

After step 16 was fixed, a separate frozen evaluation compared R27 and R28 on the first 16 contiguous 12-word chunks from historical test books 710 and 1063. The training and development selection used neither book in **this** R28 pilot. However, PAL has consulted these books in earlier rounds; **this is not a clean blind test**. No training updates or re-selection followed this probe.

| Test measure | R27 | R28 step 16 | Improvement |
|---|---:|---:|---:|
| Combined BPB (2,249 bytes, 1,298 tokens) | 2.8982332231 | 2.8970735305 | 0.0011596926 |
| Nonspace NLL | 4.6108296495 | 4.6104817182 | 0.0003479313 |
| Book 710 BPB | 3.1951315350 | 3.1943500646 | 0.0007814704 |
| Book 1063 BPB | 2.5734143064 | 2.5718408233 | 0.0015734830 |

Both historical test books improved, but only over a very short opening prefix. R28 remains unpromoted pending author-disjoint and 33K/65K/132K causal replay. Source: `source/r28_historical_probe.py`; frozen measurements: `reports/r28_historical_frozen.json`.
