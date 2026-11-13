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
