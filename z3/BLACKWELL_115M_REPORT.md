# Blackwell 115M 10-minute smoke (subword diagnostic)

> **Superseded for parity by `BLACKWELL_BYTE_115M_REPORT.md`.** The previous
> implementation was a 273-way byte model; the subword run below is not the
> comparable architecture. Byte-model parity run: 112,899,328 params,
> 717 steps, top-1 55.69% / top-5 84.26% on FineWeb valid.

Session: <https://sb-88942244062351a9.sb.molab.run/>  
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB  
PyTorch: 2.11.0+cu130, CUDA 13.0  
Container: 32 GB RAM target, 4 allocated cores (20 visible on host)

## Data

- Source: public `nameonlu/fineweb-edu` shard, 505,118,965 bytes
- Slice: 495,016,068 train bytes + 10,102,897 validation bytes
- Tokenizer: bundled layered/v4, 64,060 vocabulary
- Bins: uint16, 106,455,541 train tokens, 2,187,599 validation tokens
- Prep: 4-worker newline-chunk tokenizer, 8 MiB chunks
- Validation is a separate 2% split

## Model

| Item | Value |
|---|---:|
| Parameters | 115,042,368 |
| Layers | 10 |
| d_model / heads / KV heads | 320 / 10 / 2 |
| SwiGLU hidden | 1,280 |
| MoE layers / experts / top-k | 3 / 12 / 2 |
| Expert hidden | 1,344 |
| MTP heads | 2 (t+2, t+3) |
| Context | 512 |
| Vocab | 64,060 |
| Concept objects | C(256,2), C(256,3) top-k |
| Pair-ID embedding | 32,640 × 64 |
| Full-byte probe | 48 bytes × 256 classes |
| Long-context headroom | 8,192 RoPE, activation checkpointing |

The configuration intentionally includes the integrated probe, object-ID,
hard-negative, learned-router, full-byte, cross-resolution, MTP, and
long-context paths.

## Training

The first OOMs were real memory bugs, not context/connectivity failures:

- batch 64 OOM in the triple-object candidate tensor,
- batch 56/48 OOM when the MTP teacher materialized a dense `[B,T,64060]`
  one-hot,
- batch 32 OOM in deep-supervision loss,
- batch 16 with activation checkpointing fits.

A direct ground-truth embedding mixture replaced the dense MTP teacher one-hot,
which removes a ~12 GB allocation and preserves the same math.

| Metric | Value |
|---|---:|
| Steps completed | 454 |
| Wall cap | 600s |
| Train wall | 602.89s |
| Tokens seen | 3,719,168 |
| Step time | 1.328s |
| Throughput | 6,198 tokens/s |
| Params | 115,042,368 |
| Non-finite skips | 0 |
| FineWeb valid loss | 6.19667 |
| FineWeb valid BPB | 1.94412 |
| FineWeb valid BPC | 1.95357 |
| FineWeb top-1 / top-5 | 17.65% / 31.15% |

The 500-step log interval was configured, but the 600s cap stopped the run at
454 steps, so no intermediate 500-step line exists. The next run should raise
throughput or lower the reporting interval.

CPU reference: 4h, 2,048,000 tokens, 144.6 tokens/s. The Blackwell smoke did
3.72M tokens in 10 minutes, about **42.9x** CPU token throughput, or roughly
**7.2 CPU-hours** in ten minutes.

## Zero-shot test

Evaluation used the v4/layered byte table for exact target-byte denominators.
Tokenizer: v4 (same 64,060 subword artifact). Full teacher-forced coverage.

| Corpus | Targets | Eval bytes | Bits/token | BPB | BPC | Top-1 | Top-5 | Window BPB ± std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| WT2 | 296,497 | 1,256,430 | 11.86244 | **2.79934** | **2.80249** | 6.36% | 13.83% | 2.81950 ± 0.26489 |
| WT103 | 282,681 | 1,287,539 | 12.13010 | **2.66318** | **2.66715** | 6.26% | 13.73% | 2.67792 ± 0.22580 |
| Combined | 579,178 | 2,543,969 | 11.99308 | — | — | 6.31% | 13.78% | — |

MTP top-1: WT2 t+2 5.41%, t+3 3.36%; WT103 t+2 4.93%, t+3 3.46%.
Evaluation wall time: 22.76s.

## Why token top-1/top-5 look “crashed”

They are not comparable to the old 60–70% numbers. The old 115M reference used
a **256-way byte vocabulary**; this model uses a **64,060-way subword
vocabulary**. A 64k token top-1 of 6.3% is not a 6% byte model:

- 64k random top-1 is 0.00156%, so 6.36% is ~4,000× random.
- 64k random top-5 is 0.0078%, so 13.83% is ~1,770× random.
- The old byte model’s 70%/90% top-1/top-5 are over 256 classes.

For functional parity, map the token distribution to the next target token’s
first-byte marginal and measure 256-way byte top-k. The checkpoint gives:

| Corpus | Targets | Byte top-1 | Byte top-5 |
|---|---:|---:|---:|
| WT2 | 296,497 | **71.78%** | **84.30%** |
| WT103 | 282,681 | **72.64%** | **89.94%** |

Old 115M byte reference (WT103): **68.55% / 87.28%**. The Blackwell checkpoint
is at or above that byte-level parity despite only 3.72M training tokens.

Reproduce with:

```sh
python z3/eval_byte_parity.py \
  --checkpoint runs/blackwell_115m/final.pt \
  --corpus wt2=data/eval/wt2_test.txt \
  --corpus wt103=data/eval/wt103_test.txt \
  --tokenizer v4 --ctx 512 --batch-size 8 --device cuda \
  --out runs/blackwell_115m/byte_parity.json
```

The 115M model saw only 3.72M tokens. It is undertrained relative to the CPU
37M/4h checkpoint and is not a frontier comparison; the byte-level top-k parity
only shows that the subword-to-byte evaluation bridge is working correctly.

## Artifacts

Remote session paths:

- `/marimo/nanlm_experimental/runs/blackwell_115m/final.pt`
- `/marimo/nanlm_experimental/runs/blackwell_115m/best.pt`
- `/marimo/nanlm_experimental/runs/blackwell_115m/result.json`
- `/marimo/nanlm_experimental/runs/blackwell_115m/zero_shot.json`
- `/marimo/data/fineweb_bins/{train,valid}.bin`

No checkpoint or corpus was downloaded to the phone.
