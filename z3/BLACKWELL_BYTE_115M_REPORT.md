# Blackwell 115M byte model — previous-implementation parity

This run uses the **previous byte-level implementation**, not the 64k subword
model.

- Source code: commit `2e81fe4` (`/marimo/oldv3/z3`)
- Model: 112,899,328 parameters
- Vocabulary: 273 (256 byte values + specials)
- d_model: 256
- Layers: 8
- Heads: 8 (full MHA)
- SwiGLU: 1,024
- MoE: 4 layers, 32 routed experts, top-2, 2 shared, expert hidden 1,024
- Attention: sliding-window 512 + global every 4 layers, attention gate
- MTP: 2 tokens
- Context: 1,024
- Memory path: off
- Precision: AMP

This exactly matches the old 112.9M byte reference profile.

## Data

- Public FineWeb-Edu shard, 505,118,965 bytes
- Train: 495,016,068 byte targets
- Validation: 10,102,897 byte targets
- Byte bins: uint16 containing values 0–255, vocab size 273
- No subword tokenizer and no byte-marginal bridge

## 10-minute GPU run

GPU: RTX PRO 6000 Blackwell, 97,887 MiB  
Batch: 32 × 1,024 bytes

| Metric | Value |
|---|---:|
| Steps | 717 |
| Wall | 607.07s |
| Bytes seen | 23,494,656 |
| Step time | 0.837s |
| Bytes/s | 38,706 |
| Params | 112,899,328 |
| Final CE | 1.521139 nats/byte |
| Final BPB | **2.194539** |
| Final top-1 | **55.69%** |
| Final top-5 | **84.26%** |
| Non-finite skips | 0 |

### 500-step status

```text
step    500/200000 loss=2.3207 ce=1.5982 mtp=2.3230 lb=2.004 lr=7.38e-05
```

The run then continued to step 717 before the 600s cap. The 500-step report
exists; no 1000-step report was reached.

## Zero-shot test (same old byte evaluator)

| Corpus | Targets | BPB | Top-1 | Top-5 | MTP t+2 top-1 |
|---|---:|---:|---:|---:|---:|
| WT2 | 4,096 | **2.810258** | **53.71%** | **78.78%** | 35.60% |
| WT103 | 4,096 | **2.621964** | **52.49%** | **79.79%** | 36.11% |

The number of sampled targets is the old evaluator's fixed-position sample, not
a new full-corpus protocol. Top-1/top-5 are now the same 273-way byte metrics as
the previous implementation.

Old six-hour byte reference:

| Corpus | BPB | Top-1 | Top-5 |
|---|---:|---:|---:|
| FineWeb valid | 1.4148 | 70.38% | 90.93% |
| WT103 | 1.6969 | 68.55% | 87.28% |

The old reference saw **204.8M bytes / six hours**; this smoke saw **23.5M
bytes / ten minutes**. The gap is training scale, not a vocabulary mismatch or
a broken top-k metric.

## Artifacts

In the Molab session:

- `/marimo/runs/old_byte_115m/final.pt`
- `/marimo/runs/old_byte_115m/best.pt`
- `/marimo/runs/old_byte_115m/result.json`
- `/marimo/runs/old_byte_115m/zero_shot.json`
- `/marimo/data/byte_bins/{train,valid}.bin`
- Notebook cells: `byte_train`, `byte_report`
