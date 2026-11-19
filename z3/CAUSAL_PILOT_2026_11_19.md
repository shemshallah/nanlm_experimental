# Causal pilot results (2026-11-19)

First from-scratch run after removing the future-token sidecar leak. This is
the honest baseline, not a leaky score.

## Fix

- `concept_loop._segment_pool` is now **prefix-only**: token `t` averages
  only `j <= t` in the same word/sentence, using cumsum + segment start.
- `ConceptEncoder` returns a zero global context.
- `ConceptLoop` global loop state is zero in causal mode.
- `Block` route feedback and V3LM memory-summary feedback are skipped in
  causal mode.
- New `--causal-concept` flag keeps **all 84,455,112 parameters**; the
  feedback modules are present but their global paths are neutralized.
  This is not the 82.1M parameter-reduced variant.

## Pilot

| Field | Value |
|---|---|
| Parameters | 84,455,112 |
| Tokens | 24,363,008 |
| Wall time | 1,203.8 s |
| Batch / ctx | 32 / 512 |
| Throughput | ~20.2k tokens/s |
| VRAM cap | 37 GB |
| Peak in-domain valid | **1.5310 BPB / 25.06% top-1** |

## Clean evaluation

| Test | BPB | Top-1 | Top-5 |
|---|---:|---:|---:|
| **External FineWeb-Edu, article-level 20% test** (shard 013, 2M tokens) | **1.6079** | 21.60% | 38.38% |
| **PTB test** | **2.3011** | 13.85% | 24.69% |

## 20-minute continuation

Resumed the 24.4M-token checkpoint for 20 more minutes: +24,788,992
tokens, total **49.15M**, 1,194.1 s.

| Eval | 24.4M tokens | 49.15M tokens |
|---|---:|---:|
| In-domain valid | 1.5310 / 25.06% | **1.3652 / 30.45%** |
| External FineWeb-Edu article-level | 1.6079 / 21.60% | **1.4603 / 25.70%** |
| PTB test | 2.3011 / 13.85% | **1.9241 / 23.82%** |

External top-5 improved 38.38% → **43.87%**; PTB top-5 24.69% → **35.24%**.

For comparison:

| Model | External FineWeb | PTB |
|---|---:|---:|
| Leaky full sidecar (115M/30m checkpoint) | 0.1076 | 0.7850 |
| Causal eval of leaky weights | 6.1305 | — |
| Causal pilot, 24.4M tokens | 1.6079 | 2.3011 |
| **Causal continuation, 49.15M tokens** | **1.4603** | **1.9241** |

The causal model is undertrained but not broken. The external and in-domain
numbers are close (1.460 vs 1.365), which is what a small model that has
not memorized the corpus should look like. The near-zero 0.1076 was the
future-token channel.

## Coexistence

The pilot ran under `torch.cuda.set_per_process_memory_fraction(37/97)`
while another training used ~38 GB. Total GPU use was ~72 GB; neither job
was evicted. External evaluation peak was 2.18 GB.

## Detached execution

`z3/molab_launch.sh` wraps `nohup setsid` and writes a pidfile/log.
Python launches also use `start_new_session=True`. This protects against
notebook/terminal SIGHUP and stdout closure, **not** against the Molab
sandbox being stopped or restarted. Checkpoints on Kaggle plus a resumable
`--resume` command remain the only protection against sandbox loss.

## Interpretation

The architecture can learn causally. At 24M tokens it reaches 1.61 external
FineWeb / 2.30 PTB. A longer causal run should improve both; the old
leaky 0.1/0.8 numbers must be discarded.

## Persistence

- Kaggle: `shem42/z4-causal-pilot-84m` (24.4M-token checkpoint)
- Kaggle: `shem42/z4-causal-40m-84m` (49.15M-token checkpoint)
  - `final.pt`, `best.pt`, `config.json`, `result.json`, `eval_ptb.json`,
    external FineWeb JSON, logs
- Code: `z3/concept_loop.py`, `z3/v3_model.py`, `z3/train_v3.py`,
  `z3/molab_launch.sh`
