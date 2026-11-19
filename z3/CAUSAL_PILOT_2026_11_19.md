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

| Eval | 24.4M tokens | 49.15M tokens | **70.5M tokens** |
|---|---:|---:|---:|
| In-domain valid | 1.5310 / 25.06% | 1.3652 / 30.45% | **1.3165 / 32.35%** |
| External FineWeb-Edu article-level | 1.6079 / 21.60% | 1.4603 / 25.70% | **1.4118 / 27.15%** |
| PTB test | 2.3011 / 13.85% | 1.9241 / 23.82% | **1.8928 / 25.63%** |

External top-5: 38.38% → 43.87% → **45.86%**. PTB top-5:
24.69% → 35.24% → **36.99%**.

The third 20 minutes added 21,348,352 tokens. Gains are still positive but
diminishing (external −0.049 BPB, PTB −0.031 BPB). PTB is now moving slowly;
the external FineWeb number is more responsive.

For comparison:

| Model | External FineWeb | PTB |
|---|---:|---:|
| Leaky full sidecar (115M/30m checkpoint) | 0.1076 | 0.7850 |
| Causal eval of leaky weights | 6.1305 | — |
| Causal pilot, 24.4M tokens | 1.6079 | 2.3011 |
| **Causal continuation, 49.15M tokens** | **1.4603** | **1.9241** |

## Per-domain held-out at 70.5M tokens

Causal model, `test.bin` slices (chunk-level held-out; still optimistic):

| Domain | BPB | Top-1 | Top-5 |
|---|---:|---:|---:|
| FineWeb-Edu | 1.4268 | 26.55% | 45.20% |
| Code | 1.1980 | 45.48% | 65.43% |
| Math | 1.1943 | 45.51% | 66.69% |
| NaNROUTER news | 1.6291 | 21.82% | 39.42% |
| AG News | 1.6353 | 25.61% | 40.55% |
| Cosmopedia | 1.0866 | 31.26% | 48.81% |
| arXiv | 1.3504 | 30.47% | 48.64% |
| PubMed | 1.5172 | 27.85% | 43.82% |
| TinyStories | 1.0945 | 39.93% | 64.59% |
| FineWeb-70M | 1.6118 | 23.97% | 41.89% |
| Law | 1.5283 | 26.99% | 44.57% |
| Patents | 1.2640 | 30.25% | 47.85% |
| Finance | 1.2611 | 27.82% | 46.98% |
| Security | 1.4678 | 32.71% | 49.77% |
| Standards | 1.6213 | 26.08% | 43.24% |
| Chemistry | 1.0965 | 35.98% | 55.56% |
| Recipes | 1.4773 | 30.13% | 49.46% |
| QA | 1.4388 | 26.19% | 44.35% |

The table is now in a plausible range (1.09–1.64 BPB) rather than the leaky
0.036–0.400. Code/Math are easiest; news and standards are hardest. The
chunk-level split still means adjacent chunks of one article can cross the
boundary, so treat these as same-source held-out, not external.

## Conclusion

The causal model is undertrained but not broken. The external and in-domain
numbers are close (1.412 vs 1.317), which is what a small model that has
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
- Kaggle: `shem42/z4-causal-60m-84m` (70.5M-token checkpoint)
  - `final.pt`, `best.pt`, `config.json`, `result.json`, `eval_ptb.json`,
    external FineWeb JSON, per-domain JSON, logs
- Code: `z3/concept_loop.py`, `z3/v3_model.py`, `z3/train_v3.py`,
  `z3/molab_launch.sh`
