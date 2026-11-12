# r31 results

All files are small text/JSON. Model checkpoints are **not** committed
(`.gitignore` excludes `*.pt`). The checkpoints remain in the Kaggle kernel
outputs linked below.

## Latest experiment (r31-10min-v2)

| File | Contents |
|---|---|
| `r31_10min_v2_summary.json` | Full two-arm summary: design, configs, train timings, probes, generations, comparison |
| `hybrid151_both.probe.json` | 65,536-position held-out probe for the linear+geo hybrid: BPB, per-book BPB, 128-bin routing counts, specialization, centers/geometry |
| `hybrid151_geo.probe.json` | Same for geometry-only routing |
| `hybrid151_both.gen.json` | 200-byte sampled generation + repetition metrics |
| `hybrid151_geo.gen.json` | Same for geometry-only |
| `hybrid_both.train.log` | 100-step validation curve, 151M both-router |
| `hybrid_geo.train.log` | 100-step validation curve, 151M geometry-only |

## Plain reference (r31-10min-v1)

The first 10-minute kernel trained the plain baseline successfully but the
151M hybrid hit an OOM in the `gather` dispatch path.

| File | Contents |
|---|---|
| `r31_10min_v1_summary.json` | Plain result + failed hybrid command/log reference |
| `plain.probe.json` | 65,536-position held-out probe for the 21.4M dense plain |
| `plain.gen.json` | 200-byte sampled generation + repetition |
| `plain.train.log` | Plain validation curve |

## Benchmark

| File | Contents |
|---|---|
| `bench.json` | T4 throughput/VRAM for plain, hybrid 151M, control 21M at batch 8/16/32 |

## Kernel outputs

- Latest: https://www.kaggle.com/code/shem42/r31-10min-v2
- Plain reference: https://www.kaggle.com/code/shem42/r31-10min
- Benchmark: https://www.kaggle.com/code/shem42/r31-bench

## Reading a probe

```sh
jq '{bpb, ce, top1, union: .union.active_pairs, per_book}' \
  r31/results/hybrid151_both.probe.json

jq '.layers["1"].specialization.most_specialized[:5]' \
  r31/results/hybrid151_both.probe.json
```

The probe JSON includes the full 128-element `counts` array for each MoE layer
and the union, so utilization can be re-plotted without rerunning the model.
