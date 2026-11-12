# nanlm_experimental

Experimental branches and controlled ablations for the NaNLM / Poincaré Mirror
language model. This repo is for negative results as much as positive ones; the
logs include failed runs and protocol corrections.

**Latest experiment (2026-11-12):** R31 compute-matched MoE. A 151M-parameter
hybrid using ~21M active MAC/token ties the 21M dense baseline at 1.8M training
positions (`2.4616` vs `2.4621` BPB), while geometry-only routing is slightly
worse (`2.4881` BPB). Full details:

- [`r31/EXPERIMENTS.md`](r31/EXPERIMENTS.md) — chronological experiment log
- [`r31/README.md`](r31/README.md) — r31 system, code changes, how to run
- [`r31/docs/R31_10MIN_REPORT.md`](r31/docs/R31_10MIN_REPORT.md) — latest report

## Repo layout

```
r31/
  code/          patched Poincaré Mirror LM, probe, generation
  kernels/       Kaggle kernel scripts (bench, 10min, 10min-v2, cancelled 50min)
  datasets/      Kaggle dataset metadata for the code bundle
  results/       summaries, probes, train logs, sampled generations
  docs/          reports
```

## Headline result

| Arm | Total params | Active MAC/token | BPB (65,536 held-out) |
|---|---:|---:|---:|
| Plain dense | 21.4M | ~21.4M | 2.4621 |
| Hybrid (dense trunk + additive 128-pair top-2 MoE) | 151.0M | ~21.1M | **2.4616** |
| Hybrid, geometry-only router | 151.0M | ~21.1M | 2.4881 |

The earlier parameter-matched comparison made the hybrid look 0.27 BPB worse.
Matching active compute removes that loss. This is a 1.8M-position pilot, not
the 10M-position target.

## License

MIT. See [LICENSE](LICENSE).
