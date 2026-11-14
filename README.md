# nanlm_experimental

Experimental branches and controlled ablations for the NaNLM / Poincaré Mirror
language model. This repo is for negative results as much as positive ones; the
logs include failed runs and protocol corrections.

**Frontier v4 (2026-11-13):** the z3 text path now uses a bundled 64k v4 BPE
tokenizer, factorized 128-d embeddings, tied low-rank MTP output, 1k context,
and a tuned six-level layered tokenizer. The layered artifact aligns bits,
UTF-8 bytes, subwords, model tokens/specials, words, and sentences losslessly.
See [`z3/LAYERED_TOKENIZER.md`](z3/LAYERED_TOKENIZER.md),
[`z3/FRONTIER_FIX_PLAN.md`](z3/FRONTIER_FIX_PLAN.md), and the
[`z3/FRONTIER_SCALING_PLAN.md`](z3/FRONTIER_SCALING_PLAN.md) for the measured
Kaggle/CPU parameter ladder and the C(256,2+3) object-growth path.

**Previous experiment (2026-11-12):** R32 multidomain. A 151M-parameter hybrid
using ~21M active MAC/token ties the 21M dense baseline on 16 domain releases
(`2.9186` vs `2.9180` BPB, 1.8M positions), and routing becomes strongly
domain-dependent: code and multilingual use ~39 effective pairs while most
domains use ~100.

- [`r32/EXPERIMENTS.md`](r32/EXPERIMENTS.md) — latest experiment log
- [`r32/README.md`](r32/README.md) — multidomain corpus, domain-token scheme, hierarchical routing
- [`r32/colab/colab_multidomain_30min.ipynb`](r32/colab/colab_multidomain_30min.ipynb) — 30-minute Colab run
- [`r32/results/`](r32/results/) — per-domain BPB + 16×128 routing matrices
- [`r31/EXPERIMENTS.md`](r31/EXPERIMENTS.md) — previous compute-matched/geometry log

## Repo layout

```
r32/
  code/          domain-token Mirror LM, hierarchical router, corpus builder, probe
  colab/         30-minute multidomain notebook + quota instructions
  kernels/pilot/ Kaggle 16-domain pilot
  datasets/      source provenance and rebuild instructions
  results/       per-domain probes, summaries, train logs
  EXPERIMENTS.md latest multidomain experiment + hierarchical implementation

r31/
  code/          compute-matched MoE + router-mode code
  kernels/       bench, 10min, 10min-v2, cancelled 50min
  datasets/      Kaggle dataset metadata
  results/       probes, summaries, generations
  docs/          reports
```

## Headline results

### r32 multidomain (latest)

| Arm | Total params | Active MAC/token | BPB (131,072 held-out) |
|---|---:|---:|---:|
| Plain dense + domain token | 21.4M | ~21.4M | **2.91799** |
| Hybrid + domain token | 151.0M | ~21.1M | **2.91856** |

Tie within noise. Domain routing is not uniform: code effective pairs 38.5
(pair 102 = 18.3% of code traffic), multilingual 39.6, most other domains
~100. Mean pairwise JS divergence between domain routing distributions is
0.109; code↔multilingual is 0.390.

### r31 compute-matched / geometry (previous)

| Arm | BPB (65,536 held-out) |
|---|---:|
| Plain dense | 2.4621 |
| Hybrid | **2.4616** |
| Hybrid, geometry-only | 2.4881 |

Matching active compute removed the earlier 0.27 BPB parameter-matched loss.
Geometry-only did not beat linear+geometry.

Both are short pilots, not frontier-scale runs. See each experiment log for
caveats.

## License

MIT. See [LICENSE](LICENSE).
