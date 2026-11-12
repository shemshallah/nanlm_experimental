# r32 results

Small JSON/text artifacts from the latest multidomain pilot. Checkpoints are
not committed (`*.pt` is ignored); they remain in the Kaggle kernel output.

| File | Contents |
|---|---|
| `r32_summary.json` | Full two-arm summary: design, train timings, probes, comparison |
| `hybrid151_domain.probe.json` | 131,072-position per-domain probe for the hybrid: BPB/top-1, 128-bin routing counts, `routing_matrix[16][128]`, specialization KL, domain-token norms, geometry |
| `plain21_domain.probe.json` | Same for the 21M dense baseline (no MoE; routing fields are empty) |
| `hybrid151_domain.train.log` | 100-step validation curve, hybrid |
| `plain21_domain.train.log` | 100-step validation curve, plain |

The probe JSONs are large because they include full routing counts per layer
and per domain. Useful queries:

```sh
jq '.aggregate' r32/results/hybrid151_domain.probe.json
jq '.per_domain[] | {domain, bpb, top1, eff: .union.effective_pairs}' \
  r32/results/hybrid151_domain.probe.json
jq '.routing_matrix' r32/results/hybrid151_domain.probe.json
```

Kernel: https://www.kaggle.com/code/shem42/r32-multidomain-pilot
