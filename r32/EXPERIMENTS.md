# Experiment log

Latest first. BPB is bits per held-out byte; lower is better. The r32 corpus is
built from the new `corpora/domains/` releases, not the Gutenberg-only corpus.

---

## 2026-11-12 — hierarchical routing implementation (pending 10M run)

The next iteration adds **domain → branch → pair** routing to the 16-domain
system:

- 128 leaves = 8 branches × 16 pairs
- one branch per token (argmax), top-2 pairs inside the selected branch
- leaf score = pair score + selected branch score; branch_router gets task
  gradient through the selected branch score
- branch Switch-style balance term added to the existing pair balance term
- domain token still prefixes every window; the branch router sees it through
  the hidden state

New flags:

```sh
--hierarchical --branches 8 --pairs-per-branch 16
```

`probe_r32.py` now records `branch_routing_matrix[16][8]`, per-domain branch
effective counts, and the branch union. The code was validated with a tiny CPU
train/probe; no 10M-position hierarchical result exists yet.

A 30-minute Colab run is packaged at
[`colab/colab_multidomain_30min.ipynb`](colab/colab_multidomain_30min.ipynb).
It builds the 16-domain corpus from public releases, trains the 151M hybrid
with hierarchical routing until a 30-minute wall-clock cap, and probes 16
domains (131,072 held-out positions). On a T4 that is ~7.7M positions, not
10M; on A100/H100 it is more.

Colab has no published weekly GPU allowance; free GPU access is dynamic, Pro
uses 100 units/month, Pro+ 500–600 units/month with priority A100/H100. There
is no T100 GPU in Colab or Kaggle. Kaggle offers 30 GPU-hours/week with T4×2 or
P100.

---

## 2026-11-12 — r32 multidomain domain-token pilot (latest measured)

**Question:** on 16 domains, does the 151M-total / ~21M-active hybrid still tie
the 21M dense model at matched active compute? Does a per-document domain token
produce domain-dependent expert routing?

**Hardware:** 2× Tesla T4, ~8 minutes wall.
**Kernel:** https://www.kaggle.com/code/shem42/r32-multidomain-pilot
**Data:** https://www.kaggle.com/datasets/shem42/r32-multidomain-16domains
**Code:** https://www.kaggle.com/datasets/shem42/r32-code

### Corpus

The builder sampled 4 evenly spaced 1 MB ranges from each of the 16 public
domain releases:

| | |
|---|---|
| Domains | wikipedia-en, arxiv, pubmed, law, patents, code, math, news, recipes, security, qa, multilingual, finance, standards, humanities, chemistry |
| Train / valid positions | 43,725,193 / 4,721,470 |
| Eligible train / valid docs | 50,746 / 5,651 |
| Split | document-level 90/10, seed 2026 |
| Vocab | 273 = 257 bytes+EOS + 16 domain tokens |

Long documents (>32 KB) are truncated; exact duplicates are dropped.
`prepare_domains.py` records per-domain counts, source URLs and hashes in
`split_manifest.json`.

### Protocol

- Domain token `257+d` prepended to every window, including evaluation.
- 220 steps × batch 32 × context 256 = **1.8M positions per arm**.
- Seed 1337, same data order.
- Probe: 4 batches × batch 8 × 256 = 8,192 positions per domain; **131,072
  held-out positions total**.

| Arm | Params | Active MAC/token | Config |
|---|---:|---:|---|
| Plain + domain token | 21,397,248 | ~21.4M | dim 576, 4 layers, 24 heads, SwiGLU |
| Hybrid + domain token | 151,014,658 | ~21.1M | dim 544, 4 layers, 16 heads, 128 pairs, top-2, expert 235, additive MoE, linear+geo |

### Aggregate result

| Arm | CE | BPB | Top-1 |
|---|---:|---:|---:|
| Plain + domain token | 2.022597 | **2.917990** | 44.12% |
| Hybrid + domain token | 2.022994 | **2.918564** | 43.83% |

**Hybrid − plain = +0.000574 BPB.** A tie within noise, reproducing the r31
compute-matched tie on 16 domains instead of one Gutenberg distribution.

Training-eval at step 200 (32K positions) had plain slightly ahead:
plain CE 2.08634 vs hybrid 2.09129. The 131K probe is the final number.

### Per-domain BPB

| Domain | Plain | Hybrid | Δ (hybrid−plain) |
|---|---:|---:|---:|
| wikipedia-en | 2.8961 | 2.8912 | −0.0049 |
| arxiv | 2.8939 | 2.8997 | +0.0058 |
| pubmed | 2.6038 | 2.5878 | −0.0160 |
| law | 2.5150 | 2.5346 | +0.0196 |
| patents | 2.4755 | 2.4561 | −0.0194 |
| code | 3.7007 | 3.7351 | **+0.0344** |
| math | 3.1198 | 3.1033 | −0.0165 |
| news | 2.8441 | 2.8385 | −0.0056 |
| recipes | 2.7728 | 2.7714 | −0.0014 |
| security | 2.7728 | 2.7751 | +0.0023 |
| qa | 3.0067 | 3.0294 | **+0.0227** |
| multilingual | 4.5302 | 4.5174 | −0.0128 |
| finance | 2.4915 | 2.4775 | −0.0140 |
| standards | 2.7112 | 2.7200 | +0.0088 |
| humanities | 2.8159 | 2.8301 | +0.0142 |
| chemistry | 2.5378 | 2.5298 | −0.0080 |

Hybrid is better on 9/16 domains and worse on 7/16. The large negative cases
are code (+0.034) and QA (+0.023); the largest wins are patents (−0.019), math
(−0.017), and PubMed (−0.016). There is no aggregate win yet.

### Domain-dependent routing

128 pairs, top-2. Union over the two MoE layers, 131,072 positions per domain.

| Domain | Active | Effective | Max pair share | Top pairs (share) |
|---|---:|---:|---:|---|
| code | 124/128 | **38.5** | **18.26%** | 102 (18.3%), 13 (8.9%), 16 (4.7%), 50 (4.5%), 68 (4.4%) |
| multilingual | 126/128 | **39.6** | 10.45% | 10 (10.4%), 4 (10.4%), 9 (10.1%), 54 (9.9%), 57 (8.7%) |
| standards | 127/128 | 83.8 | 5.50% | 105 (5.5%), 7 (5.2%), 72 (4.5%), 102 (4.4%), 58 (3.1%) |
| math | 128/128 | 92.7 | 4.31% | 122 (4.3%), 53 (4.3%), 20 (3.2%), 68 (3.1%), 21 (2.6%) |
| humanities | 127/128 | 93.3 | 3.99% | 42 (4.0%), 86 (3.3%), 31 (3.1%), 78 (3.0%), 53 (3.0%) |
| recipes | 128/128 | 96.8 | 4.14% | 14 (4.1%), 42 (4.0%), 31 (2.8%), 62 (2.3%), 67 (2.2%) |
| news | 127/128 | 98.8 | 3.99% | 42 (4.0%), 13 (3.0%), 86 (2.4%), 31 (2.4%), 84 (2.0%) |
| pubmed | 127/128 | 101.2 | 3.75% | 85 (3.8%), 88 (2.9%), 18 (2.8%), 29 (2.6%), 86 (2.1%) |
| patents | 127/128 | 101.8 | 4.57% | 20 (4.6%), 50 (2.4%), 93 (2.3%), 53 (2.0%), 30 (1.9%) |
| arxiv | 128/128 | 104.0 | 3.78% | 53 (3.8%), 42 (3.1%), 20 (2.3%), 87 (1.9%), 114 (1.7%) |
| chemistry | 127/128 | 104.3 | 2.43% | 42 (2.4%), 53 (2.4%), 20 (2.4%), 50 (2.1%), 93 (2.0%) |
| wikipedia-en | 127/128 | 104.9 | 2.88% | 109 (2.9%), 42 (2.7%), 13 (2.3%), 31 (2.3%), 30 (1.9%) |
| law | 128/128 | 105.2 | 2.80% | 42 (2.8%), 31 (2.2%), 53 (2.2%), 37 (2.1%), 84 (2.0%) |
| finance | 128/128 | 105.4 | 2.50% | 13 (2.5%), 37 (2.4%), 53 (2.3%), 109 (2.1%), 31 (2.0%) |
| security | 128/128 | 107.5 | 2.39% | 42 (2.4%), 109 (2.4%), 99 (2.1%), 15 (1.9%), 30 (1.8%) |
| qa | 128/128 | 108.5 | 3.31% | 42 (3.3%), 53 (2.9%), 31 (2.3%), 87 (1.7%), 84 (1.7%) |

This is the first clear evidence that the MoE is doing domain-dependent work:
code and multilingual collapse onto a small, distinct expert set, while other
domains spread across ~100 effective pairs. It is not the old uniform 2k-per-
pair regime.

Routing-distribution divergence (Jensen–Shannon, normalized rows):

- Mean pairwise JS: **0.109**
- Most divergent: code↔multilingual **0.390**; code↔recipes 0.326;
  code↔humanities 0.310; code↔news 0.306; pubmed↔code 0.286.
- Most similar: law↔finance 0.016; wikipedia↔news 0.017; patents↔chemistry
  0.017.

Domain-token embedding norms are 0.51–0.58, so the domain tokens did not
collapse.

### Caveats

- 220 steps / 1.8M positions is a pilot; the target is 5k–10k steps and 10M+
  positions.
- The aggregate is a tie, not a win. Code and QA are clearly worse for the
  hybrid.
- **No no-domain control in this run.** The routing is domain-dependent, but
  this does not yet isolate how much of the per-domain BPB comes from the
  domain token itself.
- Byte-level tokenization is a poor fit for multilingual text (BPB 4.53); BPE
  is the obvious next step.
- One seed, no confidence intervals; the 0.0006 aggregate difference is noise.
- `probe_r32.py` originally ranked dead pairs in the top-byte list; the saved
  JSON has mean/max KL but empty `most_specialized` lists. The ranking is fixed
  in the committed code; the routing matrix and KL values are unaffected.
- Domain generation is not wired yet.

### Next

1. Rebuild the 16-domain corpus with 10–20M positions and run 2k–5k steps.
2. Add the no-domain-token control (flags were removed in the recovery
   simplification; re-add `--no-domain-token` and vocab 257).
3. Add a domain-bias router term in addition to the token prefix.
4. BPE tokenizer for multilingual/code.
5. Test hierarchical routing: domain → branch → pair, using the domain
   structure already visible in the routing matrix.

### Artifacts

- `results/r32_summary.json` — full summary
- `results/hybrid151_domain.probe.json` — per-domain BPB, routing matrix,
  specialization KL, geometry
- `results/plain21_domain.probe.json`
- `results/*.train.log`

---

## 2026-11-12 — r31 compute-matched pilot (previous)

One Gutenberg domain, 220 steps × batch 32 × context 256. Plain 2.4621 BPB vs
hybrid 2.4616 (tie), geometry-only 2.4881 (worse). See
[`../r31/EXPERIMENTS.md`](../r31/EXPERIMENTS.md).
