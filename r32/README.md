# r32 — multidomain domain-token Mirror LM

r32 moves the experiment from one Gutenberg distribution to the new
`corpora/domains/` releases: 16 public domain collections covering
encyclopedia, STEM, biomedicine, law, patents, code, math, news, cooking,
security, Q&A, multilingual text, finance, standards, humanities, and
chemistry.

The first r32 result is deliberately narrow: **does a domain-token-conditioned
151M hybrid tie a 21M dense model at matched active compute on a 16-domain
corpus, and does routing become domain-dependent?** The answer is yes to the
tie and strongly yes to domain-dependent routing. See
[`EXPERIMENTS.md`](EXPERIMENTS.md).

## Data

`code/prepare_domains.py` builds a balanced byte corpus from the public
`corpora-*` GitHub releases. Full assets are 0.05–1.5 GB each; the builder
fetches 4 evenly spaced 1 MB ranges per domain, discards partial documents at
the cut edges, truncates documents over 32 KB, deduplicates exact documents,
and splits at document boundaries 90/10.

Built corpus used by the pilot:

- 16 domains, 4 MB sampled per release
- 43,725,193 train byte positions / 4,721,470 valid byte positions
- 50,746 eligible train documents / 5,651 held-out documents
- `vocab_size = 257 + 16 = 273`

Data files (`train.bin`, `valid.bin`, `meta.json`, `split_manifest.json`) are
not committed. They are available as the private Kaggle dataset
`shem42/r32-multidomain-16domains`; rebuild locally with:

```sh
python r32/code/prepare_domains.py \
  --out-dir r32/data --cache-dir r32/cache \
  --per-domain-mb 4 --chunks 4 --valid-fraction 0.10 --seed 2026
```

## Domain-token scheme

Domain tokens are not stored in the bins. `DocumentWindowSampler` prepends
`257 + domain_id` to every training/eval window. The first target is the first
byte after the prefix, so the prefix is ordinary context and needs no loss
mask. Domain runs use:

```sh
--context 257 --train-context 256 --eval-context 256
```

`DocumentWindowSampler.domain_windows(domain_id, ...)` samples only that
domain's held-out documents for per-domain evaluation.

## Model arms

| Arm | Params | Active MAC/token (est.) | Config |
|---|---:|---:|---|
| Plain dense + domain token | 21,397,248 | ~21.4M | dim 576, 4 layers, 24 heads, SwiGLU |
| Hybrid + domain token | 151,014,658 | ~21.1M | dim 544, 4 layers, 16 heads, 128 pairs, top-2, expert 235, additive MoE, linear+geo router |

The hybrid keeps a dense SwiGLU trunk and adds the sparse mirror MoE branch.
The 16 domain-token embeddings add ~9K parameters to the plain model.

## Evaluation

Use `code/probe_r32.py`, not the legacy `evaluate` command: the legacy path is
not domain-token aware yet.

```sh
python r32/code/probe_r32.py \
  --data-dir r32/data --run-dir run/hybrid151_domain \
  --context 256 --batch 8 --batches 4 \
  --out r32/results/hybrid151_domain.probe.json
```

The probe returns per-domain BPB/top-1, 128-bin routing counts, an
`routing_matrix` of shape `[16, 128]`, per-pair KL specialization, domain-token
norms, and geometry diagnostics.

## Hierarchical routing (r32.1, code ready)

Flat 128-way routing is available, but the next run uses a **domain → branch →
pair** hierarchy:

- 8 branches × 16 pairs = 128 leaves
- one branch selected per token, then top-2 pairs inside that branch
- branch balance auxiliary term added to the pair balance term
- domain token conditions the branch and pair routers through the hidden state
- `--hierarchical --branches 8 --pairs-per-branch 16`

Implemented in `code/poincare_frontier.py`; `probe_r32.py` also reports
`branch_routing_matrix[16][8]` and per-domain branch effective counts. The
30-minute multidomain run is packaged as a Colab notebook:
[`colab/colab_multidomain_30min.ipynb`](colab/colab_multidomain_30min.ipynb),
with instructions and Colab quota notes in [`colab/README.md`](colab/README.md).

## Latest result (2026-11-12)

220 steps × batch 32 × context 256 = 1.8M positions per arm.

| Arm | Aggregate BPB | Top-1 |
|---|---:|---:|
| Plain + domain token | **2.917990** | 44.12% |
| Hybrid + domain token | **2.918564** | 43.83% |

Hybrid − plain = **+0.000574 BPB**: a tie within noise. This reproduces the r31
compute-matched tie on 16 domains instead of Gutenberg.

Routing is not uniform:

- **Code** uses 124/128 pairs, effective **38.5**; pair 102 takes **18.3%** of
  code traffic, pair 13 takes 8.9%.
- **Multilingual** uses 126/128, effective **39.6**; pairs 10/4/9/54/57 each
  take 8.7–10.4%.
- **Standards** uses 127/128, effective 83.8; pairs 105/7/72 lead.
- Most other domains spread across ~100 effective pairs.
- Mean pairwise JS divergence between domain routing distributions is 0.109;
  code↔multilingual is 0.390, while law↔finance is 0.016.

Per-domain BPB differences are small and two-sided: hybrid is better on 9/16
domains (pubmed, patents, math, finance, multilingual, chemistry, etc.) and
worse on 7/16 (code +0.034, qa +0.023, law +0.020). Full table in
`EXPERIMENTS.md` and `results/`.

## Caveats

- 1.8M positions / 220 steps is a pilot, not the 10M-position target.
- Aggregate tie is not a win; code and qa are clearly worse for the hybrid.
- No no-domain control in this run, so the BPB value of the domain token itself
  is not isolated. The routing matrix does show domain-dependent expert use.
- Byte-level BPB on multilingual text is poor (4.5) because one byte per token
  is a bad fit for non-Latin scripts; a BPE tokenizer is the obvious next step.
- Generation is not wired for domain tokens yet.
