# FA2 — Unified reciprocal tokenizer ↔ geometric meta feedback

This change extends **the one canonical `MirrorLM` in `poincare_frontier.py` on
`master`**. It creates no second predictor, side model, git branch, new
checkpoint lineage, or automatic checkpoint promotion. FA1 and NaNROUTER
remain separate projects and are not modified.

## Architecture

The inherited causal byte stream and two-pass four-scale token hierarchy
remain intact. With `--reciprocal-scales`, one additional path projects the
same byte-aligned hidden states into five simultaneous causal resolutions:

1. Byte — current byte-state feature.
2. Byte pair — current and previous byte-state features.
3. Four-byte rolling fragment — current and up to three preceding features.
4. Word prefix — running mean since previous space, tab, CR or LF.
5. Sentence prefix — running mean since previous `.`, `!`, `?`, LF or EOS.

These are *multi-resolution representations* of the byte tokenizer, **not
independently fitted BPE/word/sentence vocabularies**. A true BPE/SentencePiece
tokenizer would need a training-only merge table, byte-aligned boundary maps,
vocabulary-migration study, and separate bits/token metrics. The new route
rejects non-257-symbol and non-UTF-8-byte+EOS checkpoints to prevent interpreting
subword IDs as raw ASCII punctuation.

At each byte position **within every geometric MoE block**, the same
`MirrorLM` now performs two reciprocal, same-position passes:

```
256 point microtransformers <--> 5-scale causal tokenizer
    |       ^                             |
    v       |                             v
32,640 signed pair concept points ---+----> meta tokenizer
    |       ^                             ^
    v       |                             |
2,763,520 virtual triangle-cloud addresses -----+
    |       ^                             |
    v       |                             v
256 apex/meta MoE experts -----------------------+
    |       |                             |
    +-------+------ feedback to ALL tokenizer scales + next backbone block
```

More precisely:

1. Before **each backbone block**, token scales are recomputed from the
   current causal hidden stream and updated using any returned meta state.
2. In every geometric block, the original 256 address-specific
   microtransformers process eight selected point concepts. The token scales
   then participate in the **same** point→28 signed edge→56 triangle climb.
3. The signed pair store has exactly 32,640 learned addresses. Its d1 incidence
   map returns pair concepts to the active microtransformers. The sparse
   triangle cloud has 2,763,520 possible exact addresses, backed by hashed
   trainable vectors and exact Fourier phases. Its d2→d1 maps return
   abstract-concept feedback to those same microtransformers.
4. One **meta tokenizer** forms eight content tokens at each byte: five token
   resolutions plus pooled point, pair, and triangle features. Two rounds of
   shared cross-scale attention assemble the abstract concept and query the
   existing 256 apex specialists, including their address-specific MoE FFNs.
5. Apex feedback updates point microtransformers, pairs, triangle tokens,
   and all five token scales. The next reciprocal pass uses these updated
   states. The original tetrahedron/curvature/holonomy and apex paths then
   continue with the returned geometric features.
6. The latest apex state is propagated to the tokenizer at the next backbone
   layer. Every backbone layer receives a gated token/meta residual, including
   attention-only layers; this is **not** a post-decoder sidecar.

All point, pair, triangle, and meta parameters come from the **same already
registered geometric MoE blocks**. The meta tokenizer reuses the shared
scale-attention/projection tensors and a single zero-initialized gate; it has
no independent vocabulary head or optimizer. The original geometric memory
over completed chunks remains in place. The new path does not allocate a
dense triangle cloud: at most 8/28/56 active addresses per byte are used.

The geometric MoE still performs its original point→pair→triangle→tetra→apex
and downward feedback inside transformer blocks. R37's
isolated +2/+3 forecaster continues to detach NOW features from future losses;
new reciprocal weights train from the existing NOW loss, preserving forecast
isolation. The historical R36 holonomy and R35 curvature routes remain intact.

## Causality and migration

- Each scale includes only current/past byte-state features. Its cross-scale
  attention and sparse geometry loops are **at the same time index**.
- The optional geometric chunk bank masks all writes from the current or
  later chunks, retaining strict temporal causality.
- The reciprocal scalar gate initializes to exactly zero. All inherited NOW,
  +2, and +3 outputs are exactly identical at migration; the checkpoint still
  has one weight dictionary and one AdamW optimizer state.
- `grow` and `train` preserve old parameters and optimizer moments by name,
  enforce checkpoint tokenizer identity, and record the change in the common
  `growth_history`. Existing learned reciprocal weights cannot be removed.
- `evaluate --ablations` exposes `no_reciprocal_scale_geometry`, separately
  from the inherited token hierarchy and geometric/holonomy ablations.

To migrate safely, copy the **latest complete FA2** checkpoint (including
optimizer states) to a candidate working directory first. Verify its byte
tokenizer identity and existing `--hierarchy-rank`, `--symbiotic-rank`,
`--micro-meta`, `--simplex-transfer` capacities. If any are missing, first
enable them through a function-preserving `grow` on that same candidate.

```bash
python poincare_frontier.py grow \
  --run-dir candidate --reciprocal-scales \
  --max-logit-drift 1e-5 --report candidate/reciprocal-migration.json

python poincare_frontier.py train \
  --run-dir candidate --data-dir verified-byte-book-corpus \
  --steps 250 --batch 1 --train-context 128 \
  --cpu --cpu-threads 2 --eval-every 25 --save-every 25
```

No canonical checkpoint has been modified, continued, or promoted by this
source change. The user-reported R38 640-update synthetic checkpoint and its
exact content-addressed triangle implementation were **not present in the
GitHub `master` source** at development start, so this patch must not be
represented as an exact import or verified continuation of that private
experimental state. It is a migration-compatible bridge for the visible
R37/R36 architecture, ready for reconciliation when that R38 source and
checkpoint are available.

## Validation gates

The nine `tests/test_reciprocal_scales.py` tests cover unchanged three-horizon
predictions, all inherited Adam moments and step/RNG metadata, exact parameter
counts, active positive signed-pair/triangle/meta gradients, causal suffix
invariance, independent pair/cloud/meta return ablations, per-block traversal,
CLI growth, and 512D mixed-head autograd.

Run `python -m unittest discover -s tests -q` under `poincare_mirror/`.
The existing GitHub Actions source-test job already uses that command on
pushes to `poincare_mirror/**`.

For a deliberately small matched synthetic *mechanism check*, run:

```bash
python scripts/compare_reciprocal_scales.py --updates 64
```

That script uses identical base weights, optimizer settings, train windows,
and independent held-out key-value mappings for control and candidate. It
reports +1/+2/+3 CE, bits/byte, top-1, top-5 on identical target bytes,
plus endpoint feedback-disabled ablation. Results on this task are **not**
reliable evidence of real-book gains and cannot promote a model.

### Corrected in-block topology — measured 128-update synthetic pilot

Local reproducibility record: `reports/reciprocal_synthetic_128.json`.
Sixteen independently generated held-out documents, 128 query-byte targets.

| Held-out target | Control CE | Reciprocal CE | Control top-1 | Reciprocal top-1 | Control top-5 | Reciprocal top-5 |
|---|---:|---:|---:|---:|---:|---:|
| +1 | **2.353047** | 2.354270 | **14.06%** | 9.38% | **62.50%** | 60.94% |
| +2 | **5.396919** | 5.465668 | 0% | 0% | 0% | 0% |
| +3 | 6.197485 | **6.052396** | 0% | 0% | 0% | 0% |

Direct feedback ablation at the candidate endpoint *worsened* +1 CE from
2.354270 to 2.355620, +2 CE from 5.465668 to 5.518743, and +3 CE from
6.052396 to 6.082287. Therefore the recursive feedback is genuinely useful
**inside the candidate** on this tiny synthetic workload, but the control
still beats it on +1/+2 CE and +1 top-1/top-5. **Promotion is rejected.**

The prior endpoint-only bridge and its original pilot are preserved as a
historical superseded control in `reports/reciprocal_synthetic_128_preclosure.json`.
The corrected architecture above replaces that incomplete bridge.

Before promotion, restore the actual full optimizer-carrying FA2 candidate;
benchmark against its own unmodified ancestor on author/book-disjoint books,
both 256-byte and long contexts, with document-level bootstrap intervals.
Require positive ablation effect, no next-byte regression, and durable
improvements in +1/+2/+3 CE/BPB and top-1/top-5. Separately measure
throughput, peak resident memory, semantic retrieval, and generation quality.
