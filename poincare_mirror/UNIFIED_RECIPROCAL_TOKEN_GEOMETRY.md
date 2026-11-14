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

At each byte position the bridge performs two reciprocal, same-position loops:

```
five token scales -- cross-scale attention --> 256 point keys
             --> 8 sparse active points (existing point microtransformers)
             --> 28 signed pair tokens (existing 32,640 table)
             --> 56 oriented triangle tokens (existing 2,763,520 exact IDs)
             --> top-8 of 256 existing apex/meta specialist transforms
             --> triangle attention --> signed pair attention --> point attention
             --> tokenizer-scale feedback --> next loop --> byte decoder
```

All point, pair, triangle, and meta parameters come from the **same already
registered geometric MoE block**. The bridge registers only its own small
scale-attention/projection tensors and a single scalar residual gate. The
geometric memory over *completed* chunks, when enabled in that block, is also
shared with the bridge. There is no dense C(256,3) embedding allocation:
at most 8/28/56 active addresses per byte are instantiated, selected by the
same content-addressed point keys.

The geometric MoE still performs its original point→pair→triangle→tetra→apex
and downward feedback inside transformer blocks. The new route makes the
tokenizer feedback and apex concept features converse directly after the final
transformer block, without moving or resetting that older pathway. R37's
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

The new `tests/test_reciprocal_scales.py` covers unchanged three-horizon
predictions, all inherited Adam moments and step/RNG metadata, exact parameter
counts, active positive signed-pair/triangle/meta gradients, causal suffix
invariance, trained-bridge ablation, CLI growth, and 512D mixed-head autograd.

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

### Measured 128-update synthetic pilot

Local reproducibility record: `reports/reciprocal_synthetic_128.json`.
Sixteen independently generated held-out documents, 128 query-byte targets.

| Held-out target | Control CE | Reciprocal CE | Control top-1 | Reciprocal top-1 | Control top-5 | Reciprocal top-5 |
|---|---:|---:|---:|---:|---:|---:|
| +1 | 2.353047 | **2.350482** | **14.06%** | 10.94% | **62.50%** | 60.16% |
| +2 | 5.396919 | **5.344437** | 0% | 0% | 0% | 0% |
| +3 | 6.197485 | **6.197475** | 0% | 0% | 0% | 0% |

Direct feedback ablation at the candidate endpoint changed +1 CE from
2.350482 to 2.360541 (positive pathway effect), but *improved* +2 CE from
5.344437 to 5.337389 (negative pathway effect). The inherited +1 top-1 and
top-5 also decline. **The candidate fails promotion**, despite slightly
better aggregate +1/+2 CE. This pilot should motivate supervision/recall
improvements, not architectural gain claims.

Before promotion, restore the actual full optimizer-carrying FA2 candidate;
benchmark against its own unmodified ancestor on author/book-disjoint books,
both 256-byte and long contexts, with document-level bootstrap intervals.
Require positive ablation effect, no next-byte regression, and durable
improvements in +1/+2/+3 CE/BPB and top-1/top-5. Separately measure
throughput, peak resident memory, semantic retrieval, and generation quality.
