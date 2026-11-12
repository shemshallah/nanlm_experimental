# FA2 R24 — one unified model, one AdamW optimizer, one development candidate

**Date:** 2026-11-12. **Target branch:** `master` only.

## Change in strategy

R23's `context`, `value`, `semantic`, `joint`, and `stream` CPU arms are
**archived comparisons, not ongoing independent models**. The active Actions
workflow has **one training job**, no matrix, no experimental branch, and no
automatic promotion. Historical source remains available for reproducibility.

The accepted R18 PAL checkpoint at step 10,756 is **immutable**:
`dd040106568786e7fbad2a880f4010635b626b51c3697e2ab08712905a2de91f`.
R21/R22 experimental adapter weights are imported as *initialization*, not
declared accepted. All actual training updates use a single merged parameter
graph, one optimizer, and one candidate artifact
`reports/fa2_unified_candidate.pth`. That checkpoint is only a candidate;
R18 remains accepted until separate promotion requirements are met.

## Unified computation graph

1. **Inherited PAL lexical + subword recurrent backbone**: real previous-token
   and continuous recurrent state carried over adjacent word-aligned chunks.
2. **Inherited local attention + FFN**: bounded K/V cache preserved across
   contiguous chunks.
3. **New signed-mirror sparse MoE**: per-token top-2 routing among four SwiGLU
   experts (default width 384), combining original and reversed hidden
   coordinates. Initially zero-gated for exact inherited-logit identity.
4. **R18 learned lexical copy**: contextual 65–2048 causal token history
   (CPU default 512-token lookup budget), inherited Q/K, gain and moments.
5. **R21 relevance/novelty gate** and **R22 three-scale historical proposals**:
   trainable on the same next-token loss; their minimum-query cutoffs now
   respect absolute book positions rather than resetting to zero per chunk.
6. **R23 contextual long-memory reader**: trainable 4/8/16/32/64-token
   query/key encoders; retrieved eight-token values, 1/2/4/8-token
   multipositive contrastive losses; 65–32768 absolute-document-addressed
   slots. The reader's embedding lookup can now share gradients with the
   live inherited piece embedding table (without storing another copied table).
7. **Semantic adversarial auxiliary**: two-positive nonce/paraphrase training
   against seven distractors; development uses held-out wording. This is a
   controlled synthetic probe, **not** evidence of general reasoning.

The memory index admits only completed *earlier* document positions, with
values ending before the active chunk. Labels choose positives only in losses;
they never choose retrieval candidates. Both a **scrambled historical value** control and a **different-book donor**
control are evaluated. Donor candidates use the actual target-book causal query,
but sources and values come strictly from previously completed portions of a
different book. No future donor positions are admitted. The latter is a
necessary but not sufficient memory-specificity check: larger blind books and
adversarially matched donor frequencies are still required.

## Optimizer / continuation

One AdamW instance, nine groups:

| Group | State on initialization |
| --- | --- |
| 1–4 | Exact accepted PAL core/subword/bridge/local residual Adam moments |
| 5 | R18 Q/K and gain/recency moments migrated from accepted lineage |
| 6 | R21 gate moments transferred from lineage-verified candidate |
| 7 | R22 three-tier gate moments transferred from lineage-verified candidate |
| 8 | New R23 64-token memory reader |
| 9 | New signed-mirror sparse experts |

The optimizer asserts **every trainable parameter appears exactly once**.
A resumed unified candidate restores all nine groups, book cursors, detached
recurrent states, bounded attention K/V, and random-number generator state.
The original R18 checkpoint and R21/R22 source states are hash-checked before
and after training. **No separate independent weights and no model averaging.**

## What "end-to-end" means

Each truncated-BPTT update visits several consecutive, word-aligned chunks
using the **live** recurrent backbone. Cross-chunk recurrent-state gradients
flow through the specified `--unroll` length before detaching at the optimizer
step. The forward state carries across many more chunks and can reach a 32K
document address. **This is not 32K BPTT:** long-range retrieved values and
trainable keys receive their own gradient, but the recurrent model's gradient
is truncated to the selected contiguous unroll, for practical CPU costs.

The model is deliberately sparse rather than claiming that a four-expert
CPU prototype has frontier-level parameter count or language quality.
Scaling requires matched compute, larger context, more training tokens,
and independent blind data—not just adding architecture components.

## Verification and promotion discipline

The source CI must compile all source and pass inherited R23 causality and
stream-forward parity tests plus the R24 tied-embedding, absolute-position
gating, sparse routing and Adam-migration checks. The original R18 SHA-256
must match exactly.

Development selection uses byte-normalized NLL and a nonspace-loss penalty.
The report records **correct-vs-scrambled and correct-vs-unrelated-book advantage** in
nats/token, nonspace NLL, semantic accuracy, and actual maximum retrieved
distance for each 32K tier. Additional gates require improvement versus the
initial development reference, nonspace preservation, actual far-tier reads,
and nonregressing synthetic semantics. Passing these does **not** automatically
promote the checkpoint. Previously consulted historical test books are reported
only after selection and marked *nonblind*.

Promotion still needs newly gathered author/book-disjoint blind test books,
**correct vs unrelated document** history, paraphrase and multi-hop QA,
full pipeline regression, and statistically meaningful held-out improvements.

## CPU usage

Source-only (does not require the large original R19 archive):

```bash
cd fa2r19/source
python test_r23_context64.py
python test_r23_stream_carry.py
python test_r24_unified.py
```

Training requires a **verified exact original R19 full archive** with the
original tokenizer and source-verified NaNROUTER books. GitHub's accepted 36
checkpoint chunks by themselves do not contain those remaining inputs. In
GitHub Actions, select
[`FA2 R24 - single unified PAL and mirror-MoE CPU`](../../.github/workflows/fa2-r23-parallel.yml)
and supply the pinned full-release HTTPS `archive_url`. A blank address runs
source validation only. The workflow overlays the current `master` source,
verifies inherited R18/R19 tests, and runs only
`python source/r24_unified_cpu.py`.

Example **after** extracting the original archive and materializing R21/R22
adapter files:

```bash
cd work/release
python source/r24_unified_cpu.py --root . --steps 100 \
  --chunk-span 12 --unroll 2 --max-chunks 256 --threads 2
```

To resume, use `--resume reports/fa2_unified_candidate.pth` plus a larger
cumulative `--steps`, preserving the identical model and book-stream config.

## Scaling ladder (not claims of achieved performance)

- **CPU validation:** source gates, one model, small TBPTT, trained vs
  untrained/logit-identity and scrambled-history controls.
- **Long-document fidelity:** actually reach 8K, 32K and >32K source
  distances on new books; measure latency, candidate recall and byte loss by
  distance. Existing 32K is the maximum of the current index, not >32K.
- **Gradient horizon:** increase TBPTT unroll and expert width/number in
  gated, matched-compute rounds. Preserve the same trained trunk and Adam
  moments; any capacity expansion must prove an exact-output migration.
- **Semantics and transfer:** non-oracle paraphrase, multi-hop document QA
  and held-out lexical and relational formats with unrelated-book donors.
- **Frontier comparison:** substantial extra high-quality training data and
  compute, independent blind Gutenberg authors/books, and comparable byte
  tokenizer/BPB baselines. A source merge alone is not a frontier-model result.

**Status:** source architecture integrated; no R24 training or BPB gain is
claimed until an actual validated CPU run reports metrics.
