# PAL R18 — learned long-distance memory, one unified checkpoint

**Status: PROMOTED after historical development gates; not frontier-capable yet.**

This is a real CPU continuation of the same PAL R15/R16 weight lineage. It preserves
its 96-dimensional recurrent NOW state, rank-64 mirror P/M tensors, pretrained
subword GRU/decoder, 64-position sparse attention, FFN, original R10/R11/R12
training history and **all four original AdamW optimizer groups**. It adds one
small learned long-range memory and copies its trained state into the **same
single checkpoint file** `checkpoint/r12_active.pal12`. There are no additional
competing model files, independent decoder heads or reset inherited weights.

- R15/R16 parent SHA-256: `dc61500ec3abd424409f136ffff517242134af77435e66f135f4363cb7ec450f`
- PAL R18 accepted SHA-256: `dd040106568786e7fbad2a880f4010635b626b51c3697e2ab08712905a2de91f`
- Inherited R12 optimizer steps: **10,756, unchanged**; R10=2,020; R11=130.
- New R18 learned parameters: **25,347** (including learned gate, Q/K, temperature, distance prior).
- R18 stages: 240 Q/K hard-negative contrastive CPU updates and 120 frozen-base copy-gate CPU updates.
- Newly used large GitHub Release corpora: **none**. This execution environment cannot fetch the release assets. All train, dev and historical test books below come from the previously verified NaNROUTER R16 archive.

## What R18 actually adds

A causal remote key/value store. At token `t`, the query is the two **previous**
subword tokens. Remote memory slot `j` is keyed by tokens `j−2,j−1` and holds
only the already-observed value `token[j]`, where `j ≤ t−65`. No token at `t`
or later is inspected when constructing the prediction for `t`.

The encoder learns distinct query/key projections from the inherited subword
embedding basis, with a trainable retrieval temperature and relative distance
bias. It searches a sparse candidate pool of earlier tokens sharing a lexical
cue, with hard negatives, then adds gated copy evidence to the **original PAL
decoder logits**. The original attention and FFN still operate, so the memory
is an additional capability, not an attention replacement.

- Trained retrieval memory: 2,048 subword positions, minimum distance 65, top-8.
- Candidate lookups: bounded to 64 per query with a bounded per-token index,
  rather than dense attention over every stored token.
- Architectural indexing smoke: a 40,000-position stream successfully located
  information **39,000 tokens** back with a 65,536-token memory budget. This is
  an **indexing test only**, not long-context language-model training or an
  observed 40,000-token language-understanding result.
- Exact zero-gate preservation at migration: inherited prediction logits
  unchanged bit-for-bit. Source-mutation and future-suffix causality tests pass.

## Verified development promotion

There are 12 historical training books, 12 different historical development
books, and 14 historical test books. Queries in the contrastive stage are
selected from genuinely earlier occurrences within each book. The task is
**lexical prefix matching**, so exact literal matching is a strong oracle;
this task alone does not demonstrate semantic memory or reasoning.

| Development metric | Inherited R15 | R18 | Difference |
|---|---:|---:|---:|
| Short-window BPB (96 windows, 7,432 tokens) | 2.662620 | **2.661726** | −0.000894 |
| Short-window non-space NLL | 4.829912 | **4.828369** | −0.001543 |
| Long-window BPB (8 windows, 10,183 tokens) | 2.592224 | **2.565997** | −0.026227 |
| Long-window non-space NLL | 4.615227 | **4.591845** | −0.023382 |
| Long-window non-space top-1 | 11.867% | **12.418%** | +0.551 pp |
| Hard-negative dev retrieval top-1 (800 queries) | 12.00% untrained | **93.13%** | +81.13 pp |

All four mandated short/long language-loss gates passed, as did retrieval
accuracy/NLL and remote eligible-query loss. This is a **provisional scientific
promotion** based on historical development splits, *not an independently
sealed test* or general frontier-performance claim. R18's 120 gate updates fit
only newly added parameters; original four optimizer histories are untouched.

## Post-selection historical test, not used for checkpoint selection

| Metric | Original R15 | R18 | Control / comment |
|---|---:|---:|---|
| Test hard-negative retrieval (1,200 queries) | 12.25% untrained | **95.08%** | Exact matching would be trivial on this controlled task |
| 8-window test BPB | 2.429602 | **2.413332** | Exact-key heuristic: 2.410076 (better than learned) |
| 8-window non-space NLL | 4.531988 | **4.527414** | Same tokenizer and same book selection |
| Wider 32-window BPB | 2.457664 | **2.432189** | 39,470 subword positions, 13 books |
| Wider 32-window non-space NLL | 4.496108 | **4.475629** | 30/32 windows improved |

In the wider historical test all 13 represented books improved. The
*descriptive* bootstrap 95% interval for R18-minus-inherited BPB over these
13 books is [−0.03436, −0.01778]. It is not a confidence interval on an
independently selected benchmark and does not correct for historical exposure.

A 512-memory ablation scored 2.416343 BPB versus 2.413332 at 2,048, showing
some benefit beyond 512 positions in this particular sample. Shuffling the
stored *past* values while preserving addressing worsened test BPB to 2.434391;
this is causal evidence that the copied content matters. The exact-key
heuristic outperforming Q/K is an important **negative control**: learned
addressing has not yet beaten a simple non-neural lookup on the same task.

## Structural and continuation results

- Original inherited tensors, bridge, R12 adapter, four optimizer-group states,
  RNG histories, and R10/R11/R12 counters were compared **exactly** with the
  R16 parent. The eleven deterministic tree digests are preserved under
  `reports/r18_parent_equivalence.json` and can be rechecked without
  redistributing a second checkpoint.
- R12 inherited invariants: **11/11 passed**. R13 CPU cached/original parity:
  absolute loss difference 1.91e−6 on a 128-token synthetic test.
- R18 extension tests: **16/16 passed**, including future-suffix invariance,
  remote-source sensitivity, zero-gate logit identity, real Q/K gradients,
  a 39,000-distance sparse index smoke, and fifth-group AdamW moment migration.
- A subsequent 10-update **joint** continuation restored all 15 new parameter
  Adam moments into a fifth group alongside the four inherited groups and
  trained the decoder/memory together. It was **rejected**: long dev BPB and
  non-space NLL regressed. The accepted R18 checkpoint remained **unchanged**.
  This is a tested continuation path, not evidence the joint optimization was
  successful.

## Reproduce from the accepted current checkpoint

PyTorch CPU 2.x, NumPy, and SentencePiece are required. Run from the release
root (trusted checkpoints only; PyTorch pickle loading is not safe on files
from unknown sources):

```sh
python -m pip install -r requirements.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python source/r18_verify_lineage.py --root .
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python source/test_r18.py .
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python source/test_r12.py \
  --root . --books nanrouter --checkpoint checkpoint/r12_active.pal12
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python source/test_r13_cpu.py \
  --root . --checkpoint checkpoint/r12_active.pal12 --tokens 128 --threads 2
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python source/r18_postselection_test.py \
  --root . --short-windows 96 --long-windows 8
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python source/r18_wide_test.py
```

`r18_run.py` documents the one-time R16→R18 migration, expecting the **R16
parent checkpoint**. Do **not** rerun it against the already promoted R18
checkpoint. To explore further continued joint training **without destroying
the accepted state unless the dev gate passes**:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python source/r18_joint_continue.py \
  --root . --steps 20 --span 96 --long-span 400 --promote
```

`r18_joint_continue.py` restores all optimizer moments, maintains the shared
embedding basis after every joint update, and preserves the active checkpoint
byte-for-byte when its gates reject the candidate. It is not a multi-GPU
training system and does not yet implement long-gradient backpropagation through
thousands of memory positions in the inherited recurrent trunk.

## What is needed for a credible frontier trajectory

1. Fetch GitHub Releases for `shemshallah/NaNROUTER` (short stories, Gutenberg
   paragraphs and domain data), **deduplicate at book and passage level**, and
   freeze author-disjoint dev/new-test cohorts before doing further selection.
2. Beat the trivial **exact-copy** control and equal-parameter/compute GRU,
   transformer and state-space baselines, not just untrained Q/K projections.
   Use copy/contradiction/variable-binding and multi-hop prompts separated by
   1K, 8K, 32K and beyond, with explicit distractors and no injection of answers.
3. Train long-context retrieval jointly with linguistic prediction using
   efficient chunked state recurrences, bounded memory caches, and a learned
   relevance/novelty gate. Memory indexing out to 40K is not the same as having
   an actually trained 40K-context model.
4. Expand capacity in **the same unified model**, measuring an honest scaling
   curve over widths, active memory sizes and token budgets. Add hierarchical
   landmarks/sparse heads only when tested against matched controls. Use GPUs or
   a distributed trainer for large-scale experiments; this CPU run cannot make
   the current small PAL network frontier competitive by itself.
5. Gate promotions on independently held-out content BPB, total BPB,
   non-space metrics, long-distance retrieval, fluency, latency, and inherited
   future/word prediction guards. Report failures, not just winning runs.

**Bottom line:** R18 provides a functioning, tested learned lexical memory and a
small but consistent historical language-loss improvement. It is **a step
toward**, not completion of, a frontier-capable language model.

See `reports/` for all numeric evidence, `docs/R16_README_ORIGINAL.md` for
the preserved R16 research record, and `manifest.json` for release checksums.

## R19 — adversarial semantic retrieval continuation (2026-11-12)

**R19 experiments did not pass promotion.** The single accepted PAL R18
checkpoint remains byte-for-byte identical, including all original R10/R11/R12
weights and optimizer histories. R19 added source, tests, and evaluation logs
only: token-level sentence alignment (from the existing PAL/R18 representation),
changed-prefix admission tests at 2K–65K, unseen paraphrase, unseen key,
held-out relation family, and hard two-hop retrieval probes. The corrected
hard two-hop test scored only 5.0% for the better sentence-level candidate.

Read `docs/R19_SEMANTIC_FRONTIER.md` for accurate baselines, limitations,
reproduction commands, and the new acceptance criteria. Run
`python source/test_r19.py --root .`. No R19 candidate weights or optimizer
states were promoted to, or distributed alongside, the accepted checkpoint.
