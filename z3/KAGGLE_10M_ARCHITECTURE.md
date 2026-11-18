# Z4 115M architecture profile and 10-minute Kaggle profile

This is a static parameter/compute audit of the fixed `115m` preset plus the
short-run changes used by the `kaggle10m` preset.  The T4 measurements are in
[`KAGGLE_10M_RESULTS.md`](KAGGLE_10M_RESULTS.md).  The current best short-run
profile is **`kaggle10m_nomtp`**: 115,238,237 parameters, 900 steps, **1.5981
WikiText-2 BPB** at a 600-second cap.  It keeps the concept loop and feedback
bus but skips the MTP branch/head compute; the MTP parameters remain in the
checkpoint but are not executed.  The 3e-4/MTP-on profile is 1.7489; the
original 2e-4 profile is 1.7914.  Removing the concept loop is worse (1.7822),
so the concept path pays for itself at this horizon.

The profile was produced by:

```sh
python z3/profile_115m.py --preset 115m --ctx 512
```

## Parameter allocation

Total: **115,238,237** parameters.  The table groups the main modules;
scalar gates and norms make up the rounding remainder.

| Group | Parameters | Share |
|---|---:|---:|
| Trunk blocks | 68,209,730 | 59.2% |
| — 7 dense blocks | 13,196,309 | 11.5% |
| — 3 MoE blocks | 55,013,421 | 47.7% |
| MTP branch blocks (2 lags) | 19,055,760 | 16.5% |
| Token embedding (`tok`) | 8,199,680 | 7.1% |
| MTP output basis (`mtp_output`) | 8,199,680 | 7.1% |
| Full-byte sequence probe | 3,932,160 | 3.4% |
| Concept loop | 3,490,820 | 3.0% |
| MTP position embedding | 2,621,440 | 2.3% |
| Memory cross-attention | 491,841 | 0.4% |
| MTP adapters/gates/heads/cond | 904,324 | 0.8% |
| Byte probe, composition, task | 92,160 | 0.1% |

The parameter budget is MoE-heavy, but the active compute is not.  The 3 MoE
layers hold 47.7% of the parameters while activating only 2 routed + 2 shared
experts per token.

## Estimated compute per token

Estimated FLOPs use the 2x-MAC convention and include the full 512-token
sequence per position.  They are for ranking, not hardware claims.

| Component | FLOPs/token | Share |
|---|---:|---:|
| Main LM head | 16.48M | 10.2% |
| MTP heads (t+2, t+3) | 32.96M | 20.5% |
| Deep-supervision aux heads | 32.96M | 20.5% |
| MoE FFN (3 layers) | 14.76M | 9.2% |
| Attention (10 layers) | 13.52M | 8.4% |
| Concept encoder (10 blocks) | 12.29M | 7.6% |
| MTP branch blocks | 10.17M | 6.3% |
| Dense FFN (7 layers) | 8.60M | 5.3% |
| Feedback projections (10 blocks) | 8.19M | 5.1% |
| Full-byte sequence probe | 7.86M | 4.9% |
| MTP context/route | 0.82M | 0.5% |
| Memory cross-attention | 0.41M | 0.3% |
| Concept object retrieval | 0.33M | 0.2% |
| Byte probe / composition | 0.18M | 0.1% |
| **Total estimate** | **160.96M** | **100%** |

## Where the architecture is wasting compute

### 1. Full-vocabulary auxiliary heads are 41% of the estimate

The main head is only 10.2%.  The two MTP heads and the two deep-supervision
heads each project to all 64,060 tokens, so they cost **65.9M FLOPs/token**
together.  At the 300-step CPU stability run the MTP loss was **2.30** versus
main CE **0.89**; the MTP branch was not paying for its output projection at
that horizon.  Deep supervision also produces two extra full-vocab projections.

A short-run profile should not pay for heads whose loss weight is zero.  The
model now accepts `compute_mtp_aux`, `compute_byte_probe`, and
`compute_byte_sequence` switches.  The trainer sets them from the loss weights,
so `kaggle10m` drops the two aux heads (20.5%) and the full-byte sequence probe
(4.9%) while keeping the parameter count unchanged at 115,238,237.

### 2. MTP loss weight is too high for a 10-minute run

The preset uses `mtp_weight=0.3`.  With MTP CE at ~2.3 and main CE at ~0.9, the
weighted MTP term is comparable to the main objective.  `kaggle10m` lowers it
to **0.1** and keeps the main CE dominant.

### 3. The concept encoder, not the object memory, is the concept cost

The object memory retrieval is only 0.2% of the estimate.  The shared
`ConceptEncoder` is 7.6% and runs at every block.  The feedback projections add
5.1%.  The object gate is still ~0.018 after 300 CPU steps and the object depth
carry gate is effectively dead.  `kaggle10m` reduces the object top-k
(4 vertices / 2 pairs / 0 triples) and sets the triple-pointer loss to zero;
this trims retrieval and pointer losses without removing the token-aligned
concept state.

### 4. MTP position embedding holds 8k rows for a 512-context run

`mtp_position_embed` is 2.62M parameters for 8,192 positions.  A 512-context
10-minute run uses 512 rows.  This is a parameter/memory cost, not a large
compute cost; it stays for now to keep the exact 115M count.

### 5. The route/feedback path is near-closed but not exactly zero

The route-to-attention projection is initialized at 1% scale, not gated by
`feedback_progress`.  The five scalar feedback gates start at zero.  This is a
minor initialization inconsistency; the 300-step run opened the memory gate to
~0.007 and the route-hidden gate to ~-0.009, so the bus does learn.  It is not
the first change for a 10-minute run.

## `kaggle10m` preset

The preset inherits every 115M geometry value and keeps the parameter count at
**115,238,237**.  It changes only the short-run schedule, loss weights, and
concept top-k:

| Setting | `115m` | `kaggle10m` |
|---|---:|---:|
| Warmup steps | 300 | **30** |
| LR horizon steps | 10,000 | **1,000** |
| Batch size | 8 | **4** |
| Wall cap | none | **600s** |
| Eval cadence | 500 | **100** |
| EMA | on | **off** |
| Dropout | 0.1 | **0.05** |
| MTP weight | 0.3 | **0.1** |
| MTP aux weight | 0.2 | **0** |
| Full-byte sequence weight | 0.02 | **0** |
| Object pointer weights | 0.02 / 0.01 | **0.005 / 0** |
| Concept top-k | 8 / 4 / 2 | **4 / 2 / 0** |
| Feedback warmup | 500 | **30** |
| Context | 512 | 512 |

Explicit CLI flags still override the preset.

## Kaggle 10-minute recipe

### Fast gate: WikiText-2

The first 10-minute gate uses WikiText-2 raw train+valid (~11 MB), not a
64 MB FineWeb slice.  It downloads in seconds and builds bins in about a
minute, so the 600-second **training** cap is not dominated by data prep.  The
notebook [`kaggle_10m_115m.ipynb`](kaggle_10m_115m.ipynb) runs this path.

```sh
mkdir -p data
for split in train valid; do
  curl -L --fail -o "data/wikitext_${split}.txt" \
    "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/${split}.txt"
done
cat data/wikitext_train.txt data/wikitext_valid.txt > data/wikitext_corpus.txt
python z3/build_domain_bins.py \
  --corpus wikitext=data/wikitext_corpus.txt \
  --out data/wikitext_bins \
  --max-bytes-per-domain 12000000 \
  --train-fraction 0.98 --valid-fraction 0.01
python z3/train_v3.py --preset kaggle10m \
  --corpus data/wikitext_bins/train.bin \
  --val-corpus data/wikitext_bins/valid.bin \
  --data-manifest data/wikitext_bins/manifest.json \
  --out /kaggle/working/runs/z3_115m_kaggle10m
```

The result is comparable to the existing Z4 WikiText-2 CPU rows (31M,
1,000 steps, 2.053 BPB); the 115M T4 run has a different model and step budget,
so compare at equal wall time.

### Full-data gate: FineWeb-Edu

Use a 64 MiB FineWeb-Edu slice and the bundled layered tokenizer:

```sh
# 1. Download a range-limited public FineWeb-Edu slice.
curl -L --fail --retry 3 --range 0-67108863 \
  -o fineweb_edu.txt \
  'https://www.kaggle.com/api/v1/datasets/download/nameonlu/fineweb-edu/fineweb_edu_0000.txt'

# 2. Build document-framed bins and a manifest.
python z3/build_domain_bins.py \
  --corpus fineweb=fineweb_edu.txt \
  --out data/fineweb_bins \
  --max-bytes-per-domain 64000000 \
  --train-fraction 0.98 --valid-fraction 0.01

# 3. Ten-minute training (the preset caps wall time at 600s).
python z3/train_v3.py --preset kaggle10m \
  --corpus data/fineweb_bins/train.bin \
  --val-corpus data/fineweb_bins/valid.bin \
  --data-manifest data/fineweb_bins/manifest.json \
  --out /kaggle/working/runs/z3_115m_kaggle10m
```

If the T4 OOMs at batch 4, pass `--batch-size 2`.  If it is comfortably fast,
try `--batch-size 8`; the preset's `--grad-checkpoint` and AMP stay on.

The public Kaggle range URL used by the GitHub FineWeb workflow can return
**404 inside a Kaggle kernel**.  The HF parquet fallback is ~2 GB and converts
to ~3.5 GB of text; do **not** feed that whole file to
`build_domain_bins.py` in a 10-minute session.  Convert or slice it first, then
build from at most 8–64 MiB of text.  The WikiText-2 gate above avoids this
entirely.

For an equal-wall baseline, run the same data with the stock preset and the
same warmup:

```sh
python z3/train_v3.py --preset 115m \
  --corpus data/fineweb_bins/train.bin \
  --val-corpus data/fineweb_bins/valid.bin \
  --data-manifest data/fineweb_bins/manifest.json \
  --max-seconds 600 --batch-size 4 --warmup-steps 30 \
  --no-ema --out /kaggle/working/runs/z3_115m_baseline10m
```

This baseline keeps the aux heads and MTP weight 0.3.  Compare validation BPB
and tokens processed at the same 600-second wall cap, not step counts alone.

## What to read after the run

- `result.json`: `params`, `steps_completed`, `tokens_seen`, `nonfinite_skips`,
  `best`/`final` validation BPB, and `feedback` gates.
- `history.json`: per-eval BPB curve and whether the feedback bus opened.
- `config.json`: confirms `preset="kaggle10m"` and the resolved overrides.
- `test_domains.json` after `eval_domains.py`: held-out FineWeb BPB.

A 10-minute T4 run is an early-training comparison.  It cannot establish a
frontier BPB or long-context quality.

The 64 MiB FineWeb token build is a one-time cost.  For repeat 10-minute
experiments, upload `data/fineweb_bins` as a private Kaggle dataset and attach
it to the kernel; the 600-second cap applies to `train_v3.py`, not to data
preparation.

## Next ablations, in profile order

1. **One MTP head instead of two.**  The second MTP head plus its branch block
   is ~13% of estimated compute.  This changes the parameter count unless the
   unused branch is kept as dead weight; run it as an explicit ablation.
2. **MTP output rank 64 instead of 128.**  The MTP output basis is 8.2M
   parameters and the two MTP heads are 20.5% of compute.  A rank-64 branch
   halves that head cost; it needs a separate init path from `tok_up`.
3. **Concept loop every other block.**  The shared concept encoder is 7.6% of
   compute and runs at all 10 blocks.  Running it on alternating blocks and
   carrying the state would test whether the per-block reset is necessary.
4. **One shared expert instead of two.**  MoE active compute drops from
   4 experts to 3 per layer; parameters drop by ~3.7M.
5. **Gate the route-to-attention projection with `feedback_progress`.**  This
   is a no-parameter change that makes the step-zero model exactly the
   pre-feedback trunk instead of a 1%-scale route injection.
6. **MTP position embedding at 1,024 positions instead of 8,192.**  Saves
   ~2.3M parameters for a 512-context run; no compute change.

Do not stack all of these in the first Kaggle 10-minute test.  The first run
isolates the aux-compute skip plus the short schedule; later runs should change
one item at a time at equal wall time.

