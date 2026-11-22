# NaNCM v168 — Signed Analogical Birth (EXPERIMENTAL; NOT PROMOTED)

Author: **Justin Howard-Stanley**, Independent Researcher, Santa Cruz, California, USA.

**Lineage:** v167 remains active development baseline, v166 its unchanged trained lexical and 128+128 signed-cloud parent, v158 protected rollback, v148 fallback. NaNCM v168 experiments remain inside `experiments/nancm/v168/` on `master`; no other branch, no core checkpoint modification.

## Architectural pivot

The v164/v165 branch could only rerank exact context words (14.58% support on 480 sampled DEV words). v166 broadened to 12,000 learned words. v167 gave it causal agreement admission. v168 builds **a signed analogical-neighbor generator** which proposes word candidates using learned concept geometry rather than requiring an exact context key:

1. Read 199,956 conditional address keys / 733,069 candidate entries from the 500 NaNROUTER TRAIN books. No DEV/TEST word labels are inserted into the search index. Sidecar `checkpoints/v168_signed_ann.bin` (51,845,562 bytes, `N168ANN1`) fingerprint binds the original model and v166 lexical predictor.
2. For each prior 1–3 whole-word context, create a 32D learned lexical state plus a 16D spectral projection of inherited 128+128 oriented S+/S− cloud. Query 32 nearest TRAIN addresses (approximate inner-product in this low-dimensional space) with signed weighting 0.66 and lexical weighting 0.34. Project retrieved outputs into a normalized 64-word proposal; this can contain words not found by the v164 exact-context table.
3. The C++ wrapper composes a **normalized whole-word distribution** `P168(w|h) = (1−β)P167(w|h) + β R168(w|h)` and maps it to 27-symbol conditional prefix/EOS probabilities. Context is causal, no future target text is used. `V168_ANN_STRENGTH=0` gives *bitwise identical* v167 output.
4. `V168_ANN_STRENGTH=.05` was selected on the sampled DEV window. The model remains research-only; the v167 active binary and optimizers are unaffected.

## Tests

From this cumulative recovery archive:

```sh
make -B v168-ann
OPENBLAS_NUM_THREADS=1 make test-v168-ann
# repeat historical baseline
OPENBLAS_NUM_THREADS=1 make test-v167
# native full context benchmark (DEV and previously reused TEST)
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v168/bench_ann_native168.py --help
```

For opt-in ANN: set `V168_ANN_STRENGTH=.05` and `V168_ANN_INDEX="$PWD/checkpoints/v168_signed_ann.bin"` alongside the existing v167 and v164 native environment, then load `bin/libwordfield168_ann.so`. All ancestor models/checkpoints must remain configured. Exact rollback: `V168_ANN_STRENGTH=0`. Invalid sidecars or strengths fail closed.

Test summary: **450 exact v167 rollback prefix vectors; 1,440 active normalized prefix vectors; 1,440 exact future-suffix-exclusion tests; 4,545 exact historical documentation prefix vectors; five invalid configuration cases**. The newer branch is admitted on 244/480 sampled DEV contexts and zero historical documentation contexts.

## Frozen results — strict 96-beam complete word decoding

| Book cohort | Model | Exact top-1 | top-5 | top-32 | Spelling/EOS bits/symbol |
|---|---|---:|---:|---:|---:|
| 24 DEV books, 480 sampled positions | protected v167 | 46 | 94 | 150 | 2.558560 |
| same DEV positions | v168 β=.05 | 46 | 93 | 158 | 2.539515 |
| 24 DEV books, all 1,394 available window positions | protected v167 | 141 | 269 | 465 | 2.626200 |
| same 1,394 DEV positions | v168 β=.05 | 141 | 267 | 477 | 2.604314 |
| 24 reused TEST books, 480 sampled positions | protected v167 | 46 | 104 | 162 | 2.512776 |
| same reused TEST positions | v168 β=.05 | 47 | 103 | 165 | 2.496832 |

Do not confuse spelling/EOS bits/symbol with full 256-byte BPB. Historical TEST windows have been accessed in prior research, so are not pristine. The candidate union 192/480 parent+ANN top32 is **support coverage**, not a claim of decoder word accuracy. Larger β=.35 reduced DEV top-1 and is rejected. β=.05 trades a small top-5 regression for better top-32 and likelihood; **not sufficient for promotion**, and no meaningful advance towards 80% first-choice accuracy is claimed.

## Other arms, not promoted

* `train_raw168.py`: inherited v166 signed-cloud AdamW and whole-word generator continued on real next-word labels from 80 distinct TRAIN books; 20 other TRAIN books for validation. 250 steps. Word NLL 6.303979→6.291385, top-1 unchanged (10.77%), top-5 worsened (23.62→23.27%). Separate DEV generator NLL 6.7371→6.7158 but top-1 36→34. Preserve this distinct checkpoint `v168_raw_candidate.pt` and its optimizer history; do not replace active v167 weights.
* `train_gate168.py` and `wordfield168_api.cpp`: trained causal mixing gate lowered DEV spelling/EOS loss to about 2.4676 but DEV strict top-1 46→40, rejected.
* `train_router168.py`: TRAIN-only modal rank selector 53→55/324, but DEV top-1 46→43, rejected.

## Outstanding gates

New truly untouched books and full corpus decontamination; full 256-byte BPB and next/+2/+3; strict top-1 gain, OOV beyond fixed lexical head, Android/Termux. v167 remains active until a successor passes the designated top-1 and broader multi-book gates. There is no 80% predictor here.