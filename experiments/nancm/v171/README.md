# NaNCM v171 — Causal signed-cloud retrieval scaling

**Author:** Justin Howard-Stanley · Independent Researcher · Santa Cruz, California, USA.
**Status:** opt-in native research, **not promoted**. Active word model remains v167; inherited byte decoder remains v170. Experimental code is on existing `master` only.

The native v171 library retains trained 128+128 S+/S− cloud, v166 32D lexical head, 16D spectral ANN, v169 native byte bridge, and v170 prequential 256-byte interface **unchanged**. Two bounded and optional causal candidate-generation parameters are added:

- `V171_ANN_NEIGHBORS`: 16–512, defaults to 32
- `V171_ANN_CANDIDATES`: 32–1024, defaults to 64

The default 32/64 path passed 40 **exact v169** candidate and 27-way prefix rollbacks. The native v171 library also passed v170's 16,018-byte next256 DEV/reused-TEST reference replay, with 292 exact v169 fallback vectors and unchanged model/optimizer hashes.

## Matched ANN candidate recall, 244 previously explored DEV contexts across 16 books

| Neighbors/candidates | ANN recall | ANN top-1 | Combined native/lexical/ANN upper bound |
|---|---:|---:|---:|
| 32 / 64 | 93/244 | 21/244 | 111/244 |
| 64 / 128 | 112/244 | 22/244 | 126/244 |
| 128 / 256 | 131/244 | 22/244 | 138/244 |
| 256 / 512 | 148/244 | 23/244 | 151/244 |
| 512 / 1024 | **160/244** | 20/244 | **167/244** |

For clarity, the original *v169 portfolio* captured only 48 ANN proposals, giving a 109/244 union oracle. That differs from the matched native 32/64 upper bound of 111/244.

**Crucially, 96-beam full-decoder strict word top-1 did not improve**: 27/244 at 32/64, 128/256, 256/512 and 512/1024. Top-5 at 32/64 was 57/244; at wider settings 56/244. Therefore v171 must not become the default until a genuinely stronger selector is trained and independently tested.

The next-byte v170 performance is preserved: DEV 2.563848894 BPB (8,015), previously reused TEST 2.472756834 BPB (8,003). These windows and DEV selections are not pristine certification.

## Reproducible recovery

The cumulative v171 recovery ZIP (stored separately in ChatGPT Library) includes trained weights, full optimizer history, ANN and byte sidecars, source and native library plus benchmark outputs. GitHub is source-only. From the recovered ZIP run `make test-v171` or the individual Python scripts in this directory. This experiment is not evidence of 80% strict next-word accuracy, Android readiness or untouched generalization.
