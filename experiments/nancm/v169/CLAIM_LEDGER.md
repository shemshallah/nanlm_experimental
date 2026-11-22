# v169 Claim ledger — 2026-11-22

| Claim | Evidence | Decision |
|---|---|---|
| Native 256-way next-byte predictor exists | `native_byte169.cpp`, `test_native_byte169.py`, `results/v169_native_byte_regression.json` | **Verified** on Linux CPU |
| Exact word ABI rollback to v168/v167 | 450/300 exact 27-way vectors in `results/v169_native_regression.json` | **Verified** |
| Normalized and causal word predictions | 1,440 native normalized and 1,440 poisoned-future excluded | **Verified** |
| Protected historical docs unaffected | 4,545 exact vectors and zero new ANN admissions | **Verified** |
| Native byte vs Python reference parity | 288 whole 256-way vectors, max abs diff 3.5556e-8; full 8,015/8,003-byte matched BPB | **Verified** |
| Signed ANN uses real history | 79/244 correct top-32 vs unrelated 27/244; top-1 21 vs 4 | **Supported for ANN branch** |
| DEV mixed byte BPB improves over byte n-gram | 2.844988→2.587376; 24/24 book windows | **Measured on DEV; selected weights** |
| Reused TEST mixed BPB improves over byte n-gram | 2.820095→2.505834; 24/24 book windows | **Measured on reused TEST** |
| Improvement is strict word top-1 | v167→v169 DEV 46→46/480, broader 141→141/1,394 | **Not established** |
| 80% strict word top-1 | far below target | **OPEN** |
| v169 word reranker promotion | no DEV top-1 gain, modest DEV top-5 loss | **Rejected** |
| Production/generalization certification | no pristine untouched book corpus, contamination audit, Android build | **Not established** |
| 256-byte +2/+3 free-running | not measured | **OPEN** |

Caution: historical 27-way spelling/EOS bits per symbol are not 256-way byte BPB. v169's 256 BPB refers to a newly constructed and scored byte model. All findings are recoverable and not claims of frontier language-model performance.