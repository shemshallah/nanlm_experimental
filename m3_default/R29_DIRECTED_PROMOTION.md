# M3 R29 directed promotion — integrity and claim ledger

Decision: **default path switched to R29 by explicit user instruction**.
Scientific status: **NOT independently TEST-certified**. The R29 GATE
failed because unrelated-history lexical evidence produced lower BPB.

## Exact recoverable provenance

- Source/training commit: `2e0845c213e7926b7fb2b3866fe6b30eaa34230b`.
- Successful TRAIN/GATE/DEV Actions run: `44184424516`.
- Full 22-group R29 Adam checkpoint SHA-256:
  `02c340e82ec77ae2c6e1d20ddd808ca4f63e3c242fe6fc94ff8f2235d8ab9386`.
- TRAIN/DEV report SHA-256:
  `be2612709780c7d08888e351887f4501453984e52e16b47458271422569df829`.
- Actions artifact ID: `14373298578` (`m3-r29-lexical-full-adam-and-dev`).
- Parent R26 complete checkpoint SHA-256:
  `deb112d4d9b25cb57bd68f689660550a315032bea42c5e9376dbcdeb4a5c97a9`.
- Inherited optimizer ages: 3 groups at 1536 steps; 11 groups at 768.
- New lexical optimizer ages: 8 groups at 768 steps each.
- The inherited 14 weights and their Adam state arrays remain byte-identical.

## Measured independent-book development, NOT sealed TEST

| Arm | GATE BPB ↓ | DEV BPB ↓ | DEV nonspace BPB ↓ | DEV top-1 ↑ | DEV top-5 ↑ |
|---|---:|---:|---:|---:|---:|
| R26 accepted | 3.467857 | 3.470453 | 4.155572 | 39.625% | 69.964% |
| R29 directed | **3.452095** | **3.458221** | **4.139543** | **39.945%** | **70.195%** |
| Wrong successors | 3.639662 | 3.619872 | 4.307484 | 37.648% | 68.801% |
| Unrelated history | **3.436770** | **3.453804** | 4.138652 | 39.712% | 70.450% |

R29 improved 4/4 GATE and 10/12 DEV books, but unrelated-history control
did better than R29 in both. The DEV gain over R26 was +0.012232 BPB;
descriptive whole-book bootstrap interval +0.004429 to +0.018358.
The unrelated-history control preloaded the *complete* other book, whereas
actual R29 memory had only the causal prefix. The next ablation must compare
histories of equal length and separately report full unrelated-book evidence.

**No R29 TEST was read. No claim of learned geometry superiority is made.**

## Deployment invariants

- `M3DefaultPredictor(prior)` uses exact R29 weights.
- `predict_logp("r26")`, `"r24"`, and `"r20"` preserve inherited outputs.
- `mode="r26"` omits lexical memory, preserving the certified deployment.
- Checkpoint-dependent training remains in `m3_r29`; normal inference uses
  embedded frozen scalar coefficients only.
- GitHub CI gates exact R29 research/canonical parity, read purity, causal
  updates, document reset, and pre-existing regressions.
