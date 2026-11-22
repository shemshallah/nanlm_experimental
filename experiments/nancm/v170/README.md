# NaNCM v170 — Native prequential signed-cloud trust

Author: Justin Howard-Stanley · Independent Researcher · Santa Cruz, California, USA.
Date: 2026-11-22. **OPT-IN RESEARCH**, not a whole-word promotion; **v167 stays active**.

## Symbiotic architecture

This builds directly on v169, retaining the existing 128+128 signed S+/S− cloud, lexical head, ANN, byte/word bridge, 256-way byte n-gram and all checkpoints. The only addition is bounded **causal online evidence** favoring the expert that better predicted *already observed* bytes.

For the relevant regime (inside/outside ASCII words), after observing the previous byte:

\`E_regime = 0.9 E_regime + log(P_word(previous_byte)/P_ngram(previous_byte))\`

\`eta = clamp(sigmoid(logit(eta169) + 0.2 E_regime), 0.01, 0.95)\`

\`P170(next byte | past) = (1-eta) P_ngram + eta P_word\`

The category-specific memory state updates only on exactly sequential histories. Repeated same-history calls are idempotent. An unrelated jump automatically resets online evidence; explicitly call \`wf170_byte_reset\` at document boundaries. No future/target byte is available to its own prediction. The parameter family was DEV-selected (therefore not an independent test).

## Native full 256-way next-byte results

| Split | v169 BPB | v170 native BPB | Gain |
|---|---:|---:|---:|
| 24 DEV windows, 8,015 bytes | 2.587375523 | **2.563848894** | −0.023526630 |
| 24 reused TEST windows, 8,003 bytes | 2.505834321 | **2.472756834** | −0.033077488 |

Full native replay and independent Python probability oracle pass; original local test measured **292 exact v169 rollback distributions** (maximum difference 0), 16,018 positive normalized 256-way predictions, expert/gate-state parity to numerical precision, repeated-query idempotence, reset, and setting validity. The complete archive contains the trained sidecars and exact regression logs; GitHub is **source-only**. The GitHub regression script reproduces the full byte-level comparisons.

## Build from complete recovery ZIP

Append \`Makefile.fragment\` to the Makefile; then run \`OPENBLAS_NUM_THREADS=1 make test-v170\`. The new library exports \`wf170_byte_open\`, \`wf170_byte_next\`, \`wf170_byte_reset\`, \`wf170_byte_set_adaptive\`, \`wf170_byte_last_eta\`, and \`wf170_byte_close\`. \`wf170_byte_set_adaptive(handle,0)\` returns exact v169 behavior. \`wf148_*\` and \`wf169_byte_*\` are unchanged. All optimizer states and signed-cloud weights are retained.

**Limitations:** the gate was chosen on DEV and TEST books were previously explored; this is not untouched-test certification. It improves byte log loss, **not strict whole-word top-1**; no 80%-word, +2/+3 horizon or Android-performance claim.
