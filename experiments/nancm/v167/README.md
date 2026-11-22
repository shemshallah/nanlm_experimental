# NaNCM v167 — Agreement-Gated Signed Lexical Birth

Author: **Justin Howard-Stanley**, Independent Researcher · Santa Cruz, California, USA.

**Development-only successor to v166.** This version composes the exact same trained v166 12,000-word lexical head and its v165-inherited 128+128 signed-cloud tensor/AdamW lineage with a causal, independent confidence gate. No upstream checkpoint is overwritten. It does NOT claim production certification, a full-byte BPB result, or 80% word accuracy.

## Architecture and decision

The native v167 API calculates `P(w|h)=(1-a(h))P164(w|h)+a(h)Q166(w|h)` and derives normalized 27-symbol prefix/EOS probabilities from the complete-word mixture, preserving the unchanged `wf148_*` C ABI. `Q166` is the trained 32D / 12,000-word softmax conditional on three preceding complete words with signed `S+` / `S-` 128+128 geometry. It is not a new unlimited open-vocabulary generator; the inherited parent handles words missing from its lexicon.

v166 used v164's book-domain threshold of 1.5 for the generative branch. v167 maintains **independent lexical admission** at a DEV-selected book-domain score ≥1.4, leaving the v164 table threshold 1.5 unchanged. Empirical documentation score maximum among 1,515 sampled earlier word positions was 1.34846; every one of 4,545 historical documentation prefixes remains exactly unchanged. The threshold is specific to the established fixtures, not a certification against arbitrary other domains.

The promoted candidate uses a **causal agreement ratio** `r=Q166(w_parent_best|h)/max_w Q166(w|h)` where `w_parent_best` is v164's strongest distribution-table candidate. All inputs are available before generating the next word, never from the ground-truth suffix. Only when `r≥0.5` does the model raise the lexical mixture from 0.05 to 0.50. This is agreement about candidate ranking, not a proof the predicted word is true.

The exact experimental configuration:

```sh
V166_LEXICAL_BIRTH=.05
V166_BIRTH_MODEL="$PWD/checkpoints/v166_lexical_birth.bin"
V167_GATE_MODE=1
V167_HIGH_ALPHA=.50
V167_TRUST_THRESHOLD=.50
V167_DOMAIN_MIN=1.4
```

Configuration mode 0 with domain 1.5 is byte-identical to v166; setting `V166_LEXICAL_BIRTH=0` disables the whole lexical head and returns exactly to v164. Invalid settings fail closed. v158 and v148 remain recoverable in the full cumulative archive.

## Frozen book-level results

Identical sampling: 480 teacher-forced next words from 24 DEV books, plus 480 separate words from 24 TEST books, 20 positions per book. Strict next-word decoding uses a 96-beam, maximum 20 letters, native prefix API. Both cohorts have been previously reused in development, so these are **corroborative development diagnostics, not pristine independent proof**.

| Metric | v166 DEV α=.05 | v167 DEV selected | v166 reused TEST | v167 reused TEST |
|---|---:|---:|---:|---:|
| Strict top-1 words /480 | 45 | **46** | 45 | **46** |
| Strict top-5 words /480 | 86 | **94** | 100 | **104** |
| Strict top-32 words /480 | 146 | **150** | 159 | **162** |
| Spelling/EOS bits/symbol ↓ | 2.624520 | **2.558560** | 2.635121 | **2.512776** |
| Generator active positions | 219 | **244** | 272 | **317** |

Relative loss reduction vs the promoted v166 conservative arm: 2.51% DEV and 4.64% TEST. Neither spelling/EOS bits/symbol nor finite-beam word top-k are equivalent to true full 256-byte bits/byte. The gain in strict top-1 is one word per split, so it is fragile and must be re-checked on untouched book cohorts.

## Separate rejected neural continuation

`train_margin167.py` was also tested: it resumed the full v166 checkpoint and its optimizer moments, applying count-weighted NLL + modal hard-negative margin + KL distillation on the same lossy TRAIN-derived distributional index. TRAIN-index context-split NLL improved 6.2340784→6.0854825 and modal top-1 16.016%→20.964% on 768 contexts. Yet book DEV at α=.05 retained 45 strict top-1 hits while losing top-5 86→83. The margin-trained tensor checkpoint is retained as an **experiment**, NOT the v167 active model. The v167 active model continues using the unchanged v166 neural checkpoint.

## Build, regression, interactive inference

Run from the root of the recovery archive:

```sh
make -B v167
OPENBLAS_NUM_THREADS=1 make test-v167
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v167/active167.py --prompt 'the mountain was' --steps 3
```

`active167.py` configures v164/v166 and v167 parent checkpoints, uses the actual `bin/libwordfield167.so` code path, and offers `--mode v166` and `--mode v164` for matched rollback inference. Original optimizer state is in `checkpoints/v166_lexical_birth.pt` and the retained `checkpoints/v167_margin_candidate.pt` is clearly marked rejected.

## Remaining open

- Pristine never-touched independent-book TEST, near-duplicate screening, inherited pretraining provenance audit.
- Raw-book neural training on all 500 TRAIN books (as opposed to truncated top-eight count index), long-context semantics, global surface-to-lexeme generalization, compositional OOV generation.
- Full 256-byte probability evaluation, next/+2/+3, BPE and token metrics, Android/Termux cross-compile, calibration confidence intervals and transfer tasks.
- 80% strict next-word top-1 is **not** achieved.

Keep `experiments/nancm/v167/FROZEN_DEV_DECISION.json` and the paired results as objective evidence; never silently promote improvements measured only on TRAIN-index contexts.