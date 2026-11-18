# M3 R32 — frozen TRAIN-only anchor memory

R31 is the **directed canonical default**, despite failing unrelated-history
controls. This R32 experiment tests a specific hypothesis: unrelated-history
gains in R29–R31 partly reflect lexical smoothing from extra generic text.
R32 therefore supplies an immutable, recoverable background lexical memory
built **only from disjoint TRAIN books**. It never uses another GATE, DEV or
TEST book as a predictive memory source.

R32 augments the inherited R31 distribution with four learned logit channels:
training-book prefix statistics, training-book word-conditioned statistics,
a difference between global and causal local lexical statistics weighted by
global support, and a space intercept. Zero gains reproduce the *trained R31*
prediction bitwise. R31/R30/R29/R26/R24/R21 all remain selectable with exact
inherited coefficients and causal observation semantics.

- Full parent 28-group checkpoint from Actions run `44227555060`; SHA-256
  `dd8fbaa93a17e5568bf2dc774b5599e5761824dee74db8f71242039f743a12ff`.
- 28 inherited weights and optimizer states are copied without modification;
  only four new Adam groups are trained. The R32 checkpoint embeds the original
  TRAIN-only anchor bytes and document boundaries to make it self-contained.
- NaNROUTER Release `corpora-short-stories-0004`, canonical seeded book order:

| Split | Indices | Use |
| --- | --- | --- |
| ANCHOR | 468–483 | Immutable cross-document prior, not gradient-fitted |
| FIT | 484–499 | 16 books, train four R32 gains |
| GATE | 500–503 | Held-out acceptance screen |
| DEV | 504–515 | 12 independent books, whole-book bootstrap |
| TEST | 516–539 | Sealed; neither loaded nor scored |

All comparisons use the same R31 parent prediction. Alongside R29 and R31,
controls test wrong-value background, unrelated same-prefix-length history
and unrelated full-document history. A passing R32 must beat R29, R31, wrong
anchor and matched unrelated memory in GATE and DEV; keep or improve nonspace
BPB and top-1/top-5; improve at least 3/4 GATE and 9/12 DEV books; and clear
a positive DEV book-bootstrap lower bound. Passing DEV would warrant a sealed
TEST, **not automatic promotion**. A failed experiment leaves R31 unchanged.

Run:

```bash
python -m m3_r32.train_dev \
  --corpus-root /tmp/m3_r32_corpus \
  --parent-checkpoint /tmp/m3_r32_parent/R31_full_adam_checkpoint.npz \
  --output m3_r32/results/R32_DEV.json \
  --checkpoint m3_r32/results/R32_full_adam_checkpoint.npz
```
