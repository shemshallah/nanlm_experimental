# FA1 R7.39 — causal signed S+/S- cross-rail continuation

**Lineage:** FA1 unified R7; on `master`; source parent R7.37, with R7.38
reliability and R7.39 cross-rail paths added in-place. No accepted checkpoint
or optimizer history is reset or promoted by this experiment.

## Mechanism and information boundary

The R7.30 factorized geometry selects top-eight exact C(256,3) triangle
addresses from the currently observed context. Its R7.35 writer attaches
pre-target, older modal values only at those eight addresses. R7.38 provides
a bounded reliability multiplier. R7.39 adds two learned projections of the
*same observed causal far values*, separately weighted by the inherited
S+ and S- exact-address face coefficients. Their quarter-turn cross is

`cross(p,m) = p * roll(m,1) - m * roll(p,1)`.

Swapping rails negates this cross. The learned projection and zero-initialized
gate modulate **only already-written** triangle values. No write appears
on an unselected triangle or with a fully masked old-history row. Every
result continues through the inherited apex, pair and point feedback loop
to the inherited word and conditional autoregressive UTF-8 spelling heads.
No target word or target byte is used to construct an address, write or rail.

The attach function appends one AdamW group without replacing Parameter
objects, historical optimizer engines, or their moments. R7.37 input,
R7.38 reliability, and R7.39 signed modulation are separate ablatable paths;
the last two gates initialize to zero for exact R7.37 prediction parity.

## Verifiable material and training protocol

The exact R7.37 parent is `FA1_R737_CPU48_UNPROMOTED.pt`, SHA-256
`74a4112c97ccd828b485a7a329dee01c8fc66a0b6cb42bb49f9da3cdf6026046`.
The original R7.13 tokenizer and 32K-word tree are restored from the
recoverable full checkpoint archive. NaNROUTER `short_stories-0004.tar.zst`
is SHA-verified as
`d6ce3a23757e85fe1b9487ef984a5ba81db9c2dc6e962ac7c5df2960480e0054`.

The deterministic, content-SHA-disjoint split contains 270 TRAIN books,
15 DEV books, and 15 TEST books. Holdouts contain at least 65,536 words.
The original checkpoint's unknown prior corpus overlap remains unresolved.
TEST is reserved and is not consulted for checkpoint selection.

The bounded continuation uses 240 TRAIN predictions, 192 DEV predictions,
minibatch 8, 24 joint gradient updates, two CPU threads, learning rate
0.00012, and seed 739. The candidate is always written separately as
`FA1_R739_CPU24_UNPROMOTED.pt`, and strict model/optimizer restoration is
required before any promotion discussion.

Selection criteria: word NLL, conditional spelling BPB, first-byte and
spelling top-1/top-5, correct-versus-foreign-book writer specificity,
R7.38 reliability ablation and R7.39 signed-rail ablation on word/byte
losses, plus independent-book bootstrap intervals. Positive training loss
alone is not sufficient. The strict gate requires both signed-rail
contributions and specificity to be positive on word and conditional
literal-byte objectives, including positive lower bootstrap bounds.

**Metric limits:** conditional spelling BPB excludes EOW and does not
measure continuous raw-text compression. Inherited-trunk word top-k does
not represent full-mixture word top-k. No stronger claim is inferred.

## Results

The 24-update run completed. DEV selected **update 1**, on the loss-only
selection tier (strict gate false). The exact R7.37 checkpoint remained
unchanged. Its 21 inherited optimizer groups plus R7.38 and R7.39 groups
were retained as **one 23-group optimizer**, not independent replacement models.

| DEV metric | R7.37 parent | selected R7.39 step 1 | step 24, rejected |
|---|---:|---:|---:|
| Full word NLL | 4.968646 | 4.968767 | 4.969709 |
| Conditional spelling BPB | 5.331635 | 5.314221 | 4.986161 |
| Conditional spelling top-1 | 6.7449% | 7.6246% | 9.5308% |
| Conditional spelling top-5 | 36.8035% | 36.8035% | 37.6833% |
| First-byte top-1 | 13.0208% | 13.5417% | — |
| First-byte top-5 | 38.5417% | 38.0208% | — |

At selected step 1, disabling the new signed-rail interaction changes word
NLL by approximately zero; the signed-rail effect on conditional spelling
is `1.54e-8` nats/byte, with correct-minus-foreign specificity
`1.05e-8` nats/byte. Its book-bootstrap confidence intervals include zero.
The training-path gradient is finite but its magnitude is too small to
support predictive usefulness. None of steps 1/8/16/24 passed the strict
gate. No independent sealed TEST was opened for this rejected candidate.

The selected candidate has SHA-256
`28d450550317a716eadaf513f600a59360b145ea2281f79affd1418255b7a007`
and 266,058,024 bytes. It is **recoverable but unpromoted**, and is the
direct optimizer-preserving parent for the R7.40 amplitude-correction study.

### Numerical failure diagnosed

On the identical 192 DEV predictions the selected top-eight S+/S- face-code
RMS magnitudes are 0.01796 / 0.01715. The new cross modulation has only
`7.55e-7` RMS and `2.58e-5` maximum absolute magnitude, even though 80.73%
of examples have eligible far history. The signed product loses scale before
it reaches the inherited causal value read. R7.40 therefore adds an
identity-migrated normalized-rail control; see its separate round report.
