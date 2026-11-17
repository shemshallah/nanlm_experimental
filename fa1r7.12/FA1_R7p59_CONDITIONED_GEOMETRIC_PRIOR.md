# FA1 R7.59 — word-conditioned temporal geometric Bayesian prior

R7.58 trained successfully to a 2.38% DEV NLL improvement but its geometric
gate averaged only 0.0032, versus 0.1414 for exact-bigram evidence. Its
geometric advantage over the matched exact-bigram control was +0.001304
nats/word (95% book CI −0.000091 to +0.003060): insufficient for promotion.

R7.59 retains the full single FA1 model and 32-group R7.58 checkpoint. It
appends **one 33rd optimizer group**, with four trainable tensors, while the
entire inherited model and all Adam moments are frozen.

## Geometric distribution

Rather than pooling two words through a coarse 256×256 address, R7.59
conditions on the **exact most recent word**, but pools the older word through
its second 256-head endpoint:

`address = 256 * recent_word_id + head[older_word_id, 1]`

This is a sparse, at-most-8,192,000-address bank of normalized TRAIN-only
successor counts. Only document-contained causal triples are stored. The
apex receives a 32D learned descent and support features.

## Fair matched prior

For exact two-word context `c`, let `n(c,y)` be its TRAIN-only target count
and `n(c)` its total. With a learned common pseudocount `k`, the two tested
normalized distributions are:

```
Q_geom_smooth(y|c) = [n(c,y) + k Q_geo(y|recent, older_head)]/[n(c)+k]
Q_lex_smooth(y|c)  = [n(c,y) + k Q_word(y|recent)]/[n(c)+k]
```

The lexical control has **identical exact two-word evidence**, identical
strength `k`, and an equally normalized predecessor-conditional prior. Only
the context-dependent geometric abstraction differs. Missing geometric
support falls back to `Q_word` rather than introducing unnormalized mass.

The new path is a normalized residual mixture, bounded by 0.18, of the
inherited full R7.58 distribution and the smoothed geometric distribution.
Zero initial gain gives exact R7.58 checkpoint parity. TRAIN gradients remove
the current supervised edge from R7.57, R7.58, and R7.59 count banks.

## Rejection and evaluation controls

Independent 15-book DEV tracks include inherited R7.58, full R7.59, matched
lexical smoothing, wrong older-word geometry, wrong geometric successor
values, and no apex. The strict gate requires positive 95% book-bootstrap
lower bounds for improvement over R7.58, over lexical matched, and correct
geometry over both wrong-history/value controls.

No automatic +9,000 steps or independent TEST release on a failed gate.
Source CI tests target-independent routing and inherited optimizer safety.
Raw-text BPB and full top-5 word accuracy remain unavailable from the
inherited word-NLL forward API and are not fabricated.

The checkpoint-backed training entry point is
`fa1r7.12/source/fa1_r759_cpu_pilot.py`, using the original NaNROUTER
TRAIN/DEV recovery contract and an unpromoted R7.58 parent checkpoint.
