# FA1 R7.60: support-aware geometric shrinkage

R7.59's word-conditioned geometry improved over R7.58 but lost to a matched
lexical prior (geometric advantage -0.001938 nats/word; 95% whole-book CI
[-0.003896,-0.000138]). Sparse pooled trigram conditionals were too sharp.

R7.60 adds a learned, bounded calibration between the inherited R7.59
geometric conditional and an exactly matched lexical backoff:

`Q_760 = Q_lexical + clamp(1 - learned_shrinkage * sigmoid(selector), 0, 1) * (Q_759 - Q_lexical)`

The selector sees the inherited 32D apex and causal support counts, but not
the next-word target or its identity. Thus the prediction stays normalized,
the selector is causal, the lexical control is matched and the geometric
conditional can be shrunk but not artificially exaggerated.

- One additive three-tensor optimizer group (34 total).
- Complete 33-group R7.59 parent frozen; AdamW moments retained.
- Exactly function-preserving attachment in inference mode.
- New 1,000-update TRAIN-only CPU candidate and 15-book DEV comparison.
- A separate calibration-vs-uncalibrated R7.59 ablation and whole-book CI.
- Promotion requires significant gains over parent, matched lexical,
  wrong-older, wrong-value and uncalibrated controls.
- Independent TEST remains sealed until all DEV conditions pass.

The experiment does not claim geometric superiority or promotion until the
actual real-checkpoint CPU results have been obtained and audited.
