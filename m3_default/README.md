# Canonical M3 online predictor — signed successor memory default

M3's default is now the R20 **head → signed mirror-pair → ordered-triangle
successor** path, connected to the inherited R19 spatial-wave probability.
It is not an FA1, FA2, NaNLMind VM, or Terminal-RSI modification.

The public Python API is M3DefaultPredictor. It accepts the exact original
R16 TRAIN-derived 256-entry prior and exposes observe(byte),
predict_proba(), predict_logp() and reset(). No target byte may be observed
until after its probability is read.

Default mode is "r20" with frozen, TRAIN-selected:
- R19 space/non-space scales: 0.25 and 1.0.
- R20 directed-pair/ordered-triangle scales: 0.5 and 0.5.
- R16 frozen oscillator and spatial coefficients.
- 256 heads, 32,640 signed pairs, sparse 8,192-transition triangle history.

R19/oscillator, wrong-address, wrong-context, and triangle-only modes remain
accessible with the **same state** for causal ablations, not independent models.
A zero R20 residual recovers R19 exactly; reset starts a new book.

## Acceptance gates

R20 DEV on 12 books reduced 3.769696 → 3.461079 BPB, with non-space and
top-1/top-5 improvements. This was a **development** result, not a checkpoint
promotion. The R21 workflow performs a predeclared frozen TEST on 24 further
untouched books at seeded indices 132–155.

Unit tests enforce online-vs-batch R15/R20 numerical parity, read-only
prediction, exact distribution normalization, topology invariants and a
conventional n-gram equivalence audit.

Important: signed pair/ordered triangle counts are an address-isomorphic
encoding of ordinary bigram/trigram counts. This default is an engineering
integration of a useful causal memory mechanism, not yet proof that geometric
computation beats an equally provisioned conventional n-gram or recurrent
model. Independent TEST success and capacity-matched baselines are distinct.

Usage:

    import numpy as np
    from m3_default import M3DefaultPredictor
    predictor = M3DefaultPredictor(original_train_prior)  # shape (256,)
    for byte in document_prefix:
        predictor.observe(byte)
    next_byte_distribution = predictor.predict_proba()

No original checkpoint or optimizer state is overwritten. 
