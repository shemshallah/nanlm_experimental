# v43 learned signed edge credit
The v41 persistent graph supplies eight 128-dimensional context residuals.
Two independently optimized 8x128 sigmoid gain maps distinguish supporting
and opposing candidate point products. Evidence-conditioned rail routing
and an 8x8 trainable cross-rail interaction couple their output. The exact
finite-candidate normalizer retains all complement mass. The synthetic
preflight was executed locally: held-out 256 six-way targets, NLL 1.956684
to .920076, accuracy .738281 to .910156, unrelated residual accuracy .507812,
geometry-off .171875. Max gradient error 2.56539e-11, four geometric groups
nonzero; the linear features were deliberately zero in this synthetic test.
This is NOT corpus accuracy. Real training uses 12 reference books sampled
from the 288 extra reference books, 12 separate calibration books, and the
same ID-disjoint DEV selections as v41. Equal-budget exact-only and learned
geometry controls, independently trained, shared target-free proposal pool,
wrong-document graph and zero-scale rollback. No inherited model changes.
The new readout is not yet integrated into v40 recurrent triangle updates or
v42 byte/token events. Admission/eviction still uses fixed windows. No GPU
training or production promotion. Learnable geometry groups total 2128
parameters plus 25 linear features (2153 total).
