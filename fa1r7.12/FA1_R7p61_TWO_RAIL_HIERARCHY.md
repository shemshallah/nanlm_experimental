# FA1 R7.61 — two-rail hierarchical geometric conditional memory

R7.59's single older-word head was weaker than the equal-smoothing lexical
control. R7.60's support calibration improved NLL but remained below lexical.

R7.61 adds a truly compositional three-scale memory from the two 256-point
rails. For the exact recent word, it reads:

1. Older-word head0 (coarse, independent rail).
2. Older-word head1 (the R7.59 signed-head predecessor).
3. Their joint ordered head0/head1 intersection (256×256 address lattice).

For a next-word value, the three sparse TRAIN-only conditional estimates are
pooled with support weights. A learned positive pseudocount interpolates this
geometric conditional with the matched lexical prior, and an inherited exact
older/recent bigram smooths it at the same strength as the lexical control.
A new learned mixture scalar and support-aware selector then combine it with
the inherited R7.60 predictor. All probabilities stay normalized. No target
identity is available to the selector, and the entire state is causal.

The full R7.60 checkpoint remains intact and frozen: 34 inherited optimizer
groups, four new tensors in one 35th group, one unified FA1 model. Default
initial attachment is exactly function preserving. The 1,000-step CPU pilot
uses 270 NaNROUTER TRAIN books, 15 book-disjoint DEV books, and 512-word
contexts, with leave-one-out TRAIN memory correction.

The decisive DEV arms are parent R7.58 (all new priors disabled), R7.61 full,
equally smoothed lexical, wrong older-word geometry, wrong geometric values,
and R7.60 hierarchy disabled. Promotion requires a positive lower 95%
whole-book confidence bound against all mandatory controls. Independent TEST
remains sealed until that condition is met. The +9000 updates are likewise
gated. Word NLL cannot be confused with bits/byte; raw-byte metrics need
separate supported data and an explicit byte prediction head.
