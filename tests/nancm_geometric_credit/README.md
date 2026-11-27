# v62 unified event credit into inherited signed geometry
v60 actual completed 7754 contiguous bytes: stories BPB 2.836137477 ->
2.676952028; sentences 2.483069260 -> 2.493885998; paragraphs
2.992674153 -> 2.957057629. Retain architecture v59; no unconditional
promotion of switching weights. v60 chosen prior (.5,.4375,.0625),
hazard .35, used identically in v62.
Compose differentiable full vocabulary v50 output with frozen v53
count/analogy affine mixture and v55 copy/word/class contrast. Exact
event coefficients for known, OOV, capped prefix, empty events.
Compare real composed probabilities and literal event likelihoods
against existing inference implementation on first 32 corpus events.
Local executed logit derivative test: 136 finite differences max
5.541468325875876e-11; max normalization error 2.220446049250313e-16,
119 nonempty coordinates nonzero, empty event zero gradient. Workflow
also tests temporal recursion against v60 forward filter.
Continue original coupled v50 AdamW 768 -> 864 steps, 96 identical
contiguous segments of <=8 events, LR .0003, word-credit vs raw-event
credit. For raw arm exact differentiable causal posterior recursion
within each segment; starting posterior supplied by frozen parent after
observing only preceding events. This is truncated BPTT, not global
sequence gradient. No gradients into frozen count/analogy/contrast
gates or original v37 native parameters. Fit on primary+secondary
128-byte contiguous windows (previously used for component fitting);
evaluate parent and both continuations on same first 3 screened books
per domain and calibration, 256+ contiguous bytes each, v60 adaptive
law, token/word ranks. Small diagnostic; no production promotion.
Actual integrated corpus execution pending. Preserve inherited files,
save full continued field and moments. Historical overlap and
prefix-only screening remain limitations.
