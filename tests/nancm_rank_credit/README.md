# v65 exact folded word-ranking credit
v63 actual 7754 bytes, parent replay rank counts exact, maximum BPB
drift2.35e-9. All 15 groups receive gradients; joint worsens BPB on
all domains and word top1 207->198/1279. Reject v63 learned states.
Retain v50/v61 frozen parents. Nine signed rails (8 inherited input
rails + inherited hidden), rank24 positive/negative projections and
shared bounded vocabulary transport, 58,377 parameters at 128D
(9*128*24*2+9+24*128). Separate normalization for both lexical
experts; exact zero bypass. Actual deployed folded completed-word
score includes finite-order raw donor, lexical known or OOV coefficients,
and causal v61 posterior weights. Hard negative candidates from parent
bounded rank search and highest lexical mass; candidate losses are
NOT exact global ranking loss. Exact event likelihood plus pairwise
softplus margin and .25 folded word NLL, compared with event-only.
Recompute causal warmup, detach at 12-event segment boundary;
128 identical segments. Only new adapter Adam moments updated; all
inherited files unchanged. Float32 cached training parent distributions
renormalized in float64; full inference uses original double laws.
Select scale on cal candidate wins subject to no BPB regression;
compare forced rank-aware full strength, selected arms and exact
parent on 30 v61 screened DEV books/7754 bytes. Record parent replay
and actual certified word/byte/token metrics. No production promotion.
Local executed constructed test (torch2.10 CPU): zero identity,
normalization2.22e-16, 4 gradient groups max FD6.094974525083785e-11
(one largest coordinate/group), temporal chunk error7.105427357601002e-15.
Actual v65 corpus run pending; v64 native still running. Prefix-only
screen and historical cal overlap remain. Native field not trained
by this finite-order pilot.
