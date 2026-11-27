# v58 exact open vocabulary bridge
v57 prefix-screened DEV (2880 targets): count 387/803 top1/5;
v50 394/820; v53 412/872; v55 444/874. v55 retains lower NLL
on each domain; KN5 alone did not beat v55, KN+v55 selected on cal
regressed top1 and paragraph NLL. No new checkpoint promotion.
Lift categorical probability vector including unknown id0 into exact
word prefix law. Independent positive spelling model fitted only to
reference docs; condition on complement of dictionary via bottom-up
trie survival (no cancellation). Known word probabilities exactly
preserved up to normalization roundoff; unknown mass reserved for
actual OOV words. Local executed preflight: 414 prefix conservation
checks max error 1.1102230246251565e-16; 96 literal events max error
3.469446951953614e-18; token marginal max 3.774758283725532e-15;
folded case/stop marginal max 2.7755575615628914e-17. Exact disabled
path and known-word probabilities on constructed test. Not corpus accuracy.
New workflow first runs finite order3 raw donor and exact folded event
law on v57 screened prefixes, then recovers the original v37 chain and
compares native raw head, parent word law, v55 lift and calibrated mixture
on matched literal byte events. Native donor depends on full history:
no claim of exact folded marginal or token rank for that path. Native
sample count 4 events/book vs 12 for finite donor; compare only within
stage. These are sampled-event BPB, not contiguous stream. Full artifact
hash and inherited v37 weights/Adam/lexical state/step/RNG checks.
No changes to inherited weights. Historical calibration overlap remains,
raw training prefixes first 16384 bytes may extend beyond screened word
training prefixes; prefix screen is not complete content disjointness.
Integrated corpus pilots not yet run. No GPU claim.
