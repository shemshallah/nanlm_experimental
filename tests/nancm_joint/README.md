# v63 joint temporal geometry credit
v61 actual corpus 7754 bytes: BPB stories 2.601992957, sentences
2.460426261, paragraphs 2.919193723, all improve over v60. Certified
word top1 207/1279 vs v60 213/1279; no universal promotion.
v62 actual corpus: all 13 geometry groups receive gradients and both
Adam histories advance 768->864; mixed small DEV results, event credit
not superior. Preserve original v50 and v61 as component parents.
v63 adds 2304 signed hidden-to-transition weights, initially zero.
Same exact event likelihood plus .25 conditional lexical branch NLL
(includes known and OOV target, not a certified folded rank objective).
Previous event likelihood ratios now differentiable, rather than
precomputed from frozen predictor within each segment. Causal warm-up
recomputed with current weights; detach at max16-event chunk boundary.
128 identical segments, allocation-only and jointly trained controls;
v61 Adam 96->224, v50 Adam 768->896 only joint, separate new adapter
Adam 0->128. Frozen v50/v61 parent control. Fit 24 unindexed 128-byte
windows previously used for credit; evaluate all 30 screened v61 DEV
books and 10 calibration books, exact same 256+ byte windows. Parent
replay delta and rank equality reported; no unverified promotion.
Actual local test executed on constructed data: 233 finite differences,
max 7.473878352141916e-11, disabled signed adapter v61 event parity 0,
chunk continuation error 1.7763568394002505e-15, transition/adapter/
lexical logits/hidden gradients nonzero. This is NOT corpus accuracy.
No native v37 parameter modification; v59 native still running.
Historical calibration overlap and finite prefix-only screen remain.
