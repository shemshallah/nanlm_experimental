# v57 content-disjoint prefix screening and absolute-discount backoff
v55 completed, exact historical parent replay; paired DEV NLL improved on
all domains and true history beat three wrong contexts. v56 weak gains
and replay discrepancy, reject. Actual 32-word shingle audit found 1041
calibration matching positions against reference, 72 story DEV positions,
1088 positions shared between sentence DEV files, and 325 additional
paragraph positions against reference. Book IDs and whole-file hashes are
not sufficient. Existing v50/v53/v55 artifacts remain immutable; historical
calibration overlap means these are diagnostic rather than certified scores.
v57 screens BEFORE evaluation: reject any exact 16-word sequence overlap
with reference indexed prefixes and primary/secondary fitting prefixes,
or any earlier selected document, plus ID and file hash duplicates. Ten
calibration and ten new DEV books per domain, exclude all 36 historical
v55/v56 DEV IDs; reserve split bucket9 untouched. Calibration may reuse
historical calibration books that pass the new screen. Prefix-only
screen, no approximate duplication or full-document disjoint claim.
New exact tuple-indexed interpolated absolute-discount continuation law
(max five preceding words, D=.75, distinct-predecessor lower counts,
1% prior escape at unigram). Compare count, v50, v53, v55, KN5, truncated
KN3 using same lower continuation tables, and calibration-selected KN
mixtures with frozen v50 field or v55. Wrong-field control retains KN.
Local executed test: 2540 normalizations max 2.220446049250313e-16,
75 independent direct-count comparisons exact, two overlap cases rejected.
No new neural weights, no reset, no production promotion. Actual corpus
evaluation queued by push; categorical unknown class only, no raw BPB.
