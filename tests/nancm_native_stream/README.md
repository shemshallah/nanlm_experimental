# v64 native recurrent stream and bounded ranking
v59 actual native run45927769016 SUCCESS, immutable v37 parameters/Adam
verified. Sampled 543 bytes: raw 2.88129549/2.92272868/2.60754688
(stories/sentences/paragraphs), native word bridge 2.47631481/2.54085605/
2.15252004, hybrid 2.51083004/2.50826773/2.10451315. No native
folded word accuracy claimed. v63 still active at implementation.
Cache by entire literal prefix, immutable probability vectors, bounded
2048-entry LRU, no suffix approximation, use only frozen native donor.
Local executed exhaustive test: 702 possible completed words up to
length2, exact top5 probability error0, certified after19 logical
queries, insufficient budget correctly uncertified, cache eviction
and reset checked. Constructed donor, NOT actual native speed/accuracy.
Native rank uses exact casing histories and conservative prefix bound;
cap events excluded from completed-word ranking, no approximate
pruning. Budget counts all raw queries even hits, no partial expansion.
Pilot: recover native v59 and trained v61 artifacts; replay all same
v59 sampled native windows and report BPB/rank parity, calibrate a native
persistent switch on first3 screened cal books 128+ bytes, evaluate
first2 screened DEV books/domain 128+ contiguous bytes, raw/native
static/native switch/v59 fixed/transferred v61 controls. Native transfer
is frozen v61, not new native training; previous likelihood features
computed from native expert likelihoods only after observed event.
One preselected complete event per domain for bounded folded top5,
native token marginal, exact arithmetic roundtrip with fresh donor
memo at decoder. Do not report full native word/token accuracy from
these audits. Inherited weights and optimizer histories immutable.
Report cache misses/hits and elapsed time; speedup not claimed until
actual run. Historical calibration overlap and finite screened prefixes
remain. No production promotion. GPU parity still outstanding.
