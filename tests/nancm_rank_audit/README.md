# v66 exact word search and failure attribution
v65 actual corpus run45932796711 succeeded but both arms selected scale
zero. Forced rank-aware 132/1279 correct vs v61 207/1279, BPB regressed
all domains. Reject learned adapters; preserve v61 parent exactly.
v66 changes no weights. Tight conservative bound on individual lexical
completion: max(largest known probability in prefix interval, OOV prefix
mass). Blend uses weighted sum of each component's bound; raw prefix
mass remains a valid bound on any completed word. Outward float margin,
strict prune only below kth score. Never certify a budget-limited result.
Local executed independent constructed finite-word test: exhaustive
18278 words, 585 prefix checks, exact top5, baseline45 expanded nodes
vs tightened12, insufficient budget uncertified. Workflow additionally
tests actual Lift, OOV spelling and Blend. Actual corpus audit on exact
v61 30 screened books and 7754 contiguous bytes; replay original ranks
and BPB, compare search cost, certification by 128 and 1200 expansions,
report lexical top1/5/32 coverage and mutually exclusive certified
error classes (OOV, outside lexical32, lexical6-32, lexical top5).
Raw-majority winning errors and top1 displaced diagnostic are
descriptive, not proof of causal intervention. Timing includes fresh
event caches per method; baseline has cached observed literal event.
No native speed claim, v64 still executing. No new training, no
production promotion. Historical calibration overlap and finite prefix
screen remain.
