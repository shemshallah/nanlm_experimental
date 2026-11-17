# M3 R27 — one-shot independent acceptance protocol

R26 has passed its frozen four-book GATE and 12-book DEV. It improved
all 12 DEV books and lowered next-byte BPB 3.509647 → 3.448599,
non-space BPB 4.120100 → 4.046099, and raised top-1 37.01% → 39.18%.
Its candidate was fixed *before* any evaluation of independent TEST books.

`r26_frozen_candidate.json` records the exact 14 coefficients, all 14
optimizer step counts, the full recoverable trained NPZ SHA-256, DEV report
SHA-256, model source commit, and acceptance requirements. Its 24 untouched
TEST books are the pre-reserved indices 276–299 from the SHA-verified public
NaNROUTER short-story batch, using the original selection seed, 1,024-byte
warm-up, and 4,096-byte excerpt length. They have not been used by any R26
TRAIN, GATE, DEV, or parameter-selection code.

The locked workflow verifies both frozen artifacts and the original ancestry,
then independently scores R24, frozen R25, frozen R26, wrong-successor, and
unrelated-book controls. Each byte is scored before being observed. No
optimizer executes. Promotion requires positive overall/nonspace improvement,
a positive whole-book bootstrap lower bound, at least 18 of 24 books improved,
top-1/top-5 nonregression, and superiority over both negative controls and
frozen R25.

The one-shot test is independent of the DEV-gating decision and no tuning may
occur after its results are observed. A successful test makes R26 eligible for
promotion; the original R24 implementation must remain available as an exact
fallback.
