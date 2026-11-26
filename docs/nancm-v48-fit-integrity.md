# v48 fit integrity correction
v43 trained on documents included in its reference continuation index,
achieved 100% fitting candidate recall, and selected zero transport on the
unindexed calibration books (NLL 5.939168 -> 6.746856 at full strength).
Its forced transport damaged all DEV cohorts. No promotion.
Correct v43/v44/v45/v46/v47 fitting by reserving primary and secondary
cohorts from unused train-bucket books if at least eight exist; otherwise
use disjoint 4/4/4 subsets of the inherited calibration books. No fitting
document is present in the reference index. Record full-file SHA256
and IDs; reject cross-cohort exact duplicates. This does not prove
partial-text disjointness, and inherited calibration used all 12 cal books.
The v45 continuation refuses a v44 artifact without verified corrected
fitting provenance, and requires its fitting IDs to match the primary cohort.
Local test passed on generated files: 12/12/12 spare route, 4/4/4 fallback,
and cross-cohort duplicate rejection. Existing runs pinned to older commits
are unaffected. Refit required. No checkpoint/Adam deletion or overwrite.
