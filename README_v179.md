# NaNCM v179 — one-model COW parallel full-context verification

Latest opt-in experiment: four native Linux workers share one pre-fork trained model's immutable memory via COW; each maintains its own causal state, exact v176 word/byte probabilities and exact speculative sampling.

Three 48-distinct-VAL-book runs reached 3.313x, 3.364x and 3.479x (mean 3.385x) whole-context CPU throughput against the unchanged serial v176 reference, including word prediction, byte verification of top-five complete words, and one seeded sample. Zero observed output discrepancies across checks. Pool PSS after 48 histories grew only ~3.4 MiB over post-fork baseline.

No fresh TEST or Android claim, no improved BPB or language quality, no single-stream speedup; default remains protected. See `experiments/nancm/v179/README.md`, `RESULTS.json`, `CLAIM_LEDGER.md`, and the cumulative reproducible ZIP.
