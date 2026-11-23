# NaNCM v179 claim ledger

## Verified / promote as opt-in research backend
- POSIX parent loads v176 model **once**, forks 1–4 independent byte and word inferencers. COW pages share model physical memory across workers. Tested on Linux native executable.
- Synthetic exact P and Q parity, cache reset, malformed requests, 256 identical speculative samples, and full multi-context word, byte, sample results.
- 48 reused-VAL distinct-book histories: per-run 240 exact 256-way Q/P vector pairs, 384 identical seeded v176 samples, 240 exact verifier scores; complete-context path compared 48 sets of top-five IDs, names, probabilities, log scores and samples.
- Full-context 4-worker parallel throughput vs serial: 3.31257x, 3.36427x, 3.4789x on 48 matched contexts; mean 3.38525x (model-open excluded; per-run averages of 2 serial vs 3 parallel full passes).
- PSS parent+4 workers initially 277,218 KiB, after 48 contexts 280,679 KiB. Compare within PSS only; v177 794 MB measurement was RSS.

## Rejected / not promoted
- One-context multi-prefix prefill did not yield a reliable speedup; per-prefix IPC overhead remains considerable.
- No whole-word ranking improvement, no new TEST BPB improvement, no compression gain, no Android field verification.
- No support claimed for in-process fork after multithreading; API fails closed if `/proc/self/task` indicates more than one task.

## Remaining milestones
- Held-out new book-disjoint TEST and fixed-run CPU/Android repeat with control for thermal throttling.
- Production process supervisor: worker health, crash restart, pipe I/O deadlines, output transactionality, cancellation, memory budgets.
- Reduce fork startup cost or build genuinely immutable shared model tables for portable thread-state execution (separate future architecture, preserve v179 as a working independent mode).
