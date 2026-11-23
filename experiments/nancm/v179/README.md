# NaNCM v179 — True concurrent COW verifier with exact inherited P/Q

**2026-11-23. Opt-in research; default v178/v176 inference is unchanged.**

## What changed

Unlike v177's threads (which loaded an additional full v170 byte model per worker, +794 MB RSS and 0.791x serial speed), v179 loads **one** complete trained v176/v175/v170 native model in a dedicated *single-threaded* process and forks 1–4 POSIX workers. Linux copy-on-write pages allow its immutable world-model, signed S+/S− cloud, ANN sidecars, temporal word predictor, and byte tables to be physically shared. Each child has independent mutable inference state; no shared C++ object is evaluated concurrently by different threads. `wf179_fork_open_full` refuses a multithreaded Linux process by checking `/proc/self/task`. The Python tool only **spawns a separate native executable**; it never calls fork inside Torch.

Three modes are available:

1. `wf179_fork_prefill`: compute P for top-five unique prefixes on separate workers, reconstruct **the exact inherited v176 Q law** using v177's `parallel177::fill_proposal`; fallback verifier is still v176. This is correct but still slower/near parity for **one short context**.
2. `wf179_fork_verify_contexts`: parent constructs v176 top-five candidates serially for many independent histories; each child receives a batch of prefixes per history. This raises multi-context word-verifier throughput without changing prediction accuracy.
3. **`wf179_fork_full_contexts`**: each worker independently executes inherited `wf176_hybrid_begin`, `wf176_hybrid_verify_top5`, **and** `wf176_hybrid_sample`. Returns the exact original word IDs, names, whole-word probabilities, byte-verifier log scores, and seeded sample bytes/acceptance/rejection. Parent does *no* neural inference or word candidate calculation. Completion-driven POSIX polling distributes independent contexts to available workers.

All three modes use inherited trained weights and the exact original byte law, with no approximations, quantization, retraining, or optimizer changes.

## Trained results

48 distinct books from the **reused 55-book validation cohort** (not fresh TEST), book-disjoint individual histories. One complete-context request includes a word-predictor begin, evaluation of five entire word paths, and one seeded speculative sample. Model load is excluded in the throughput ratios, but separately reported. Serial and multi-worker models share checkpoint paths; timing uses two serial and three parallel passes per run and compares their per-pass means.

| Workers | Full-context throughput relative to serial | PSS, initial parent+workers |
|---:|---:|---:|
| 1 | 0.983x | 277,053 KiB |
| 2 | 1.867x | 277,109 KiB |
| 4 | 3.313x (initial run) | 277,218 KiB |

**Four-worker replicate speedups:** 3.313x, 3.364x, 3.479x (mean 3.385x); measured CPU cgroup budget 4 cores. After the initial 48-context workload, the pool's PSS rose from 277,218 to 280,679 KiB (~3.38 MiB), despite using four independently mutable worker states. This is **PSS**, which apportions shared resident pages, not summed per-process RSS. It must not be compared as a like-for-like measure to v177's reported 794 MB incremental RSS.

In the 48-context run, **240/240 full 256-way P/Q vector comparisons**, **384/384 seeded single-stream samples**, and **240/240 complete-word log likelihoods** were exact. The *fully parallel* path additionally checked exact 5 word names/indices/probabilities and one exact seeded sample per context (48 contexts); no tolerance was needed for probability equality. Three runs reproduced the full-context speed improvement. `RESULTS.json` and `results/v179_*` contain raw per-arm numbers and methodology.

### Limitations and deployment status

* This improves **throughput across independent contexts**, not BPB, top-one accuracy, or single-stream generation speed. Single short top-five verifications remain slower or near parity.
* Fork relies on Linux/Android POSIX COW, not cross-platform threads. Launch only from a *dedicated single-threaded native process*, not after Python/BLAS/GPU threads have been spawned. This experimental code assumes the workers exit cleanly; no hardened production supervisor, timeout cancellation, or crash recovery exists.
* The same validation books have been repeatedly explored. A pristine book-disjoint TEST set and Android/Termux test are still required before any deployment promotion. This is an opt-in native execution backend, not the model default.
* Full multi-context throughput excludes one-time model/worker setup. `worker_open_seconds` / `baseline_open_seconds` in results document that cost.

## Reproduce (trained sidecars required)

```
make v179-source-smoke
make v179 v179-standalone
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v179/prepare_real179.py --n 48 --workers 1
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v179/prepare_real179.py --n 48 --workers 2
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v179/prepare_real179.py --n 48 --workers 4
```

The one-ZIP v179 cumulative archive contains trained sidecars, retained inherited optimizer states, native binaries, all prior v178 contents, regression tests, source, book windows, and provenance. GitHub remains source-only on `master`, with no checkpoint or model binaries committed.
