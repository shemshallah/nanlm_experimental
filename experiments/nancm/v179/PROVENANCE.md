# NaNCM v179 provenance

Base: v178 deterministic recovery ZIP (989 entries), SHA-256 `d7f67c4a17a96fc0f7d0a196971f80c3fa9da6713a60fe2f888665ea4022deeb`, inherited v177 SHA `501a360d31be5e63f9da7e480432f640cbbae9e6b5e3b9e0cee39bc76ced5572`.

Original native inference files were not changed. New `experiments/nancm/v179` fork worker, stand-alone benchmark runner, source-only test, and paired benchmark output. All older checkpoints, Adam histories, logs, tests, and full ancestral documents remain in the ZIP. Original v178 root Makefile preserved as `ancestry/v178/Makefile` for exact rollback. Root Makefile in v179 adds **one opt-in include**, not a change to default training/inference targets.

Corpus: retained 55-book reused validation split from v175's recovery data; 48 independent-book histories chosen deterministically by NumPy RNG seed 179. Source `prepare_real179.py`, exact `results/v179_book_contexts.bin` and per-run JSON/log files are archived. No TEST books were used in score tuning, and no fresh-test improvement is claimed.

Benchmarks use a dedicated C++ process with 4-core cgroup CPU allowance, 4-GiB memory cgroup, 5 visible CPUs, one inherited full model loaded before fork. PSS read from Linux smaps_rollup, summed over parent and children. Model open costs and replay speed are reported separately. Complete full-context timing compares per-pass means (2 serial vs 3 parallel passes), includes v176 word begin + five-word verifier + one speculative sample. Inherited P and Q 256-way vector values compare by exact float bytes.
