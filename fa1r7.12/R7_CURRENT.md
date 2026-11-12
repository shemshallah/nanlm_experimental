# Single active FA1 R7 architecture

**Active architecture:** R7.21 unified architecture. Source of truth: `source/fa1_r721_unified.py`. CLI: `source/r7_current.py`. Work only in `master`, within `fa1r7.12/` for the R7 lineage. No additional R7 branch, no separate FA1/FA2 replacement model.

The one unpromoted R7.21 checkpoint contains the inherited R7 trunk, mirror and directed writer, calibration state, near/far gates, and five continuous optimizer histories. The accepted R7.13 checkpoint remains untouched at step 14,416 until independently validated whole-book testing justifies promotion. R7.14–R7.20 files are historical test/reference implementations, not competing active architectures.

- Unified checkpoint SHA-256: `7d2ff0c17430075e3fc6ddeb02f61bf447c07eed027ae77afac0506737734b1f`
- Accepted ancestor SHA-256: `54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d`
- Full candidate checkpoint: stored separately from the source ZIP; never commit the 94.8 MB binary to Git source tree
- Checkpoint optimizer histories: 5, with near/far moments grafted into the existing fifth optimizer
- R7.21 joint training pilot: 96 steps executed, **none selected**; weights in unified checkpoint are the exact migrated R7.20 weights
- New-book test gate: blocked pending corpus and runner-accessible trusted checkpoint; no model promotion

See [`FA1_R7p21_UNIFIED_ARCHITECTURE.md`](FA1_R7p21_UNIFIED_ARCHITECTURE.md) for parity, CPU trial and recovery instructions.