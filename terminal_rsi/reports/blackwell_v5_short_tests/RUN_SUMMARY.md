# Terminal-RSI v5 short-test receipt

Device: NVIDIA RTX PRO 6000 Blackwell Server Edition (94.97 GiB)
Parent: verified v103 checkpoint SHA256 `d39d27ba...`, 2 BPB class
All runs are short probes; no long training run.

## 60 s full-v5 smoke

- config: batch 2048 / microbatch 1024, 3 parallel replicas, 2 candidates,
  shard 128, min_gain 0.001, unit loss 0.1, distill 0.5, MoE 0.01
- updates: 5
- updates/s: 0.0730
- peak allocated / reserved: **72.78 / 93.89 GiB**
- DEV 32-position: 2.2208 -> 2.2335 BPB
- hierarchy gates after 5 updates:
  - bottom-up `[0.00137, 0.00033, 0.00128]`
  - top-down `[0.00128, 0.00002, -0.00003]`
  - lateral `[-0.00205, 0.00122, 0.00206]`
- boundary BCE 0.592
- fast cached generation from `"The "`: **"social\nmit chair"**
- latent word boundaries 3/16

## 45 s hierarchical + unit-head eval

- hierarchical DEV, 128 positions: **1.8081 BPB**
- byte top1 **60.94%**, top5 **88.28%**
- token top1 37.5%, top5 70.8%, exact top5 79.2%
- word top1 11.5%, top5 26.9%, exact top5 42.3%
- unit-head DEV (real 4k-way classifiers, one book, 128 positions):
  - token top1 1.56%, top5 9.38%
  - word top1 1.56%, top5 1.56%
  - sentence top1/top5 0%
- The unit heads are wired and trained, but 5 updates is nowhere near enough
  for 4k-way classifiers. These numbers are not evidence against the heads.

## Compile probe

- `torch.compile` on the 100M upgrade branch: **succeeded**
- peak allocated / reserved: **66.95 / 67.34 GiB** (reserved drops by ~26 GiB
  versus eager because of the compiled allocation pattern)
- updates: 3 in 45 s, 0.058/s; no speedup at this short horizon, but the
  memory reduction is real.
- The first attempt failed on `or self.upgrades` truthiness; fixed with an
  explicit `is None` check.

## Ablation probe (4 updates each)

| variant | status | after BPB | byte bits | byte top5 |
|---|---:|---:|---:|---:|
| full | 0 | 2.12917 | 1.94247 | 0.8594 |
| no_loops | 0 | 2.12916 | 1.94246 | 0.8594 |
| no_pair | 0 | 2.12919 | 1.94248 | 0.8594 |

All three are identical within 2e-5 BPB. The hierarchy gates are ~10^-3 after
four updates, so the loops have not opened enough to affect the parent-driven
output. This is a **step-budget** result, not evidence that the loops help or
hurt. A meaningful ablation needs the gates at ~0.01-0.1, either after a longer
distillation run or with a gate-warmup schedule.

## Tests

- 11/11 v5 CPU tests pass, including per-sample full-parent distillation
  normalization and KV-cache equivalence against a one-pass full sequence.
- 10/10 Blackwell preflight tests, 13/13 RSI-v4 tests, hierarchy tests were
  passing earlier in the session.
