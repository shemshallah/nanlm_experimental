# FA2 R68: single active model and continued causal training (2026-11-17)

The user explicitly promoted R68 despite the strict three-depth DEV screen
failing at 0 and 512 warm-up bytes. This is a **manual research override**,
not a claim that the statistical promotion criterion was satisfied.

## Exactly one lineage

`R58 step3185 -> R66 step3441 -> R67 rank-safe configuration -> R68 step4209`

- One `MirrorLM`, not a separate memory-only language model.
- One AdamW optimizer; the 196 inherited Adam states were retained with exact
  historical parameter ordering. Only the existing four joint-memory trust
  weights changed across R66 and R68 training.
- The inherited FA2 backbone, geometric addresses, causal key hierarchy,
  persistent book memory, successor trust rails, and space/EOS rank guard
  remain attached in the model's existing `forward` path.
- Independent TEST was not accessed. Parent checkpoints and earlier research
  archives remain recoverable.

## Active recovery and default evaluation

`FA2_ACTIVE.json` is the authoritative pointer to the 269 MB external
checkpoint and its SHA-256; the repository intentionally does not contain the
large checkpoint binary. Unpack the current FA2 recovery archive so that the
working directory contains `checkpoint/FA2_ACTIVE.pt`. The default R63 book
evaluator resolves and verifies this active checkpoint when `--checkpoint`
is omitted; explicit `--checkpoint` retains mandatory checksum validation.

Code consumers may call:

```python
from poincare_mirror.fa2_active import load_active_fa2
model, optimizer, saved = load_active_fa2(include_optimizer=True)
```

An alternate location can be supplied with `FA2_ACTIVE_CHECKPOINT` or the
loader's `path=` argument. Checksum and trained configuration mismatches fail
closed. This is a single unified load path, including the optimizer state.

## Verified independent-book DEV

Twelve distinct DEV books; 256 scored positions per book at each causal
warm-up depth; four paired counterfactual controls; whole-book bootstrap
with multiplicity correction. All BPB values are over the 257 byte/EOS target.

| Warm-up | Neutral BPB | R68 BPB | Neutral top-1 | R68 top-1 | Neutral top-5 | R68 top-5 | Strict screen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 4.976114 | 4.846222 | 24.707% | 24.642% | 49.805% | 51.497% | FAIL |
| 128 | 5.279317 | 4.945813 | 23.014% | 24.023% | 45.117% | 48.405% | PASS |
| 512 | 4.248797 | 4.037237 | 31.087% | 31.022% | 58.529% | 60.189% | FAIL |

No +2 or +3 predictive gain was measured from the joint memory rail: the
existing evaluator's +2 and +3 values are exactly the neutral values. The
next accepted architectural advance must wire the causal geometric history
to future horizons and demonstrate a corrected advantage against unrelated
history. Do not open independent TEST to tune future specialists.

## Training/recovery safeguards

- R68 used 12 TRAIN books, 9,216 distinct positions, 768 new updates.
- `train_r68_multidepth.py` verifies the R67 input SHA-256 and restores all
  historical Adam parameters in their original append order.
- `promote_r68_active.py` validates all three DEV reports and requires a
  deliberate `--user-override` before creating the active checkpoint.
- The active checkpoint checksum is
  `c377659754cb0d7e2a4583c04d816571ba60547f4e0939124613e7bbb6dd1580`.
- The user may revert the default pointer to the protected R67 or R58 archive
  without discarding the trained R68 candidate.
