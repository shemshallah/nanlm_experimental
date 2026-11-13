# Poincaré Mirror R31 — GitHub CPU continuation (one model)

This is a continuation of the *existing* 21,304,992-parameter model, not a new
model, a random restart, or an FA1/FA2 merge. The only accepted starting file
is the R30 step-2,801 checkpoint with **all 63 Adam state entries** and the
original model and random-generator states.

| Property | Verified requirement |
| --- | --- |
| Starting checkpoint | `latest.pt`, step 2,801 |
| Exact SHA-256 | `9ad6a368a9e68f4c1ccff366bee88047f7475f93ccb4d1da1c999db7bddee902` |
| Size | 255,749,067 bytes |
| Model | width 192, six layers, 16 heads, 32 signed-mirror pairs, 64 signed expert channels |
| Optimizer | AdamW with at least 63 retained state entries |
| Tokenizer | original 257-class UTF-8 byte/EOS mapping |
| Prior training | R27→R28→R30 lineage, cumulative step 2,801 |

## Bootstrapping Actions

The GitHub workflow lives at
`.github/workflows/poincare-mirror-r31-nanrouter-cpu.yml` in
`shemshallah/nanlm_experimental`. The corpus is downloaded at run time from the
verified public `shemshallah/NaNROUTER` GitHub release
`corpora-short-stories-0004`; the source repository does not carry books or
model weight binaries.

GitHub cannot read a private ChatGPT Library item directly. Before running the
manual continuation, the user must create a GitHub Release in
`shemshallah/nanlm_experimental` named
`poincare-r30-step2801-canonical` and upload the existing verified checkpoint
under the asset name **`poincare_mirror_latest.pt`**. The workflow refuses any
substitution because the first operation checks both its full SHA-256 and the
architecture, step number, Adam moments, and tokenizer.

Then run the GitHub Actions workflow with the default values (64 additional
steps, four CPU threads). Tests on source pushes can run before this release is
seeded. The release bootstrapping requirement is a *real blocker*, not a
request to reinitialize from scratch or replay R27 synthetic training.

## Honest held-out protocol

1. Restore the exact R30 weights and optimizer from the release and verify
   the SHA-256 before unpickling.
2. Download and verify NaNROUTER's batch-0004 Release archive and checksum
   manifest. Exclude the 38 Gutenberg IDs used in the historical R30 experiment
   before building the new splits.
3. Sort books by deterministic seeded content hash, deduplicate exact copies,
   and reserve separate 10% development and 10% untouched test *books*.
4. Adopt the new corpus **with zero gradient updates**. This preserves the
   original weights, optimizer history, RNG and training step.
5. Evaluate fixed development positions at 256-byte context, perform a bounded
   CPU continuation with 256/512/1,024-byte mixtures, then evaluate the exact
   same positions.
6. Eligibility requires development CE for +1 and +2 to improve by more than
   0.0001 nats/byte, +3 CE to regress less than 0.005, and accuracy to avoid a
   decline greater than 1 percentage point. Fresh test books are consulted
   *once* after development eligibility, not for selection.
7. Preserve candidate weights/metadata as GitHub Actions artifacts. If the
   candidate passes the gate, publish it as an explicitly **unpromoted**
   prerelease. The R30 starting Release is never overwritten.

This design does not claim a frontier model, guaranteed improvement, or a
completed training round merely because a workflow exists. Real CPU numbers
are reported only after the runner has restored the full checkpoint and
completed its measured experiments.

## Local verification

```bash
python -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install 'numpy>=1.26,<3'
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -q
python scripts/verify_lineage.py --checkpoint ../run/latest.pt \
  --expected-sha256 9ad6a368a9e68f4c1ccff366bee88047f7475f93ccb4d1da1c999db7bddee902 \
  --expected-step 2801 --minimum-adam-states 63
```
