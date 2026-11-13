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

## Unified symbiotic Diamond, source-ready (not an accepted weight update)

The original R30 weights contain 16 physical attention heads and 32 trained
signed-mirror expert pairs. **They do not already contain 256 trained physical
attention heads.** The optional `symbiotic_rank` extension adds 256 *conceptual*
point addresses and 256 learnable Overmind apex slots **inside the same MoE
residual backbone**. These are not independent models or new optimizers.

For each token and active mirror block, one shared point router selects eight
of the 256 points. Their canonical orientation gives precisely 28 S+/S− pairs,
56 learned-value triangles with signed 2D curvature, and 70 tetrahedra with
signed 3D volume. These are sparse addresses in the full spaces of
`C(256,2)=32,640`, `C(256,3)=2,763,520`, and
`C(256,4)=174,792,640`; the full tensors are **not** materialized. Existing
expert sign probabilities determine the S+/S− mixture. Four pooled levels feed
two shared apex attention/feedback passes; downward attention reads back from
triangle → pair → point, and its projected result returns to the original
residual stream used by causal attention, the decoder and +2/+3 objectives.

The apex uses a vectorized causal prefix summary of the *current document-safe
window*. This is genuine within-window historical access, but **not** yet a
cross-window persistent memory, a proven reasoning system, or 256 physical
GPU attention heads. A targeted ablation, `no_symbiotic_geometry`, measures
the same trained checkpoint with the geometric residual switched off.

The geometric output projection is initialized **exactly to zero** after model
initialization. `grow --symbiotic-rank 32` cold-migrates a verified R30
checkpoint in place, checks +1/+2/+3 logit drift, maps all old Adam moments by
parameter name, and retains training step, RNG and corpus provenance. The old
checkpoint Release remains immutable, and migration/test reports go into
GitHub Actions artifacts. The default workflow input enables rank 32; use
rank 0 for a non-geometric R31 control.

## Enforced independent-book and long-context gate

The workflow now evaluates the old and new checkpoints on the **same**
deterministic 256-token validation positions. The original, post-adoption
step-2801 baseline is separately retained on the ephemeral Actions runner.
At fixed random seed and book hashes, the same 1,024 validation target bytes
are compared at 256 vs 4,096 input tokens, both before and after training.
Only once the development gate passes does the workflow open untouched
test books and score old and new weights on matching test positions.

`scripts/compare_r31_independent.py` requires held-out next-byte CE
improvement, +2/+3 non-regression within 0.005 nats, at most one percentage
point accuracy regression, long-context BPB no worse by more than 0.005, and
no newly enlarged long-minus-short penalty. Data hashes, window seeds,
checkpoint steps and token counts must match. Passing produces **manual
promotion eligibility**, not automatic promotion. The stronger frontier
diagnostic requires 4,096 context to outperform 256 on the same target bytes.

Until the exact 255,749,067-byte R30 release asset exists in GitHub, source
tests can run but the Action cannot recover/train the canonical checkpoint.
Do not start from a freshly initialized model to circumvent that constraint.

## Local verification

```bash
python -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install 'numpy>=1.26,<3'
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -q
python scripts/verify_lineage.py --checkpoint ../run/latest.pt \
  --expected-sha256 9ad6a368a9e68f4c1ccff366bee88047f7475f93ccb4d1da1c999db7bddee902 \
  --expected-step 2801 --minimum-adam-states 63
```
