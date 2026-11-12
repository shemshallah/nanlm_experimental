# FA2.1 source and checkpoint publication protocol

FA2.1 is a continuation of PAL R19. It has **one accepted checkpoint**, not
multiple independent models. PAL R18 weights and inherited optimizer moments
remain authoritative because R19 semantic candidates failed promotion.

## Reproducible known-good release

- File: `PAL_R19_SEMANTIC_FRONTIER_AUDIT_ONE_CHECKPOINT_FULL.zip`
- Compressed: 103,876,485 bytes; 239 entries; uncompressed 216,480,770 bytes.
- Parent archive SHA-256: `0761209f51c7eef54e6e0cee6606d01ede4b2231b6dd1222e481e7dc7b20eef9`
- Accepted checkpoint: `checkpoint/r12_active.pal12`
- Checkpoint SHA-256: `dd040106568786e7fbad2a880f4010635b626b51c3697e2ab08712905a2de91f`
- Model status: inherited R18 accepted. Semantic adapter candidate: rejected.

The complete archive exists in the owner's ChatGPT Library. **It has not yet
been uploaded to GitHub Releases.** The original accepted checkpoint is now
mirrored as **36 committed compressed binary parts** in `checkpoint_parts/`,
with hash-pinned `MODEL_PARTS.json` and `restore_fa2_checkpoint.py`.
[GitHub Actions run #6](https://github.com/shemshallah/nanlm_experimental/actions/runs/43308935437)
successfully restored and SHA-verified every part. The original corpus and
tokenizer are still not fully mirrored, so GitHub-hosted *training* requires the
full archive via `archive_url`. No placeholder weights are substituted.

## Use GitHub Actions to publish the exact model

1. Put the complete release ZIP at a downloadable HTTPS URL (a short-lived
   presigned URL is sufficient for a one-time workflow run).
2. Run the `FA2.1 - Verify, run CPU gates, optional one-checkpoint continuation`
   workflow using **Run workflow**. Set `archive_url` to the real ZIP URL,
   `train_steps` to `0` and `publish_archive` to `true`.
3. The workflow checks the whole ZIP SHA, accepted checkpoint SHA, unique
   checkpoint, archive safety, and source structure, then runs the inherited
   lineage and R18/R19 invariant tests before creating GitHub Release
   `fa2.1-r19`. Failures prevent publication.
4. After the initial release is uploaded, future runners may use the GitHub
   Release asset URL and can continue training through the same checkpoint.

To test a CPU continuation: run the same workflow with `train_steps` set to a
small positive number and `publish_archive` set to `false`. It uses the existing
`r18_joint_continue.py` dev and legacy regression gates; its tested candidate
is **not** automatically published as a new accepted release. The next round
must repackage, independently verify, and update the pinned SHA-256 only after
promotion is justified. Run reports are retained as short-lived Actions
artifacts, not described as permanent model checkpoints.

## Verified local archive integrity

The FA2.1 source checker was run against the actual local R19 archive and
confirmed both SHA-256 digests, 239 entries, and one accepted checkpoint.
Its path-traversal rejection self-test also passed.

The scientific next gate is held-out paraphrase and multi-hop downstream
language-model improvement, not just indexed lexical retrieval. Claims about
65K-token reasoning must use actual end-to-end document encoding.
## CPU continuation and R20 soft admission — November 12, 2026

Accepted weight SHA-256 remains `dd040106568786e7fbad2a880f4010635b626b51c3697e2ab08712905a2de91f`.
The original four inherited optimizer groups and R18 memory optimizer moments
were preserved. Both attempted conventional continuations failed promotion:

| CPU candidate | Short BPB before → after | Long BPB before → after | Accepted? |
|---|---|---|---|
| 50 updates, original LR, 14,193 targets | 2.599446 → 2.599548 | 2.565997 → 2.568984 | No |
| 100 updates, LR × 0.15, 28,585 targets | 2.599446 → 2.600335 | 2.565997 → 2.566477 | No |

See `reports/fa2_1_cpu50.json` and `reports/fa2_1_cpu100_lr015.json`.
The source continuation driver now accepts `--lr-scale`, and GitHub Actions
exposes the corresponding manual-run input without resetting Adam moments.

### R20: experimental causal, nonexact admission

`source/r20_soft_admission.py` reuses PAL's **existing trained R18 Q/K
projections**. It removes the exact two-token candidate-admission requirement
and tests retrieval over older causal keys, in blocks. Its scalar gain is
initialized at zero for exact prediction preservation. This is **not yet
a trained or promoted semantic language-model extension**.

Twelve local structural tests passed, including causal suffix invariance and
gradients through the inherited query/key layers. Unrestricted copying hurt
non-space loss. Restricting retrieved values to non-space content improved
historical long-context development BPB **2.565997 → 2.563825** and historical
book-disjoint test BPB **2.413332 → 2.411362**, using development-selected
gain 1.0. Test non-space NLL improved **4.527414 → 4.522701**.
The 32-window short-context historical test slightly regressed,
BPB **2.398048 → 2.398177**.

**Decision:** no promotion. This is a limited frozen-weight copy ablation,
not paraphrase understanding, multi-hop reasoning, or independent new-book
generalization. Historical test books have been used in previous rounds.
The R20 code and reports are committed for reproducibility; only one
original accepted checkpoint remains.

Next acceptance gate: incorporate a trainable content/relevance/novelty gate
in the *same* checkpoint, compare against exact-match and lexical baselines
on genuinely unseen paraphrase keys, two-hop chains, and independently
held-out prose, and require improvements across long and short BPB plus
non-space NLL and inherited future/word anchors. Longer than 2K memory indexing
does not by itself establish trained long-context language understanding.

## FA2.2 R21 trainable semantic-memory gate — November 12, 2026

Continued one FA2 model from the unchanged R18 accepted checkpoint. Implemented a trainable content relevance/novelty gate with 138 new parameters, reusing R18 projections, 65-token minimum causal gap, and R20 content-only candidates. A conservative 192-position warmup preserves short-context predictions exactly. The 13/13 R21 causal/identity/gradient/integrity tests passed; the inherited R18 and R19 test suites also passed.

CPU: **1,000 cumulative gate-only updates**, with actual gate Adam moments restored at 300; 24 training, 8 long development, 8 long historical test, 12 short development and 12 short historical test windows. The selected development long BPB delta against R18 was **−0.003190** and non-space NLL **−0.006849**; historical book-disjoint long BPB delta **−0.000872** and non-space NLL **−0.002807**. Short metrics were exactly unchanged, by activation threshold. Gains are modest and historical test books are reused, not pristine; R21 is not proven a semantic QA or multi-hop improvement.

**No promotion.** Original 10,756-update checkpoint SHA-256 remains `dd040106568786e7fbad2a880f4010635b626b51c3697e2ab08712905a2de91f`. A small quarantined `fa2_2_r21_gate_candidate.pth.b64` stores ONLY the new gate parameters and its own Adam state for future continuation; no independent base model was initialized. Full methods: `docs/FA2_2_R21.md`. Metrics: `reports/r21_fa2_2_cpu_candidate.json`. Promotion still requires an untouched new-book eval, genuine paraphrase/hard-negative and two-hop downstream gain, and complete inherited regression gates.

### FA2.2 R21 400-chunk follow-up

Reusing the *unchanged, frozen* 1,000-update gate at the same 400-chunk windows as R20 yielded long BPB **2.562485 dev / 2.410224 historical test** and non-space NLL **4.584142 dev / 4.519880 historical test**. Both outperform the R20 gain-1 control and accepted R18 on these historical windows. No retraining or hyperparameter selection occurred on these 400-chunk windows for R21; these are nevertheless heavily reused historical splits, not new books. The accepted SHA remains `dd040106568786e7fbad2a880f4010635b626b51c3697e2ab08712905a2de91f`. Full evidence: `reports/r21_400_horizon_probe.json`.
