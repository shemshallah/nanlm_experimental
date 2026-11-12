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
been uploaded to GitHub Releases.** The connected GitHub integration supports
text and Git objects, not a direct handoff of the local 103.9 MB artifact. No
placeholder or randomly initialized model is substituted.

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