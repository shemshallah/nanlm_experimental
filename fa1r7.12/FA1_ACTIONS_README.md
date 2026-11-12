# FA1 R7.14 — GitHub Actions CPU training and artifact policy

Repository: https://github.com/shemshallah/nanlm_experimental/tree/master/fa1r7.12

## Verified auxiliary CPU run

- Workflow: https://github.com/shemshallah/nanlm_experimental/actions/workflows/fa1-router-training.yml
- Initial run: https://github.com/shemshallah/nanlm_experimental/actions/runs/43297578390
- NaNROUTER corpus: GitHub Release `corpora-short-stories-0001`, checksum-verified.
- Whole-book-disjoint split of 160 deduplicated, long-enough books: 110 train, 23 dev, 27 test; author-disjointness is NOT established.
- 512 attempted and 512 selected CPU Adam steps; inherited router optimizer step 192 -> 704.
- 8 segments x 768 historical word tokens, supervised to find the segment containing the most recent matching key with conflicting older key/value associations.
- Dev top-1 9.57% -> 94.78%; dev CE 2.21018 -> 0.28761; post-selection test top-1 96.30%, CE 0.27374.
- Latest-occurrence exact-lookup oracle has 100% on this task. These numbers do not establish semantic retrieval, paraphrase understanding, or improved language-model prediction.
- The trained module is a *surrogate-feature* scalar router, NOT FA1's embedding-conditioned directed memory. It is never automatically promoted to the accepted language model.
- The workflow commits `actions/actions_latest.json` and `actions/router_resume.json`, preserving training metrics and small optimizer moments. It also uploads versioned run artifacts for 90 days.

## Second verified Actions continuation

- Run: https://github.com/shemshallah/nanlm_experimental/actions/runs/43298404054
- Reloaded previous router optimizer step **704**, completing another 512 accepted auxiliary updates and reaching **1,216** (including the R7.13 seed's original 192).
- Development CE 0.28761 -> **0.15023**, top-1 94.78% -> **100%**; reused historical test CE 0.27374 -> **0.14824**, top-1 96.30% -> **100%**.
- Same 160 books used for both runs; this is an increasingly optimized exact-key recency task, not a new independent held-out test or semantic generalization. Exact last-occurrence lookup is still a 100% oracle.
- The latest commit and full machine-readable results are canonical at `fa1r7.12/actions/actions_latest.json` and `fa1r7.12/actions/router_resume.json` in the `master` branch.

## Exact accepted full-model CPU continuation (not yet run)

Workflow: https://github.com/shemshallah/nanlm_experimental/actions/workflows/fa1-full-checkpoint-cpu.yml

This manual-only workflow FAILS CLOSED until the verified full recovery archive is available on GitHub as a release asset. It then:

1. Checks full R7.13 ZIP SHA-256 (`99f25e5ff604492c6c043335eddbf638415a423f43d94f5c9a426f5a283a2728`).
2. Checks inner checkpoint SHA-256 (`54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d`), exactly joint step 14,416, all five optimizer states.
3. Runs a CPU *candidate* continuation of the same inherited model, writing `fa1_candidate_unpromoted.pt` separately, not overwriting accepted weights.
4. Uploads the candidate and exploratory evaluation report as run artifacts. Historical books are NOT sealed; no automatic promotion.

To enable the full-model workflow, upload the already-verified `FA1_R7p13_HIERARCHICAL_SINGLE_CHECKPOINT_FULL.zip` from your ChatGPT Library as a GitHub Release asset tagged `fa1-accepted-r7.13`, for example using GitHub CLI on a machine holding the file:

```bash
gh release create fa1-accepted-r7.13 FA1_R7p13_HIERARCHICAL_SINGLE_CHECKPOINT_FULL.zip \
  -R shemshallah/nanlm_experimental \
  --title 'FA1 R7.13 accepted checkpoint and recovery archive' \
  --notes 'Exact parent FA1/PAL 14416 accepted checkpoint; five Adam histories. Not a new model.'
```

Then open the full-checkpoint workflow in GitHub Actions and select **Run workflow**, accepting or changing the candidate-update count. Its current pipeline is a conventional LM candidate continuation; it is not yet the jointly optimized hierarchical-memory objective and will not automatically pass a hard-negative promotion gate.

## Small ZIP / long-term asset policy

The `FA1_R7p14_SOURCE_ONLY_SLIM.zip` distribution excludes checkpoints and preprocessed corpora (the two dominant storage costs). It includes source, tokenizer files, research reports, and hash identities. The original complete R7.13 recovery ZIP stays separately in the user's ChatGPT Library until the GitHub Release copy is verified. GitHub's text source mirror does not itself contain model binaries.