# FA1 R7.23 CPU continuation status (no new model)

FA1 remains **one R7.22 unified checkpoint**, descended from accepted R7.13
step **14,416**, with **five** optimizer histories. R7.23 denotes the CPU
training/data integration, not a second architecture or training branch.

## Verified local CPU pilot (historical splits only)

- Input: exact R7.22 checkpoint SHA-256 `eb0b7f0505f027095af23bc85ef00f2c036f3d2d5fb160ee0c45e60fb02e3455`.
- 512 train / 256 dev examples, 1,024 optimizer steps, short semantic gate only.
- Baseline dev NLL **5.170198440551758** nats/word; zero-gated final NLL identical.
- Dev short-semantic candidates available **224/256**; retrieved correct target **100/256**.
- The gate closed, **zero selected updates**, no saved trained candidate, no promotion.
- The source R7.22 checkpoint is unchanged. The 128K–1M long semantic gate was not trained.
- These are *historical* splits, NOT independent NaNROUTER whole-book metrics.
- Stop criterion triggered; simply continuing this gate-only optimization is not valuable.

## GitHub NaNROUTER readiness

GitHub Actions workflow `.github/workflows/fa1-r723-nanrouter-continuation.yml`
completed its preflight in run
https://github.com/shemshallah/nanlm_experimental/actions/runs/43338569192.

- Downloaded NaNROUTER release `corpora-short-stories-0001`.
- SHA-256 verified **110,044,213 compressed bytes**:
  `bceb6ef83ffa89ccb257bf55d9a5ad0e56a70d2b7bfd54599bacd26dd5beb2a5`.
- Remote training job **skipped** (not executed): GitHub Releases currently do not
  contain the *exact* R7.22 checkpoint and the ancestor metadata bundle.
- The source now accepts separate document-disjoint corpora without crossing
  source-book boundaries. Synthetic multi-book local integration smoke test passed.
- When seed assets exist, workflow runs staged 256/512/1024/2048/4096 updates,
  stopping as soon as development NLL or causal memory controls fail.
  No auto-promotion; outputs remain unpromoted Action artifacts.

## Required one-time GitHub Release seed

Create GitHub Release tag `fa1-r722-cpu-seed` in this repository with these
two **exactly named** assets:

1. `FA1_R7p22_UNIFIED_CANDIDATE.pt` (94,809,113 bytes), SHA-256
   `eb0b7f0505f027095af23bc85ef00f2c036f3d2d5fb160ee0c45e60fb02e3455`.
2. `FA1_R7p13_HIERARCHICAL_SINGLE_CHECKPOINT_FULL.zip` (214,538,430 bytes), SHA-256
   `99f25e5ff604492c6c043335eddbf638415a423f43d94f5c9a426f5a283a2728`.

These exist in ChatGPT Library, but were not available to the GitHub runner.
Do **not** replace them with FA2 or random checkpoint tensors. Release binaries
rather than committing them into source control.

After uploading, run the R7.23 workflow via Actions -> Run workflow on master,
or `gh workflow run fa1-r723-nanrouter-continuation.yml -R shemshallah/nanlm_experimental`.

## Scientific gates

Dev selection requires positive NLL gain and positive advantages versus
unconditional old-frequency and independent train-donor controls. Disjoint
books are split by content hash before tokenization. Historical test books
are not used for selection. A sealed independent-book test and byte-accurate
loss measurement are **still required** before any capability promotion.
