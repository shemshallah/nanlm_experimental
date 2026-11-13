# FA1 R7 — one active architecture on `master`

The canonical current architecture is **R7.22**. All new work remains in `fa1r7.12/` on the repository's existing `master` branch. Earlier R7 rounds are historical source/test references, not competing active models. FA2 and NaNLMind are separate projects and must not be mixed into this lineage.

- One self-contained *learned-state* R7.22 candidate checkpoint, SHA-256 `eb0b7f0505f027095af23bc85ef00f2c036f3d2d5fb160ee0c45e60fb02e3455`; full recovery still needs tokenizer/vocabulary metadata.
- Accepted R7.13 ancestry remains untouched at **step 14,416**, SHA-256 `54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d`.
- R7.21 source ancestor SHA-256 `7d2ff0c17430075e3fc6ddeb02f61bf447c07eed027ae77afac0506737734b1f`.
- The 105 tensors are inside **one** PyTorch model, with the original five continuous optimizer histories; new semantic gates join the existing fifth AdamW optimizer.
- Inherited near 8K/32K, packed exact far 32K–128K, and two new embedding-associative readers (32K–128K and 128K–1,048,576-token bands) coexist within this same architecture.
- The two new gates are **exactly off**, and inactive indices are skipped on ordinary inference. `--probe-memory` explicitly measures them without changing learned probabilities.
- R7.22 ran **48 semantic-gate training updates; none selected**. The accepted checkpoint and five inherited optimizer states remain intact.
- One million tokens is *addressable retrieval*, **not** proven 1M reasoning, attention, fluency, or independent-book generalization.
- Independent whole-book loss, contamination screening, causal controls and book-cluster confidence intervals are required for a capability promotion. None are available yet for the new reader.

Canonical source: `source/fa1_r722_unified.py`, `source/fa1_r722_semantic.py`, and `source/r7_current.py`. Read `FA1_R7p22_UNIFIED_1M_ASSOCIATIVE.md` for evidence and limitations. The checkpoint is separate from source ZIPs and GitHub source control.