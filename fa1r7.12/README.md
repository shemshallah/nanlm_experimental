# FA1 — PAL R7.12 frontier-memory research

**Codename:** FA1 (formerly PAL R7.x). This directory tracks FA1 R7.12 and forward experiments, without changing the separately accepted PAL checkpoint.

## Accepted state
- Accepted model: inherited PAL R7.7 at 14,416 joint updates; five optimizer histories preserved.
- R7.11 directed-memory adapter: experimental, auxiliary update 80; NOT merged into the language model.
- R7.12 streaming gate adapter: experimental only, zero-gate default; no LM promotion.
- The full verified 240,360,600-byte R7.12 reproducibility release is retained in the ChatGPT artifact Library; the GitHub connector supports text file commits but does not expose release-asset uploads. Do **not** treat this code mirror as a complete checkpoint upload.

## Next experiment
Learn hierarchical causal context routing with matched correct, reversed, and unrelated-memory controls and a gate requiring development language-model gain plus correct-memory advantage. Tests and source under this directory will be committed as they are validated.

## Data & safety
NaNROUTER corpora must be book-disjoint, deduplicated, and split before any sealed evaluation. No outcome on historical reused books is described as a frontier-quality benchmark.
