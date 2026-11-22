# v167 development-promotion decision

v166 was promoted by explicit user instruction on November 22, 2026 at alpha .05 (see `experiments/nancm/ACTIVE_BASELINE.md`). v167, using the same trained v166 sidecar and optimizer history, improves the matched previously reused DEV and TEST sample in all measured strict top-k and spelling/EOS loss metrics, with exact v166 rollback and 4,545 docs-vector protection.

**Approve as the next active DEVELOPMENT baseline** after native tests and archive integrity checks. This does not imply robust statistical significance or independent PRODUCTION certification, and retains v166 .05, v158, and v148 as protected fallbacks. The separately trained margin candidate is explicitly rejected and preserved only in the research archive.