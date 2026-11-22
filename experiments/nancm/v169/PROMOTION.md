# v169 promotion decision — NOT promoted

**Active development remains v167**, by the existing user's promotion. Protected fallback chain: v166 conservative (unchanged full optimizer) → v164 → v158 → v148. Experimental architectures v168 and v169 reside within the same `master` lineage.

The new **native 256-byte mixed predictor** is a valuable experimental milestone: measured DEV 2.587376 BPB and previously reused TEST 2.505834 BPB, improving over 256-byte trigram alone on all 24 book windows per split. But byte mixing weights were selected on DEV, and TEST fixtures were accessed previously. It is therefore not production-certified. Keep its sidecar, native implementation and optimiser lineage within the cumulative archive.

The **strict next-word** path remains at 46/480 DEV and 141/1394 broader DEV top-1; not even a small dependable improvement over protected v167. The fitted candidate-portfolio ranker and document-local suffix cache are rejected. A newly pristine book cohort, independent contamination audit and meaningful first-choice gain must precede a word-baseline promotion. No checkpoint was overwritten or optimizer state reset.