# v176 research log

2026-11-23: Inspected user uploaded v133 source/metrics: online word-T5 72.9% v133 and 75.4% v126, frozen T5 53.6% v133. Confirmed no trained v133 checkpoint. Designed a conceptual signed word→BPE subword→256-byte verifier cascade with open escape and exact acceptance.

Trained 14-feature causal echo trust gate, frozen parent v175, TRAIN-only, selected step20 on 55-book VAL. Refit firstletter and BPE first-token histograms on 396 TRAIN books; learned 1600 BPE merges (1856 tokens). Evaluated full 12000-way normalized NLL, strict top1 and top5.

Tuned beta=.5 first-letter and beta=.35 BPE on VAL; reused TEST improved 1364→1483 top1 and 3123→3354 top5. Signed native crosslevel and v175 optimizer histories preserved.

Engineered topfive word trie + open byte escape and exact min(1,P/Q) acceptance, rejection residual (P−Q)+. Native five-proposal byte scoring caches common prefixes; range coder is lossless separate mode.

One runtime reset during Library staging erased v176 working directory; recovered 16 already staged source files from Library, re-extracted unchanged v175 cumulative release, and regenerated the new gate and BPE sidecars from TRAIN-only data. Reproduced full word results exactly and 1000 native 12k parity/rollback vectors.

Caveats: no parallel verifier speedup, no natural full-text BPB, reused TEST, inherited pretrained-book overlap uncertain.