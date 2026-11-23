# NaNCM v176 — claim and promotion ledger

ACCEPT standalone hybrid strict in-vocabulary word prediction on 55 validation books: top-1 1573→1687, top5 3596→3826, NLL 5.93931→5.81716. REUSED TEST n11760 1364→1483 top1 and 3123→3354 top5, NLL 6.00422→5.88244. Beta .5 firstletter, .35 BPE tuned only on VAL; TEST already reused.

ACCEPT 1,000 native/PyTorch normalized 12k parity and inherited exact rollback. ACCEPT inherited 256-byte full-vector sampled parity; accept exact distribution-preserving speculation law and arithmetic encoder/decoder reversibility.

DO NOT CLAIM v133 trained weights transferred; online 75% top5 is different metric/population. DO NOT CLAIM 80% accuracy, natural text BPB gain, live performance speedup, natural punctuation fidelity, untouched TEST certification, complete independent parent pretraining provenance, or promotion of byte v170/open vocabulary v167.

REJECT unconditional beam rank boost and unnormalized word/BPE score mixing. The only normalized 12k final expert is the convex gated log-mix + renormalized firstletter/BPE likelihood ratio.

Next: implement a batched block-verifier that checks multiple drafted positions in one native call and measures real walltime, and gate top5 acceptance by live word-start byte confidence.