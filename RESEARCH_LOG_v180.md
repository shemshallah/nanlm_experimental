# Research log v180 · 2026-11-23

1. Target redirected from CPU throughput to strict exact next-word accuracy, with 50% as *goal* rather than result.
2. Audited v175 corpus 396/55/49 TRAIN/VAL/TEST books, fixed 12k target vocab; v176 baseline 1687/13200 top-1, 3826/13200 top-5 and 5.817160 word NLL on reused VAL. No new TEST inspection.
3. Reconstructed 3.11568M observed TRAIN positions from 95,040 overlapping windows without collision, producing 2.650489M valid causal transitions.
4. Compiled exact 14-bit packed suffix tables of orders 1–4; 19,200 TRAIN-book holdout and 13,200 reused VAL evaluations showed ~15.2% top-1 standalone memory but 25% 8-candidate recall.
5. Collected full frozen v176/BPE signed cloud 12k word probabilities on TRAIN calibration and reused VAL; top64+8 oracle upper bound 63.41% on reused VAL, revealing recall as bottleneck.
6. Designed mathematically normalized sparse residual P180 with Dirichlet-like context counts; trained gamma on TRAIN-only NLL, frozen at [.1,.1,.1,.1], then evaluated previously inspected VAL. This achieved +206 strict top-1, +335 top-5, -0.10347 NLL.
7. Tested train-top1 calibration; worse on VAL (1838) and rejected. Tested feature-engineered LGBM neural+memory ranker; worse on VAL (1792) and rejected. No model checkpoint or optimizer moments reset.
8. Added leak-free prequential 48+ historical book memory; it improves upper bound but isn't fused. Native v180 C++ integrated under opt-in API, source smoke PASS, real 180-context full vector parity error 3.3e-8.
9. Completed full 13,200-context native word parity: top-1 1,893; max 12k-vector deviation 9.54e-8; fitted paired 55-book confidence diagnostics. Prepared source-only CI, cumulative recovery and master commit. No 50% claim.
