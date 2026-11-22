# NaNCM v170 strict byte accuracy follow-up

Evaluated from **actual native C++ 256-probability outputs** on every matched byte position of the v169 24-book DEV and previously reused 24-book TEST windows.

| Split / metric | v169 | v170 |
|---|---:|---:|
| DEV top-1 (8,015 bytes) | 3,912 (48.81%) | **3,963 (49.44%)** |
| DEV top-5 | 6,519 (81.33%) | **6,528 (81.45%)** |
| DEV top-16 | **7,719** | 7,718 |
| DEV BPB | 2.587376 | **2.563849** |
| Reused TEST top-1 (8,003 bytes) | **4,087 (51.07%)** | 4,073 (50.89%) |
| Reused TEST top-5 | 6,633 (82.88%) | **6,646 (83.04%)** |
| Reused TEST top-16 | 7,710 | **7,714** |
| Reused TEST BPB | 2.505834 | **2.472757** |

Bookwise paired comparison: **20/24 DEV** and **23/24 reused TEST** improve BPB. A book-level 10,000-replicate bootstrap gives exploratory 95% intervals for positive *v169 minus v170* BPB: DEV **[0.0131, 0.0346]**, reused TEST **[0.0226, 0.0447]**. These are conditional on exposed books and DEV selection; they do not certify unseen generalization.

**Important:** strict byte top-1 is *lower* on reused TEST by 14/8003, even though BPB and top-5 improve. **No strict whole-word top-1 improvement has been shown; the v167 word-development baseline is unchanged.** Top-k is next-byte, not next-token/next-word.

Full per-book metrics, exact source, binary checkpoints, rank-test output and archives in the cumulative v170 recovery ZIP. Source-only CI checks compilation and exported ABI symbols, not runtime prediction accuracy without the trained checkpoint.
