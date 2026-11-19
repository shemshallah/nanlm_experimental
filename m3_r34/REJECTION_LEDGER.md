# R34 two-way feedback — rejected after independent DEV

The fully trained 36-group candidate and its machine-readable report are
preserved in `results/` for recovery. The 32 inherited R32 Adam groups were
unchanged; 768 CPU updates trained four new groups on NaNROUTER Release
`corpora-short-stories-0004` FIT books 540–555. R32 remains the accepted
production default.

| 12-book DEV (560–571; 36,864 bytes) | BPB | nonspace BPB | top-1 | top-5 | improved books |
| --- | ---: | ---: | ---: | ---: | ---: |
| certified R32 | 2.983009 | 3.940129 | 47.7946% | 75.9521% | — |
| R34 two-way | 2.981746 | 3.941845 | 47.8706% | 76.0281% | 7/12 |
| R34 anchor-only | 2.984924 | 3.946186 | 47.8461% | 76.0227% | 4/12 |
| R34 local-only | **2.979823** | **3.935792** | 47.8163% | 75.9576% | **10/12** |

R34 passed the earlier four-book GATE, but **failed three DEV conditions**:
nonspace regression, insufficient book wins, and two-way feedback failing to
outperform its local-only ablation. It was not promoted. The 24 R34 TEST books
572–595 were never read. Local-only R34 is an ablation result, not an accepted
checkpoint; future experiments must train fresh candidates against R32.

Recovery and provenance:

- GitHub Actions run: https://github.com/shemshallah/nanlm_experimental/actions/runs/44412961045
- Artifact ID: `14477782692`.
- `results/R34_36_group_candidate.npz` SHA-256: `0d63bff7ce91d6444d1c2eaf4de15021f981401ba210e0208e08e5ac43bf8b5c`
- `results/R34_FIT_GATE_DEV.json` SHA-256: `93c0fd544d5e9d3a55044f747d2b17e7cef3111ff9b12be130ad8f98dc17ab2e`
- Accepted R32 parent SHA-256: `67e7417fb38c8d914714cee243fb080da6e63cf5f9e9c9737e28a5b168c1da1a`

The sealed independent R32 TEST result is separate; the R34 DEV values here
must not be compared to its different 24-book TEST baseline.
