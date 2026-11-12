# z2 — 30-minute Colab mixed-corpus result

Kernel: Colab notebook `z2/z2_colab.ipynb`
Corpus: `corpora-paragraphs-0010` + `corpora-short-stories-0004` +
`corpora-sentences-0010` + `hw5-gutenberg-c256-g1`, mixed to 200MB.

- BPE 32k, 46,564,000 tokens total
- 84,859,523-param v2
- resumed from z1 `best.pt`
- 13,936 steps in 1,812 s
- ctx 512, batch 8, LR 1e-4, time-cosine to zero, EMA 0.999

| Checkpoint | Bits/token | Top-1 | Top-5 |
|---|---:|---:|---:|
| Baseline, z1 on mixed val | 9.9271 | 9.57% | 21.88% |
| Best, step 13,000 | **9.5700** | **11.91%** | 23.44% |
| Final, step 13,936 | 9.5742 | 10.94% | 23.63% |
| Final EMA | 9.5756 | 11.33% | **24.02%** |

Improvement over baseline: **-0.357 bits/token**, **+2.34 pp best top-1**.
Final is within 0.004 bits of best, so the continuation is stable.

Artifacts remained in Drive:

```text
/content/drive/MyDrive/nanlmv3_z2/best.pt
/content/drive/MyDrive/nanlmv3_z2/final.pt
/content/drive/MyDrive/nanlmv3_z2/ema.pt
/content/drive/MyDrive/nanlmv3_z2/result.json
```

Geometry state:

- centroid movement: 0.1478
- curvature: -1.0097
- branch temperature: 0.9442
- memory alpha: 0.2136
