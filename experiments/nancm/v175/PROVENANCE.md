# v175 provenance

Author: Justin Howard-Stanley, Independent Researcher, Santa Cruz, CA. Date 2026-11-22.

Source: NaNROUTER Releases `corpora-short-stories-0003`, exact compressed `.tar.zst` SHA256 `15e7b5937a8ae8f5dc41f9f4b650202f9a88fb2a299daee38653c6a3d4ba8a20`, downloaded/verified by GitHub Actions #45094346173. Sampled NPZ files and split records: `results/v175_release3/` with exact SHA256SUMS. 615 inherited book IDs excluded by script. All splits before examples; no book identity shared across train/validation/test. 396/55/49 books. First TEST inspection after 280-step v175 pilot, subsequent diagnostics reused. Inherited teacher's earlier aggregate training may overlap source texts.

Parent recovered v174 archive SHA256 `06679875863433345ca9081caa5a1985e8be1df9cb967a0e53a1be8eedcbb7b2`. Inherited frozen 128+128 signed conceptual lattice, 32D lexical teacher, parent optimizer and v173 exact-CE optimizer preserved exactly. Checkpoints/v175 only added; parent checkpoints unmodified. New 4-group AdamW state19; selected v175 step4955 by validation NLL after attempted5080, checkpoint total step5755.

Source commits to master only `65d58fe772320c9d563a6fd8cf1f9fcca6a1dc4f`, `d36feab0073faba308a680bf69cee382ecf94871`. Source smoke #45095193869 PASS; full trained numerical regression local only. No other architecture or repo modified.