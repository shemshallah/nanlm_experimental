# R35 — trainable local symbiosis, preserving the M3 R32 default

R34's full two-way anchor-feedback trained successfully but failed independent
DEV. Its **local-only** ablation improved BPB on ten of twelve DEV books. R35
tests that signal independently with just two trainable local-disagreement
coordinates, not four. The same R32 local lexical memory, frozen TRAIN-only
anchor, 256-way distribution and 32 optimizer histories are inherited exactly.

**Every new coefficient is initialized to zero**, so the candidate initially
returns byte-identical distributions to accepted R32. The 32→34 Adam migration
leaves the first 32 parameter groups, moments, steps and TRAIN-only lexical
anchor bit-exact. This is an *experiment*, not another production model.

CPU experiment on NaNROUTER Release `corpora-short-stories-0004`:

- FIT: books 540–555 (same FIT corpus as R34, explicitly reused).
- GATE: books 556–559 (reused previously opened GATE; exploratory only).
- **Fresh DEV**: books 596–607, not opened in R34.
- R34's sealed independent TEST 572–595 remains untouched.
- The R34 trained local-only 2-coefficient ablation is a frozen comparator,
  from its versioned, SHA-verified candidate artifact in `m3_r34/results/`.
- Training: batch size 64, Adam, 16 FIT books; no R32 or R34 weights updated.
- Report R32/R34-local/R35 BPB, nonspace BPB, top-1/top-5, per-book gains,
  whole-book bootstrap, exact model/optimizer lineage.
- No production promotion without another prelocked independent TEST and
  unrelated-history control on a separate release. `m3_default` stays R32.

Run `python -m unittest discover -s m3_r35 -p 'test_*.py' -v`; the CPU FIT and
fresh DEV are also wired to the workflow
`.github/workflows/m3-r35-cpu-train-dev.yml` on `master`.
