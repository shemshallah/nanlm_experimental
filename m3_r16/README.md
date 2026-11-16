# M3 R16 — Counterfactual Wave Specialist

This is a continuation of the independent NaN16 Mirror-Wave Memory Predictor,
not FA1, FA2, Terminal-RSI, or the NaNLang VM.

## R15 evidence and problem

On four held-out NaNROUTER books (12,286 DEV byte predictions), R15's
trained 5-feature oscillator achieved 3.792282 BPB versus 3.819603 BPB for
exact historical memory. R15's full spatial and reciprocal waves measured
3.815327 and 3.815181 BPB respectively.  The wave engine's physical tests
passed, but full propagation diluted the useful temporal signal.

## R16 solution

Retain the R15 **oscillator prediction as the carrier**. For each spatial
specialist, form five genuinely geometric counterfactual features:

```
  geometric_features = wave_features - oscillator_features
  log P_candidate(b) = log P_oscillator(b)
                         + selected_scale * learned_geo_weights · geometric_features[b]
                         - log Z
```

The spatial and reciprocal specialists use the same wave clock, pulse,
damping, on-site frequency, and exact-memory prior as the oscillator.
They differ only in the graph Laplacian or head–pair coupling.  This makes
the geometric component directly measurable against the matched no-graph
control, rather than substituting the whole oscillator distribution.

### Leakage-proof gate

1. Split TRAIN books into an inner-fit group and two disjoint gate books.
2. Derive the inner prior and fit oscillator + geometric weights on inner-fit.
3. Evaluate fixed scale candidates 0, .25, .5, 1 on the two gate books.
4. Freeze the winning scale, breaking ties toward zero.
5. Refit both weight sets on all TRAIN books only.
6. Evaluate completely separate DEV books once. No TEST books consulted.

Scale zero returns exactly the trained oscillator readout. It is the
function-preserving fallback, not a claim of achieved geometric gain.

## Benchmark and promotion

The GitHub workflow uses NaNROUTER short stories batch 0004, verifies its
SHA-256, then uses eight TRAIN and six DEV books with a new deterministic
book split, 4,096-byte windows and 1,024-byte warmup. It reports memory,
oscillator, oscillator-plus-spatial, and oscillator-plus-reciprocal BPB,
BPC, top-1/top-5, non-space BPB, and throughput. For geometric candidates,
it also reports per-book gains and descriptive whole-book bootstrap intervals.

```bash
python3 -m unittest discover -s m3_r16 -p 'test_*.py' -v
python3 -m m3_r16.counterfactual_wave_gate \
  --corpus-root /path/to/NaNROUTER/short-stories \
  --train-books 8 --dev-books 6 --bytes-per-book 4096 \
  --warmup 1024 --seed 160016 \
  --output m3_r16/results/R16_NaNROUTER_DEV.json
```

Promotion remains blocked until a geometric specialist beats its inherited
oscillator on independent book-disjoint tests and longer-context gates;
the historical R12 checkpoint is not recovered.  Small exploratory DEV
improvements remain candidates, not frontier claims.
