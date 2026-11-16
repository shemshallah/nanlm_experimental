# M3 R19 — Class-conditional spatial correction (TRAIN-gated)

R18 confirmed spatial-wave predictive information on 32 new TEST books:
+0.0035369305 BPB, 28/32 positive, descriptive whole-book bootstrap interval
[+0.00246722, +0.00455985]. Nearly all the gain came from literal spaces:
+0.0201589 BPB on byte 0x20, but −0.0005654 BPB on non-space.

R19 is an unpromoted development candidate. It does not refit or replace
R16 oscillator weights, original spatial coefficients, original TRAIN prior
or any historical checkpoint. M3 remains distinct from FA1, FA2, Terminal-RSI.

## Architecture

For each possible next byte b:
- C[b] is the frozen oscillator log-probability.
- D[b] is the frozen spatial-minus-uncoupled wave residual logit.
- logit[b] = C[b] + (s_space if b==0x20 else s_nonspace) * D[b]
- log P[b] = logit[b] - logsumexp(logit)

With scales (0,0) the inherited oscillator is exact. Scales (1,1)
reproduce the existing spatial distribution. (1,0) isolates the spatial
logit correction to the space byte; normalization still changes non-space
probability mass. The wrong-address control cyclically rotates all 256
candidate residual scores by 37 before applying the same selected scales.

## Causal, book-disjoint R19 protocol

Corpus: SHA-256 verified NaNROUTER short-stories batch 0004. Seed 160016,
4096-byte UTF-8-aligned prefixes and a 1024-byte warmup.

- Indices 0–7: original R16 TRAIN; unchanged unigram prior only.
- Indices 8–51: original R16 DEV, R17 TEST, R18 TEST. Verify their 52
  committed SHA-256 fingerprints. Never fit or evaluate these books.
- Indices 52–67: 16 fresh TRAIN calibration books. Grid screen for
  overall and non-space improvement, then shortlist at most four candidates.
- Indices 68–71: four fresh TRAIN gate books. Select among the shortlist
  subject to no overall/non-space regression on the gate. Zero remains valid.
- Indices 72–83: twelve fresh book-disjoint DEV books. Compare oscillator,
  original full wave, space-only, selected class gate and shifted-address
  control once. DEV cannot update parameters or choose a gate.

Scales drawn from (0, .25, .5, 1), giving 16 class-pair candidates.
A safe TRAIN gate does NOT guarantee DEV performance.

Report BPB, BPC, non-space BPB, top-1/top-5, throughput, whole-book
bootstrap intervals, per-book signs and wrong-address attribution.
DEV candidate status requires positive overall and non-space BPB and a
positive whole-book bootstrap lower bound. No automatic promotion;
independent TEST and capacity-matched non-geometric control still required.

Run .github/workflows/m3-r19-class-conditional-wave.yml on master.
