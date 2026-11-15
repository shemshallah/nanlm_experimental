# NaNCM architecture (for future Claude)

One file: `nancm.py`. It embeds the C core (`CSRC`), compiles it with gcc on first use (cached in `--work`), and holds everything else in Python. The model stays **bytewise**; token, subword and word knowledge enters only as extra inputs or as metrics.

## Layout of nancm.py
1. **C core** (bitwise PAQ/lpaq-style CM): 19 hashed contexts (orders 1-8, word, previous words, BPE-subword contexts from `seg()` with leading-space handling, sparse, case-folded, class, indirect), 2 match models, bit-history states + state maps, 6 first-layer mixers (+3 optional `-mx`), 3 LM-selected mixers, 2 second-layer mixers, 5 APMs, learned final stage (`-fin 1`).
2. **In-process token->byte LM engine** (`LME` in C, same math as `Byte_LM` in Python). Next-byte distribution = sum over all tokenizations consistent with the bytes so far; hypotheses are (last token, trie node), merged when equal, pruned to a beam (`-lmk`, default 8). Two instances: `lm_sub.bin` (BPE-subword bigram) and `lm_word.bin` (whole-word bigram, OOV words spelled as single-byte tokens). Any vocabulary of byte strings works, so subword and word are the same code. No sidecar files, so every training byte gets LM inputs.
3. **External LM slot** (slot 2): `sidecar` command + `-lmtr/-lmtrstart/-lmev`, for GPT-2-style HF models. **Untested here (no weights, no network).**
4. **Guess contexts** (`-gc 1`): the LMs' top-1 byte and confidence bucket become CM contexts (h[19], h[20]). Measured, not kept by default (see below).
5. **Metrics**: byte bpc, top1/top5; plus *spelled-unit* accuracy for token / subword (continuation pieces) / word: a unit counts as guessed at top-k only if every one of its bytes is within the model's top-k given the true prefix. This is a teacher-forced spelling metric, not a free-running next-token ranking.
6. **Python**: `prep` (corpus split by file hash, BPE training, LM tables, eval windows + unit flags), `run`, `ablate`, `selftest`, `sidecar`.

## Commands
```
python3 nancm.py prep --work W [--corpus PATHS] [--wordlists PATHS] --maxb 4000000
python3 nancm.py selftest --work W
python3 nancm.py run --work W --maxb 4000000
python3 nancm.py ablate --work W --maxb 1000000
```
`--wordlists` primes the training stream (one word per line, first) and is **untested with the real NaNROUTER lists**. Eval is on held-out *files*, and LM tables are built from the first `--lmfrac` (0.6) of the stream so the mixer meets an honest LM on the rest.

## Verified
- Python converter == brute force on a toy vocab: 3e-16 (unpruned).
- C engine vs Python reference on real vocabularies: 6e-6 (subword), 5e-4 (word) max difference; the gap shrinks as the Python beam grows (6.5e-3 at K=64 down to 5e-4 at K=400), so it is pruning, not a bug.
- Beam error vs K=128 on real vocabs: subword K=4 KL 1.6e-4, K>=8 zero; word K=2 KL 0.05, K=4 max dp 0.004, K>=8 zero. Beam 8 is safe on these vocabularies (bigram state only; an n-gram or neural LM has more distinct states and needs re-measuring).

## Ablations (1 MB train, 60 held-out windows, ~27k scored bytes; differences under ~0.005 are noise)
| config | frozen | online |
|---|---|---|
| no LM | 1.9322 | 1.5738 |
| + subword LM | 1.9087 | 1.5658 |
| + subword + word LM | 1.8852 | 1.5541 |
| + guess contexts | 1.8942 | 1.5514 |
| + guess, 3 extra mixers | 1.8933 | 1.5494 |
| + guess, `-fin 0` | 1.8933 | 1.5535 |
| + guess, `-sm 3` | 1.9046 | 1.5499 |

Kept: subword LM, word LM, `-fin 1`, `-sm 1`, `-mx 0`. Dropped: guess contexts (frozen worse, online gain within noise), extra mixers (cost, no gain). Also fixed: second-layer mixer initial weights now cover the active mixers (before, the LM mixer started at zero weight).

## Caveats
- Corpus is system docs (perl pod, TeX docs, Python docstrings) because the sandbox had no network: numbers are NOT comparable to earlier tinyshakespeare/wt2 results.
- Both LMs are bigrams; gains from a stronger LM are untested.
- Unit metrics are over few units (see `n` in UNITS lines); treat them as indicative.

## Final run (single-file nancm.py, 4 MB train, ~2 min CPU, same 60 windows)
frozen 1.7279 bpc (top1 0.667, top5 0.889); online 1.3794 bpc (top1 0.735, top5 0.917).
Spelled-unit accuracy (online, top1/top5): token 0.412/0.766, subword 0.560/0.883, word 0.272/0.628.
Going from 1 MB to 4 MB of training moved online from 1.5541 to 1.3794, mostly from more CM data. I did not run a no-LM baseline at 4 MB, so the LM's share of that gain at 4 MB is unmeasured.

## Wrong-byte diagnosis (v10.1)
`FIRSTMISS` lines / `firstmiss_*` JSON give, for each missed unit, the index of its first non-top-1 byte (4 MB model, online):
- word units (leading space is byte 0): idx0 885, idx1 1897, idx2 158, idx3 52, idx4+ ~30 -> 92% of missed words fail at the boundary byte or first letter.
- subword continuation pieces: 84% fail at their first byte. Tokens: 55% idx0, 96% by idx1.
Conclusion: errors are about *what comes next* (word-start / first-piece choice), not consistency inside a word.
Tried: `-bm 1` (mixer selected by word position + both LMs' confidence). 1 MB: frozen 1.8852 -> 1.8772, online 1.5541 -> 1.5519, first-miss histogram unchanged. Not the fix; left off by default (`--bm 1` to enable).
Next levers (untested): a stronger word-level predictor at boundaries (trigram or neural LM via the sidecar slot), real wordlists/priming.

## v10.2 (this round): word-boundary / structure contexts (`-xc MASK`, default 127)
Where the bits go (4 MB model, online, 28.8k scored bytes): first letter of a word 31% of all bits (3.43 bits/byte, top1 36%), 2nd letter 14%, punctuation 16%, mid-word lowercase 19%, spaces 6.5%. So the lever is *what comes next at a boundary*, plus line structure of the docs.
New hashed contexts, bit i of `-xc` enables context 21+i (NCTX is now 32; `nhit` still counts only contexts <21):
0 word 4-gram-ish (prefix+pw1,pw2,pw3) | 1 skip (prefix+pw2) | 2 above (bytes at the same column of the previous line, with c1) | 3 line-head hash + prefix | 4 punctuation/char-before-word | 5 words-in-line count | 6 first letters of last 3 words
7 word 5-gram (pw1..pw4) | 8 above (3 bytes) + c1,c2 | 9 column + above byte | 10 previous line head + current line head
Leave-one-out at 2 MB train (all 7 = frozen 1.8407 / online 1.4774; control without them 1.8888 / 1.5272): removing trigram +0.014 online, above +0.011/+0.024 frozen, skip +0.005, wic +0.005, linehead +0.003, punct +0.002, pfirst +0.0015.
Batch 2 added one at a time on top of the 7: 5-gram 1.8371/1.4679, above-3 1.8358/1.4722, col+above 1.8309/1.4714, prevline-head 1.8397/1.4756. Default stays 127 until the joint run is verified; try `-xc 2047` (use NANCM_FLAGS="-xc 2047" python3 nancm.py run ...).
Dev note: env NANCM_FLAGS appends raw flags to the C core call.
