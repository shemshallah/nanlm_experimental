# Layered tokenizer

`z3/layered_tokenizer.py` implements one lossless document representation at
six aligned levels:

| Level | Unit | ID space | Round-trip |
|---|---|---|---|
| bit | 0/1, MSB first | 2 | exact byte packing |
| byte | UTF-8 byte | 256 | exact |
| subword | v4 64k byte-level BPE | 64,000 | exact bytes |
| token | flat model stream + `<doc>` / `<eos>` | 64,000 | exact after stripping frame |
| word | Unicode word / punctuation / space span | tuned top-K + subword fallback | exact spans |
| sentence | sentence span, max 512 chars | structural span | exact tiling |

Every subword token has a byte span. Every word and sentence has a byte span
and an overlapping subword-token range. This is the alignment needed for
multi-level objectives, hierarchical attention, and future-token supervision.

## Defaults

- Subword: bundled v4 64k tokenizer
  (`z3/tokenizers/v4_64k/tokenizer.json`).
- Token specials: `<pad> <bos> <eos> <doc> <now> <future_1> <future_2> <future_3>`.
- Word: tuned vocabulary from `z3/tokenizers/layered_v1/layered_tokenizer.json`.
- Sentence: boundary split with a 512-character cap.
- Text training and evaluation now select the layered artifact automatically.
  `--tokenizer bin` still selects pre-tokenized uint16 bins; `--tokenizer v4`
  remains available as a subword-only ablation.

## Tuning, one layer at a time

```bash
python z3/tune_layered_tokenizer.py \
  --corpus corpora/fineweb_sample.txt \
  --stage all --max-bytes 20000000 \
  --train-subword-vocabs 32000,64000 \
  --word-vocab-sizes 16000,32000,64000,128000 \
  --sentence-max-chars 256,512,1024,2048 \
  --out runs/layered_tokenizer
```

Stages run in order and each writes an explicit gate:

1. bit: exact round-trip.
2. byte: exact UTF-8 round-trip.
3. subword: exact round-trip, byte coverage, bytes/token.
4. token: unique specials, in-vocabulary IDs, `<doc>`/`<eos>` framing.
5. word: coverage gate; selects the smallest vocabulary meeting it.
6. sentence: lossless tiling, observed length at or below the target.

Use `--stage bit|byte|subword|token|word|sentence` to retune a single layer.
The report is `tuning_report.json`; the usable artifact is
`layered_tokenizer.json`.

## Measured v1 result

Tuned on a 5,000,000-byte FineWeb-Edu sample, native subword backend:

```text
bit        exact, 8 bits/byte
byte       exact UTF-8
subword    64k, 4.6522 bytes/token, byte coverage 1.0
token      <doc>/<eos> frame exact, all IDs valid
word       16k vocab, 13,282/13,282 types covered, 6.045 bytes/unit
sentence   512 max chars, 4,218 sentences, 100% tiling
```

All six gates passed. The artifact is
`z3/tokenizers/layered_v1/layered_tokenizer.json`; the full report is beside
it.

## API

```python
from layered_tokenizer import LayeredTokenizer

tok = LayeredTokenizer()
enc = tok.encode("A sentence. Another one!")
enc.bits            # 8 bits/byte
enc.bytes           # UTF-8 bytes
enc.subword.ids     # v4 subword IDs
enc.subword.spans   # per-subword byte spans
enc.token_ids       # <doc> + subword IDs + <eos>
enc.word_units      # text, byte span, word ID, subword range
enc.sentence_spans  # byte spans that tile the text
```

The pure-Python backend is used when HuggingFace `tokenizers` is unavailable;
the native backend is used when it is installed. Both operate on the same
artifact and are covered by `z3/test_layered_tokenizer.py`.

## Next integration

The flat language model currently consumes the subword/token stream. The word
and sentence spans are exposed for the next model step: span-boundary
embeddings, hierarchical attention, and word/sentence-level auxiliary losses.
