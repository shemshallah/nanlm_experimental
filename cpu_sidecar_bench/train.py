#!/usr/bin/env python3
"""CPU benchmark for the causal sidecar builder.

Measures SpanSidecar.from_token_ids and its components on Kaggle CPU so we
can tell whether the ~1.7 s/step T4 loop is GPU-bound or CPU-sidecar-bound.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path("/kaggle/working/nanlm_experimental")
TOK = REPO / "z3" / "tokenizers" / "layered_v1" / "layered_tokenizer.json"

if not (REPO / "z3" / "train_v3.py").is_file():
    subprocess.run([
        "git", "clone", "--depth", "1",
        "https://github.com/shemshallah/nanlm_experimental.git", str(REPO),
    ], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tokenizers"],
               check=True)

sys.path.insert(0, str(REPO / "z3"))
from layered_tokenizer import (  # noqa: E402
    MathSpan,
    SubwordEncoding,
    _char_byte_offsets,
    load_artifact,
)
from concept_loop import SpanSidecar  # noqa: E402


def ms(fn, n: int = 5) -> float:
    fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


tok = load_artifact(TOK)
text = ""
for name in ("README.md", "z3/CAUSAL_GPU12H_2026_11_22.md",
             "z3/LR_REWARM_FINDING_2026_11_22.md"):
    path = REPO / name
    if path.is_file():
        text += path.read_text() + "\n"
if len(text) < 8000:
    text = (text + "\n") * (8000 // max(1, len(text)) + 1)
ids = tok.encode(text, add_document_frame=False).token_ids
while len(ids) < 48 * 512:
    text = text + "\n" + text
    ids = tok.encode(text, add_document_frame=False).token_ids
windows = [ids[i:i + 512] for i in range(0, 40 * 512, 512)]
print("tokens", len(ids), "windows", len(windows), flush=True)

for w in windows[:2]:
    SpanSidecar.from_token_ids(w, tok)

batches = [windows[i:i + 8] for i in range(0, 40, 8)]
full = []
for batch in batches:
    t0 = time.perf_counter()
    rows = [SpanSidecar.from_token_ids(w, tok) for w in batch]
    full.append((time.perf_counter() - t0) * 1000.0)
    SpanSidecar.collate(rows)
print("FULL_BATCH_MS", [round(x, 1) for x in full],
      "median", round(statistics.median(full), 1), flush=True)

w = windows[0]
n = len(w)
text_one = tok.subword.decode(w)
raw_bytes = [tok.subword.token_bytes[i] for i in w]
offsets = _char_byte_offsets(text_one)
spans = []
cursor = 0
for raw in raw_bytes:
    spans.append((cursor, cursor + len(raw)))
    cursor += len(raw)
words = tok.word.encode_with_offsets(text_one, offsets, spans, token_offset=0)
sentences = [(offsets[s], offsets[e]) for s, e in tok.sentence.split(text_one)]
math_spans = [
    MathSpan(span.text, offsets[span.start], offsets[span.end], span.kind)
    for span in tok.math.spans(text_one)
]
tokens = [tok.subword.id_to_token.get(i, "") for i in w]
encoding = SimpleNamespace(
    subword=SubwordEncoding(
        text=text_one, ids=w, tokens=tokens, token_bytes=raw_bytes,
        spans=spans, backend="ids",
    ),
    word_units=words,
    sentence_spans=sentences,
    math_spans=math_spans,
)

parts = {
    "decode": lambda: tok.subword.decode(w),
    "token_bytes_spans": lambda: [
        (tok.subword.token_bytes[i], (0, len(tok.subword.token_bytes[i])))
        for i in w
    ],
    "char_offsets": lambda: _char_byte_offsets(text_one),
    "word_offsets": lambda: tok.word.encode_with_offsets(
        text_one, offsets, spans, token_offset=0),
    "sentences": lambda: list(tok.sentence.split(text_one)),
    "math": lambda: list(tok.math.spans(text_one)),
    "from_encoding": lambda: SpanSidecar.from_encoding(encoding),
    "full_row": lambda: SpanSidecar.from_token_ids(w, tok),
}
print("PART_MS_PER_ROW", json.dumps(
    {k: round(ms(v), 2) for k, v in parts.items()}), flush=True)
print("CPU_SIDECAR_BENCH_DONE", flush=True)
