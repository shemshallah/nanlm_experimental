#!/usr/bin/env python3
"""Lossless, aligned tokenization at six levels.

Levels
------
bit
    The raw signal: 8 bits per byte, MSB first by default.
byte
    UTF-8 bytes.  All higher layers are spans over this sequence.
subword
    The bundled v4 64k byte-level BPE vocabulary.  Every subword is a
    contiguous byte span and decodes back to the original bytes.
token
    The flat model stream: subword IDs plus document/special IDs.  `<doc>`
    opens a document and `<eos>` closes it; `<now>` and future slots are
    reserved.
word
    Unicode word/punctuation/whitespace spans, with an optional tuned
    top-K word vocabulary and byte/subword fallback for OOV words.
sentence
    Sentence spans with configurable maximum length.  The spans tile the
    document exactly, so sentence boundaries never lose bytes.

The implementation has two subword backends.  The native HuggingFace
``tokenizers`` backend is used when installed (exact v4 IDs and offsets).
The pure-Python backend parses the same tokenizer JSON, applies the learned
BPE merges per pre-token, and is used by the dependency-free tests.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

try:  # optional exact backend
    from tokenizers import Tokenizer as _HFTokenizer  # type: ignore
except ImportError:  # pragma: no cover - exercised in dependency-free mode
    _HFTokenizer = None


DEFAULT_SUBWORD_PATH = (
    Path(__file__).resolve().parent / "tokenizers" / "v4_64k" / "tokenizer.json"
)

# GPT-2 ByteLevel's reversible byte<->unicode alphabet.
def bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_ENCODER = bytes_to_unicode()
BYTE_DECODER = {v: k for k, v in BYTE_ENCODER.items()}


def _char_byte_offsets(text: str) -> list[int]:
    """Map every character boundary to a UTF-8 byte boundary."""
    offsets = [0]
    total = 0
    for ch in text:
        total += len(ch.encode("utf-8", errors="ignore"))
        offsets.append(total)
    return offsets


def _split_pretokens(text: str) -> list[str]:
    """Lossless GPT-2-like pre-tokenization used by the pure backend."""
    pattern = re.compile(
        r"'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+",
        re.UNICODE,
    )
    pieces = pattern.findall(text)
    if "".join(pieces) != text:
        # Never lose a byte if the regex misses a Unicode edge case.
        pieces = [text]
    return [p for p in pieces if p]


def _bpe_merge(
    chars: list[str], ranks: dict[tuple[str, str], int]
) -> list[str]:
    """Apply BPE merges to one pre-token, returning token strings."""
    if len(chars) < 2:
        return chars
    symbols = list(chars)
    while len(symbols) > 1:
        best_rank: int | None = None
        best_pair: tuple[str, str] | None = None
        for left, right in zip(symbols, symbols[1:]):
            rank = ranks.get((left, right))
            if rank is not None and (best_rank is None or rank < best_rank):
                best_rank = rank
                best_pair = (left, right)
        if best_pair is None:
            break
        merged = best_pair[0] + best_pair[1]
        out: list[str] = []
        i = 0
        while i < len(symbols):
            if (
                i + 1 < len(symbols)
                and symbols[i] == best_pair[0]
                and symbols[i + 1] == best_pair[1]
            ):
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        symbols = out
    return symbols


class BitLayer:
    """Raw bit stream; the base layer for every byte."""

    def __init__(self, msb_first: bool = True):
        self.msb_first = bool(msb_first)

    def encode(self, data: bytes) -> list[int]:
        shifts = range(7, -1, -1) if self.msb_first else range(8)
        return [(byte >> shift) & 1 for byte in data for shift in shifts]

    def decode(self, bits: Iterable[int]) -> bytes:
        values = list(bits)
        if len(values) % 8:
            raise ValueError("bit length must be a multiple of 8")
        out = bytearray()
        shifts = range(7, -1, -1) if self.msb_first else range(8)
        for offset in range(0, len(values), 8):
            byte = 0
            for shift, bit in zip(shifts, values[offset:offset + 8]):
                byte |= (int(bit) & 1) << shift
            out.append(byte)
        return bytes(out)

    def pack(self, bits: Iterable[int]) -> bytes:
        return self.decode(bits)


class ByteLayer:
    """Strict UTF-8 byte layer."""

    def encode(self, text: str) -> bytes:
        return text.encode("utf-8")

    def decode(self, data: bytes) -> str:
        return data.decode("utf-8")


@dataclass
class MathSpan:
    text: str
    start: int
    end: int
    kind: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "kind": self.kind,
        }


class MathLayer:
    """Lossless math/notation spans and atom diagnostics.

    This layer does not rewrite text.  It marks display/inline equations,
    LaTeX commands, numbers, and operators so the word/subword layers and the
    router can distinguish prose from notation.  A separate canonicalizer is
    available for data preparation where normalization is allowed.
    """

    _DISPLAY = re.compile(r"\$\$.*?\$\$|\\\[.*?\\\]", re.DOTALL)
    _INLINE = re.compile(r"(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)", re.DOTALL)
    _COMMAND = re.compile(r"\\[A-Za-z]+")
    _NUMBER = re.compile(
        r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
    )
    _OPERATOR = re.compile(r"[=+\-*/^_<>≤≥≈≠×÷]")

    def spans(self, text: str) -> list[MathSpan]:
        occupied = [False] * len(text)
        found: list[MathSpan] = []
        for kind, pattern in (("display", self._DISPLAY), ("inline", self._INLINE)):
            for match in pattern.finditer(text):
                start, end = match.span()
                if any(occupied[max(0, start):end]):
                    continue
                found.append(MathSpan(match.group(0), start, end, kind))
                for i in range(start, end):
                    occupied[i] = True
        for kind, pattern in (
            ("command", self._COMMAND),
            ("number", self._NUMBER),
            ("operator", self._OPERATOR),
        ):
            for match in pattern.finditer(text):
                start, end = match.span()
                if any(occupied[max(0, start):end]):
                    continue
                found.append(MathSpan(match.group(0), start, end, kind))
                for i in range(start, end):
                    occupied[i] = True
        return sorted(found, key=lambda span: (span.start, span.end))

    def tune(
        self,
        text: str,
        subword: "SubwordLayer",
        *,
        min_atomic_rate: float = 0.0,
    ) -> dict[str, Any]:
        spans = self.spans(text)
        by_kind: dict[str, int] = {}
        subword_counts = []
        atom_total = 0
        atomic = 0
        for span in spans:
            by_kind[span.kind] = by_kind.get(span.kind, 0) + 1
            n_sub = len(subword.encode(span.text).ids)
            subword_counts.append(n_sub)
            if span.kind in ("number", "operator", "command"):
                atom_total += 1
                atomic += int(n_sub == 1)
        raw = text.encode("utf-8", errors="ignore")
        covered = 0
        if raw:
            # Math spans are char offsets; report coverage in characters as a
            # cheap proxy and byte coverage for the final report.
            covered_bytes = 0
            for span in spans:
                covered_bytes += len(
                    text[span.start:span.end].encode("utf-8", errors="ignore")
                )
            covered = covered_bytes / len(raw)
        atomic_rate = atomic / atom_total if atom_total else 0.0
        return {
            "spans": len(spans),
            "by_kind": by_kind,
            "byte_coverage": covered,
            "mean_subword_per_span": (
                sum(subword_counts) / len(subword_counts)
                if subword_counts else 0.0
            ),
            "max_subword_per_span": max(subword_counts) if subword_counts else 0,
            "atomic_number_operator_command_rate": atomic_rate,
            "gate": {
                "applicable": bool(spans),
                "min_atomic_rate": min_atomic_rate,
                "passed": bool(spans) and atomic_rate >= min_atomic_rate,
            },
        }

    @staticmethod
    def canonicalize(text: str) -> str:
        """Optional data-preparation normalization (not used for lossless IDs)."""
        import unicodedata

        value = unicodedata.normalize("NFKC", text)
        replacements = {
            "×": "*", "÷": "/", "−": "-", "–": "-", "—": "-",
            "≤": "<=", "≥": ">=", "≠": "!=", "≈": "~",
        }
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value


@dataclass
class SubwordEncoding:
    text: str
    ids: list[int]
    tokens: list[str]
    token_bytes: list[bytes]
    spans: list[tuple[int, int]]
    backend: str = "pure"

    @property
    def n_bytes(self) -> int:
        return sum(len(b) for b in self.token_bytes)


class SubwordLayer:
    """v4 byte-level BPE with byte spans for every token."""

    def __init__(
        self,
        path: str | Path = DEFAULT_SUBWORD_PATH,
        *,
        prefer_native: bool = True,
    ) -> None:
        self.path = Path(path).expanduser()
        if not self.path.is_file():
            raise FileNotFoundError(f"subword tokenizer not found: {self.path}")
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        model = payload.get("model", {})
        self.vocab: dict[str, int] = {
            str(k): int(v) for k, v in model.get("vocab", {}).items()
        }
        merges = model.get("merges", [])
        self.ranks: dict[tuple[str, str], int] = {}
        for rank, pair in enumerate(merges):
            if isinstance(pair, str):
                left, right = pair.split(" ", 1)
            else:
                left, right = pair
            self.ranks[(str(left), str(right))] = rank
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        added = payload.get("added_tokens", [])
        self.special_tokens = [
            str(item["content"]) for item in sorted(added, key=lambda x: int(x["id"]))
        ]
        self.special_ids = {token: self.vocab[token] for token in self.special_tokens}
        self.base_vocab_size = (
            max(self.vocab.values()) + 1 if self.vocab else 0
        )

        # The v4 BPE is not byte-complete: 60 byte values (including ``{``)
        # never made it into the 64k vocabulary.  Add explicit one-byte
        # fallback IDs above the base vocabulary.  This is lossless and keeps
        # every existing base ID stable.
        self.extra_byte_ids: dict[int, int] = {}
        next_id = self.base_vocab_size
        for byte_value, char in BYTE_ENCODER.items():
            if char not in self.vocab:
                self.vocab[char] = next_id
                self.extra_byte_ids[byte_value] = next_id
                next_id += 1
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.extra_byte_count = len(self.extra_byte_ids)
        self.token_bytes: list[bytes] = [b""] * (self.base_vocab_size + self.extra_byte_count)
        for token, token_id in self.vocab.items():
            if token in self.special_ids:
                raw = token.encode("utf-8")
            else:
                raw = bytes(
                    BYTE_DECODER[ch] for ch in token if ch in BYTE_DECODER
                )
            self.token_bytes[token_id] = raw
        # Byte fallback: every byte has a single-byte token, either from the
        # original vocabulary or from the added fallback ID block.
        self.byte_ids: dict[int, int] = {}
        for byte_value, char in BYTE_ENCODER.items():
            token_id = self.vocab.get(char)
            if token_id is not None:
                self.byte_ids[byte_value] = token_id
        self.native = None
        if prefer_native and _HFTokenizer is not None:
            try:
                self.native = _HFTokenizer.from_file(str(self.path))
            except Exception:
                self.native = None

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _token_ids(self, token: str) -> list[int]:
        token_id = self.vocab.get(token)
        if token_id is not None:
            return [token_id]
        # A merged token that is absent from the vocabulary is decomposed
        # back to complete bytes, including the added fallback IDs.
        out: list[int] = []
        for ch in token:
            byte_value = BYTE_DECODER.get(ch)
            if byte_value is None or byte_value not in self.byte_ids:
                raise KeyError(f"BPE token missing from vocabulary: {token!r}")
            out.append(self.byte_ids[byte_value])
        return out

    def _native_encode(self, text: str) -> SubwordEncoding:
        assert self.native is not None
        encoding = self.native.encode(text)
        char_offsets = _char_byte_offsets(text)
        spans: list[tuple[int, int]] = []
        token_bytes: list[bytes] = []
        tokens = list(encoding.tokens)
        for token, (start_char, end_char) in zip(tokens, encoding.offsets):
            spans.append(
                (
                    char_offsets[max(0, min(start_char, len(text)))],
                    char_offsets[max(0, min(end_char, len(text)))],
                )
            )
            token_bytes.append(self.token_bytes[int(encoding.ids[len(token_bytes)])])
        # A native tokenizer that disagrees on byte coverage is not usable
        # for alignment; fall back rather than silently corrupting spans.
        if sum(len(b) for b in token_bytes) != len(text.encode("utf-8")):
            return self._pure_encode(text)
        return SubwordEncoding(
            text=text,
            ids=list(encoding.ids),
            tokens=tokens,
            token_bytes=token_bytes,
            spans=spans,
            backend="native",
        )

    def _pure_encode(self, text: str) -> SubwordEncoding:
        ids: list[int] = []
        tokens: list[str] = []
        token_bytes: list[bytes] = []
        spans: list[tuple[int, int]] = []
        byte_cursor = 0
        for piece in _split_pretokens(text):
            raw = piece.encode("utf-8", errors="ignore")
            mapped = [BYTE_ENCODER[b] for b in raw]
            for token in _bpe_merge(mapped, self.ranks):
                for token_id in self._token_ids(token):
                    raw_token = self.token_bytes[token_id]
                    ids.append(token_id)
                    tokens.append(self.id_to_token.get(token_id, token))
                    token_bytes.append(raw_token)
                    spans.append((byte_cursor, byte_cursor + len(raw_token)))
                    byte_cursor += len(raw_token)
        return SubwordEncoding(
            text=text,
            ids=ids,
            tokens=tokens,
            token_bytes=token_bytes,
            spans=spans,
            backend="pure",
        )

    def encode(self, text: str) -> SubwordEncoding:
        if self.native is not None:
            try:
                return self._native_encode(text)
            except Exception:
                # Missing byte tokens (e.g. ``{``) can make the external
                # tokenizer reject a string; the complete pure backend keeps
                # the representation lossless.
                return self._pure_encode(text)
        return self._pure_encode(text)

    def decode(self, ids: Iterable[int]) -> str:
        raw = b"".join(self.token_bytes[int(i)] for i in ids)
        return raw.decode("utf-8", errors="replace")

    def manifest(self) -> dict[str, Any]:
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        return {
            "path": str(self.path),
            "sha256": digest,
            "base_vocab_size": self.base_vocab_size,
            "vocab_size": self.vocab_size,
            "extra_byte_fallbacks": self.extra_byte_count,
            "backend": "native" if self.native is not None else "pure",
            "special_tokens": self.special_tokens,
        }


@dataclass
class WordUnit:
    text: str
    start: int
    end: int
    word_id: int | None
    subword_start: int
    subword_end: int
    fallback_subwords: int = 0
    is_word: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "word_id": self.word_id,
            "subword_start": self.subword_start,
            "subword_end": self.subword_end,
            "fallback_subwords": self.fallback_subwords,
            "is_word": self.is_word,
        }


class WordLayer:
    """Unicode word/punctuation/space spans with an optional top-K vocab."""

    _RE = re.compile(r"[^\W\d_]+|\d+|[^\w\s]+|\s+", re.UNICODE)

    def __init__(self, vocab_size: int = 0, vocab: dict[str, int] | None = None):
        self.vocab_size = int(vocab_size)
        self.vocab: dict[str, int] = dict(vocab or {})
        if self.vocab and not self.vocab_size:
            self.vocab_size = len(self.vocab)
        self._reverse = {v: k for k, v in self.vocab.items()}

    def spans(self, text: str) -> list[tuple[str, int, int, bool]]:
        return [
            (m.group(0), m.start(), m.end(), bool(re.search(r"\w", m.group(0))))
            for m in self._RE.finditer(text)
        ]

    @classmethod
    def tune(
        cls,
        text: str,
        candidates: Sequence[int],
        *,
        min_coverage: float = 0.95,
    ) -> tuple[int, dict[str, Any]]:
        spans = cls().spans(text)
        words = [
            (piece, start, end)
            for piece, start, end, is_word in spans
            if is_word
        ]
        counts = Counter(piece for piece, _, _ in words)
        total = max(1, sum(counts.values()))
        results = []
        for size in sorted(set(int(x) for x in candidates if int(x) > 0)):
            chosen = [word for word, _ in counts.most_common(size)]
            covered = sum(counts[word] for word in chosen)
            coverage = covered / total
            results.append(
                {
                    "vocab_size": size,
                    "coverage": coverage,
                    "fallback_rate": 1.0 - coverage,
                    "distinct_words": len(counts),
                    "types_covered": len(chosen),
                }
            )
        passing = [r for r in results if r["coverage"] >= min_coverage]
        best = (
            min(passing, key=lambda r: (r["vocab_size"], -r["coverage"]))
            if passing
            else max(results, key=lambda r: r["coverage"])
        )
        return int(best["vocab_size"]), {
            "counts": counts,
            "total_words": total,
            "candidates": results,
            "selected": best,
            "gate": {
                "min_coverage": min_coverage,
                "passed": bool(passing),
            },
        }

    def fit(self, text: str, vocab_size: int) -> "WordLayer":
        spans = self.spans(text)
        counts = Counter(
            piece for piece, _, _, is_word in spans if is_word
        )
        vocab = {
            word: index
            for index, (word, _) in enumerate(counts.most_common(vocab_size))
        }
        return WordLayer(vocab_size=len(vocab), vocab=vocab)

    def encode_with_offsets(
        self,
        text: str,
        byte_offsets: Sequence[int],
        subword_spans: Sequence[tuple[int, int]],
        *,
        token_offset: int = 0,
    ) -> list[WordUnit]:
        starts = [s for s, _ in subword_spans]
        ends = [e for _, e in subword_spans]
        units: list[WordUnit] = []
        for piece, start_char, end_char, is_word in self.spans(text):
            start = byte_offsets[start_char]
            end = byte_offsets[end_char]
            lo = bisect.bisect_right(ends, start)
            hi = bisect.bisect_left(starts, end)
            if hi < lo:
                hi = lo
            word_id = self.vocab.get(piece) if is_word else None
            fallback = 0 if word_id is not None or not is_word else max(0, hi - lo)
            units.append(
                WordUnit(
                    text=piece,
                    start=start,
                    end=end,
                    word_id=word_id,
                    subword_start=lo + token_offset,
                    subword_end=hi + token_offset,
                    fallback_subwords=fallback,
                    is_word=is_word,
                )
            )
        return units


class SentenceLayer:
    """Lossless sentence spans with optional long-sentence splitting."""

    _BOUNDARY = re.compile(r"[.!?]+[\"')\]]*(?=\s|$)", re.UNICODE)

    def __init__(self, max_chars: int = 512, min_chars: int = 1):
        self.max_chars = max(1, int(max_chars))
        self.min_chars = max(1, int(min_chars))

    def split(self, text: str) -> list[tuple[int, int]]:
        """Return character spans that tile ``text`` exactly."""
        if not text:
            return []
        boundaries = [0]
        for match in self._BOUNDARY.finditer(text):
            end = match.end()
            if end > boundaries[-1] and end < len(text):
                boundaries.append(end)
        boundaries.append(len(text))
        spans: list[tuple[int, int]] = []
        for start, end in zip(boundaries, boundaries[1:]):
            if end <= start:
                continue
            cursor = start
            while end - cursor > self.max_chars:
                cut = text.rfind(" ", cursor + 1, cursor + self.max_chars)
                if cut <= cursor:
                    cut = cursor + self.max_chars
                if cut <= cursor:
                    break
                spans.append((cursor, cut))
                cursor = cut
            if end > cursor:
                spans.append((cursor, end))
        # Coalesce tiny spans with the next span without crossing max_chars.
        merged: list[tuple[int, int]] = []
        for span in spans:
            if (
                merged
                and span[1] - span[0] < self.min_chars
                and span[1] - merged[-1][0] <= self.max_chars
            ):
                merged[-1] = (merged[-1][0], span[1])
            else:
                merged.append(span)
        return merged

    @classmethod
    def tune(
        cls,
        text: str,
        candidates: Sequence[int],
        *,
        target_max_chars: int = 512,
    ) -> tuple[int, dict[str, Any]]:
        results = []
        for max_chars in sorted(set(int(x) for x in candidates if int(x) > 0)):
            layer = cls(max_chars=max_chars)
            spans = layer.split(text)
            lengths = [e - s for s, e in spans]
            coverage = sum(lengths) / max(1, len(text))
            results.append(
                {
                    "max_chars": max_chars,
                    "sentences": len(spans),
                    "mean_chars": (sum(lengths) / len(lengths)) if lengths else 0.0,
                    "max_observed_chars": max(lengths) if lengths else 0,
                    "coverage": coverage,
                    "over_target": sum(1 for x in lengths if x > target_max_chars),
                }
            )
        passing = [
            r
            for r in results
            if r["coverage"] >= 0.999
            and r["max_observed_chars"] <= target_max_chars
        ]
        # Prefer the longest natural sentences that still respect the
        # target, not the most aggressively fragmented split.
        best = (
            max(passing, key=lambda r: (r["mean_chars"], r["max_chars"]))
            if passing
            else min(results, key=lambda r: r["over_target"])
        )
        return int(best["max_chars"]), {
            "candidates": results,
            "selected": best,
            "target_max_chars": target_max_chars,
            "gate": {"passed": bool(passing), "min_coverage": 0.999},
        }


@dataclass
class LayerEncoding:
    text: str
    bits: list[int] | None
    bytes: bytes
    subword: SubwordEncoding
    token_ids: list[int]
    word_units: list[WordUnit]
    sentence_spans: list[tuple[int, int]]
    math_spans: list[MathSpan] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self, include_bits: bool = False) -> dict[str, Any]:
        return {
            "chars": len(self.text),
            "bytes": len(self.bytes),
            "bits": len(self.bytes) * 8 if include_bits else None,
            "subword_ids": self.subword.ids,
            "subword_tokens": self.subword.tokens,
            "subword_spans": self.subword.spans,
            "subword_backend": self.subword.backend,
            "token_ids": self.token_ids,
            "words": [unit.as_dict() for unit in self.word_units],
            "sentences": [[s, e] for s, e in self.sentence_spans],
            "math": [span.as_dict() for span in self.math_spans],
            "metadata": self.metadata,
        }


class LayeredTokenizer:
    """Compose the six levels into one lossless, aligned tokenizer."""

    def __init__(
        self,
        subword_path: str | Path = DEFAULT_SUBWORD_PATH,
        *,
        word_vocab_size: int = 0,
        word_vocab: dict[str, int] | None = None,
        sentence_max_chars: int = 512,
        sentence_min_chars: int = 1,
        specials: Sequence[str] | None = None,
        prefer_native: bool = True,
    ) -> None:
        self.subword = SubwordLayer(
            subword_path, prefer_native=prefer_native
        )
        self.word = WordLayer(word_vocab_size, word_vocab)
        self.sentence = SentenceLayer(
            max_chars=sentence_max_chars, min_chars=sentence_min_chars
        )
        self.math = MathLayer()
        default_specials = self.subword.special_tokens
        self.specials = list(default_specials if specials is None else specials)
        for token in self.specials:
            if token not in self.subword.special_ids:
                raise ValueError(f"special token has no subword ID: {token}")
        self.doc_id = self.subword.special_ids.get("<doc>")
        self.eos_id = self.subword.special_ids.get("<eos>")
        self.now_id = self.subword.special_ids.get("<now>")

    @property
    def vocab_size(self) -> int:
        return self.subword.vocab_size

    def encode(
        self,
        text: str,
        *,
        add_document_frame: bool = True,
        include_bits: bool = False,
    ) -> LayerEncoding:
        subword = self.subword.encode(text)
        byte_offsets = _char_byte_offsets(text)
        token_offset = 1 if add_document_frame and self.doc_id is not None else 0
        words = self.word.encode_with_offsets(
            text,
            byte_offsets,
            subword.spans,
            token_offset=token_offset,
        )
        sentence_spans = [
            (byte_offsets[s], byte_offsets[e])
            for s, e in self.sentence.split(text)
        ]
        token_ids = list(subword.ids)
        if add_document_frame and self.doc_id is not None:
            token_ids = [self.doc_id] + token_ids
        if add_document_frame and self.eos_id is not None:
            token_ids = token_ids + [self.eos_id]
        bits = None
        if include_bits:
            bits = []
            for byte in text.encode("utf-8", errors="ignore"):
                bits.extend((byte >> shift) & 1 for shift in range(7, -1, -1))
        return LayerEncoding(
            text=text,
            bits=bits,
            bytes=text.encode("utf-8", errors="ignore"),
            subword=subword,
            token_ids=token_ids,
            word_units=words,
            sentence_spans=sentence_spans,
            math_spans=self.math.spans(text),
            metadata={
                "add_document_frame": add_document_frame,
                "token_offset": token_offset,
                "specials": self.specials,
            },
        )

    def decode_token_ids(self, ids: Iterable[int]) -> str:
        return self.subword.decode(ids).replace("<doc>", "").replace("<eos>", "")

    def decode_words(self, units: Sequence[WordUnit]) -> str:
        return "".join(unit.text for unit in units)

    def decode_sentences(self, text: str, spans: Sequence[tuple[int, int]]) -> str:
        raw = text.encode("utf-8", errors="ignore")
        return b"".join(raw[s:e] for s, e in spans).decode("utf-8", errors="replace")

    def manifest(self, *, include_word_vocab: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": 1,
            "layers": {
                "bit": {"order": "msb_first", "bits_per_byte": 8},
                "byte": {"encoding": "utf-8", "vocab_size": 256},
                "subword": self.subword.manifest(),
                "token": {
                    "doc_id": self.doc_id,
                    "eos_id": self.eos_id,
                    "now_id": self.now_id,
                    "specials": self.specials,
                },
                "word": {
                    "vocab_size": self.word.vocab_size,
                    "regex": self.word._RE.pattern,
                    "fallback": "subword",
                },
                "sentence": {
                    "max_chars": self.sentence.max_chars,
                    "min_chars": self.sentence.min_chars,
                    "boundary_regex": self.sentence._BOUNDARY.pattern,
                },
                "math": {
                    "spans": "display|inline|command|number|operator",
                    "lossless": True,
                    "canonicalize": "optional data preparation only",
                },
            },
        }
        if include_word_vocab:
            payload["word_vocab"] = dict(self.word.vocab)
        return payload


def write_artifact(
    tokenizer: LayeredTokenizer,
    path: str | Path,
    *,
    include_word_vocab: bool = True,
) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = tokenizer.manifest(include_word_vocab=include_word_vocab)
    payload["word_vocab"] = dict(tokenizer.word.vocab)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def load_artifact(
    path: str | Path, *, prefer_native: bool = True
) -> LayeredTokenizer:
    artifact_path = Path(path).expanduser()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    layers = payload.get("layers", {})
    subword = layers.get("subword", {})
    word = layers.get("word", {})
    sentence = layers.get("sentence", {})
    subword_path = Path(
        subword.get("path") or DEFAULT_SUBWORD_PATH
    ).expanduser()
    if not subword_path.is_absolute():
        subword_path = (artifact_path.parent / subword_path).resolve()
    if not subword_path.is_file():
        # A tuned artifact may have been produced in a throwaway Kaggle
        # workspace.  The bundled v4 tokenizer is the canonical fallback.
        subword_path = DEFAULT_SUBWORD_PATH
    return LayeredTokenizer(
        subword_path=subword_path,
        word_vocab_size=int(word.get("vocab_size", 0)),
        word_vocab=payload.get("word_vocab"),
        sentence_max_chars=int(sentence.get("max_chars", 512)),
        sentence_min_chars=int(sentence.get("min_chars", 1)),
        specials=layers.get("token", {}).get("specials"),
        prefer_native=prefer_native,
    )
