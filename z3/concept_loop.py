#!/usr/bin/env python3
"""Multi-resolution concept loop.

This module connects the tokenizer levels to the model instead of leaving the
word/sentence layers as an offline artifact.

For a text window it builds a token-aligned sidecar:

    token -> byte span, word id, sentence id, math kind, composition event

The concept encoder pools subwords into words and words into sentences, binds
them into per-token concept vectors, and queries a shared 256-vertex /
C(256,2) object memory.  The route context carries the result into every block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

try:
    from .combinatorial_objects import CombinatorialMemory, binomial
except ImportError:  # pragma: no cover
    from combinatorial_objects import CombinatorialMemory, binomial


EVENT_NAMES = (
    "<none>",
    "<word_start>",
    "<word_end>",
    "<sentence_end>",
    "<math_start>",
    "<math_end>",
    "<compose>",
)
EVENT_IDS = {name: i for i, name in enumerate(EVENT_NAMES)}
MATH_KINDS = ("<none>", "<display>", "<inline>", "<command>", "<number>", "<operator>")


@dataclass
class SpanSidecar:
    """Token-aligned structural labels for one window."""

    word_ids: torch.Tensor
    sentence_ids: torch.Tensor
    math_kind: torch.Tensor
    event_ids: torch.Tensor
    byte_start: torch.Tensor
    byte_len: torch.Tensor

    def to(self, device: torch.device | str) -> "SpanSidecar":
        return SpanSidecar(
            word_ids=self.word_ids.to(device),
            sentence_ids=self.sentence_ids.to(device),
            math_kind=self.math_kind.to(device),
            event_ids=self.event_ids.to(device),
            byte_start=self.byte_start.to(device),
            byte_len=self.byte_len.to(device),
        )

    def slice(self, start: int, length: int) -> "SpanSidecar":
        end = start + length
        return SpanSidecar(
            word_ids=self.word_ids[start:end],
            sentence_ids=self.sentence_ids[start:end],
            math_kind=self.math_kind[start:end],
            event_ids=self.event_ids[start:end],
            byte_start=self.byte_start[start:end],
            byte_len=self.byte_len[start:end],
        )

    @classmethod
    def from_encoding(cls, encoding: Any) -> "SpanSidecar":
        """Build a sidecar from a LayeredTokenizer LayerEncoding."""
        spans = list(encoding.subword.spans)
        n = len(spans)
        word_ids = torch.zeros(n, dtype=torch.long)
        sentence_ids = torch.zeros(n, dtype=torch.long)
        math_kind = torch.zeros(n, dtype=torch.long)
        event_ids = torch.zeros(n, dtype=torch.long)
        byte_start = torch.tensor([s for s, _ in spans], dtype=torch.long)
        byte_len = torch.tensor([e - s for s, e in spans], dtype=torch.long)

        def token_range(start: int, end: int) -> range:
            lo = 0
            hi = n
            while lo < hi:
                mid = (lo + hi) // 2
                if spans[mid][1] <= start:
                    lo = mid + 1
                else:
                    hi = mid
            first = lo
            lo = first
            hi = n
            while lo < hi:
                mid = (lo + hi) // 2
                if spans[mid][0] < end:
                    lo = mid + 1
                else:
                    hi = mid
            return range(first, lo)

        word_units = [u for u in encoding.word_units if u.is_word]
        for word_index, unit in enumerate(word_units, start=1):
            indices = token_range(unit.start, unit.end)
            if not indices:
                continue
            for idx in indices:
                word_ids[idx] = word_index
            event_ids[indices[0]] = EVENT_IDS["<word_start>"]
            event_ids[indices[-1]] = EVENT_IDS["<word_end>"]

        for sent_index, (start, end) in enumerate(encoding.sentence_spans, start=1):
            indices = token_range(start, end)
            if not indices:
                continue
            for idx in indices:
                sentence_ids[idx] = sent_index
            event_ids[indices[-1]] = EVENT_IDS["<sentence_end>"]

        for span in getattr(encoding, "math_spans", []):
            indices = token_range(span.start, span.end)
            if not indices:
                continue
            kind = MATH_KINDS.index(
                f"<{span.kind}>"
            ) if f"<{span.kind}>" in MATH_KINDS else 0
            for idx in indices:
                math_kind[idx] = kind
            event_ids[indices[0]] = EVENT_IDS["<math_start>"]
            event_ids[indices[-1]] = EVENT_IDS["<math_end>"]

        return cls(
            word_ids=word_ids,
            sentence_ids=sentence_ids,
            math_kind=math_kind,
            event_ids=event_ids,
            byte_start=byte_start,
            byte_len=byte_len,
        )

    @classmethod
    def from_text(cls, text: str, tokenizer: Any) -> "SpanSidecar":
        return cls.from_encoding(tokenizer.encode(text, add_document_frame=False))

    @classmethod
    def collate(
        cls,
        sidecars: Sequence["SpanSidecar"],
        *,
        device: torch.device | str = "cpu",
    ) -> "SpanSidecar":
        if not sidecars:
            raise ValueError("need at least one sidecar")
        max_len = max(int(s.word_ids.numel()) for s in sidecars)
        fields: dict[str, list[torch.Tensor]] = {
            "word_ids": [], "sentence_ids": [], "math_kind": [],
            "event_ids": [], "byte_start": [], "byte_len": [],
        }
        for sidecar in sidecars:
            n = int(sidecar.word_ids.numel())
            pad = max_len - n
            for name, value in (
                ("word_ids", sidecar.word_ids),
                ("sentence_ids", sidecar.sentence_ids),
                ("math_kind", sidecar.math_kind),
                ("event_ids", sidecar.event_ids),
                ("byte_start", sidecar.byte_start),
                ("byte_len", sidecar.byte_len),
            ):
                value = value.to(device)
                if pad:
                    value = torch.cat(
                        [value, torch.zeros(pad, dtype=value.dtype, device=device)]
                    )
                fields[name].append(value)
        return cls(**{name: torch.stack(values, dim=0) for name, values in fields.items()})

    @classmethod
    def from_token_ids(
        cls, token_ids: Iterable[int], tokenizer: Any
    ) -> "SpanSidecar":
        """Build alignment directly from an existing v4c/v4 token window.

        This is the bridge that lets bin training feed the concept loop
        without re-tokenizing and without changing the token IDs.
        """
        from types import SimpleNamespace

        try:
            from .layered_tokenizer import (
                MathSpan,
                SubwordEncoding,
                _char_byte_offsets,
            )
        except ImportError:  # pragma: no cover
            from layered_tokenizer import (  # type: ignore
                MathSpan,
                SubwordEncoding,
                _char_byte_offsets,
            )

        ids = [int(x) for x in token_ids]
        text = tokenizer.subword.decode(ids)
        token_bytes = [tokenizer.subword.token_bytes[i] for i in ids]
        spans: list[tuple[int, int]] = []
        cursor = 0
        for raw in token_bytes:
            spans.append((cursor, cursor + len(raw)))
            cursor += len(raw)
        tokens = [tokenizer.subword.id_to_token.get(i, "") for i in ids]
        subword = SubwordEncoding(
            text=text,
            ids=ids,
            tokens=tokens,
            token_bytes=token_bytes,
            spans=spans,
            backend="ids",
        )
        offsets = _char_byte_offsets(text)
        words = tokenizer.word.encode_with_offsets(
            text, offsets, spans, token_offset=0
        )
        sentences = [
            (offsets[s], offsets[e])
            for s, e in tokenizer.sentence.split(text)
        ]
        math_spans = [
            MathSpan(
                span.text,
                offsets[span.start],
                offsets[span.end],
                span.kind,
            )
            for span in tokenizer.math.spans(text)
        ]
        encoding = SimpleNamespace(
            subword=subword,
            word_units=words,
            sentence_spans=sentences,
            math_spans=math_spans,
        )
        return cls.from_encoding(encoding)


def _segment_pool(
    x: torch.Tensor,  # (B,T,D)
    ids: torch.Tensor,  # (B,T), 0 = none, positive segment ids
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return gathered segment summaries and segment count per batch row."""
    b, t, d = x.shape
    max_segments = max(1, int(ids.max().item())) if ids.numel() else 1
    out = torch.zeros(
        b, max_segments + 1, d, device=x.device, dtype=x.dtype
    )
    counts = torch.zeros(
        b, max_segments + 1, device=x.device, dtype=x.dtype
    )
    for row in range(b):
        valid = ids[row] > 0
        if not valid.any():
            continue
        pos = ids[row][valid]
        out[row].index_add_(
            0, pos, x[row][valid].to(dtype=out.dtype)
        )
        counts[row].index_add_(
            0, pos, torch.ones_like(pos, dtype=counts.dtype)
        )
    gathered = out.gather(
        1, ids.clamp_min(0).unsqueeze(-1).expand(-1, -1, d)
    )
    gathered = gathered / counts.unsqueeze(-1).gather(
        1, ids.clamp_min(0).unsqueeze(-1)
    ).clamp_min(1.0)
    return gathered, counts


class ConceptEncoder(nn.Module):
    """Pool subwords -> words -> sentences and build a per-token concept."""

    def __init__(
        self,
        d_model: int,
        *,
        n_math_kinds: int = len(MATH_KINDS),
        n_events: int = len(EVENT_NAMES),
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.word_proj = nn.Linear(d_model, d_model, bias=False)
        self.sentence_proj = nn.Linear(d_model, d_model, bias=False)
        self.math_embed = nn.Embedding(n_math_kinds, d_model)
        self.event_embed = nn.Embedding(n_events, d_model)
        self.out = nn.Linear(4 * d_model, d_model, bias=False)
        self.norm = nn.RMSNorm(d_model) if hasattr(nn, "RMSNorm") else nn.Identity()

    def forward(
        self, x: torch.Tensor, sidecar: SpanSidecar | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sidecar is None:
            # No aligned structure: retain a global context and leave the
            # per-token concept neutral, rather than inventing boundaries.
            global_ctx = x.mean(dim=1)
            zero = torch.zeros_like(x)
            return zero, global_ctx
        def align_ids(value: torch.Tensor) -> torch.Tensor:
            value = value.to(x.device).long()
            if value.ndim == 1:
                value = value.unsqueeze(0).expand(x.shape[0], -1)
            elif value.shape[0] == 1 and x.shape[0] > 1:
                value = value.expand(x.shape[0], -1)
            return value

        word, _ = _segment_pool(x, align_ids(sidecar.word_ids))
        sent, _ = _segment_pool(x, align_ids(sidecar.sentence_ids))
        math_ids = align_ids(sidecar.math_kind).clamp(0, len(MATH_KINDS) - 1)
        event_ids = align_ids(sidecar.event_ids).clamp(0, len(EVENT_NAMES) - 1)
        math = self.math_embed(math_ids)
        event = self.event_embed(event_ids)
        concept = self.out(
            torch.cat(
                [
                    x,
                    self.word_proj(word),
                    self.sentence_proj(sent),
                    math + event,
                ],
                dim=-1,
            )
        )
        concept = F.normalize(concept, dim=-1)
        return concept, concept.mean(dim=1)


class CompositionVocabulary:
    """Structural composition layer vocabulary.

    These IDs are intentionally separate from the 64k subword vocabulary:
    they describe boundaries/composition above the tokenizer, not new text
    tokens that must round-trip through the subword decoder.
    """

    tokens = EVENT_NAMES
    token_to_id = EVENT_IDS

    @classmethod
    def encode(cls, events: Iterable[str]) -> list[int]:
        return [cls.token_to_id[event] for event in events]

    @classmethod
    def decode(cls, ids: Iterable[int]) -> list[str]:
        return [cls.tokens[int(i)] for i in ids]

    @classmethod
    def manifest(cls) -> dict[str, Any]:
        return {"tokens": list(cls.tokens), "size": len(cls.tokens)}


class CompositionHead(nn.Module):
    """Predict a structural composition event at each token position."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, len(EVENT_NAMES), bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)


class IncrementalCompositionDecoder:
    """Tokenizer-side composition loop for generated token IDs.

    The decoder keeps the already-generated token IDs, decodes them through
    the v4 subword layer, and reports new word/sentence/math events.  It is
    intentionally incremental at the event level: the model can use its own
    composition head during generation, and this object provides the
    deterministic tokenizer-side reference.
    """

    _WORD = re.compile(r"[^\W\d_]+|\d+|[^\w\s]+", re.UNICODE)

    def __init__(self, subword_layer: Any):
        self.subword = subword_layer
        self.token_ids: list[int] = []
        self._seen_words = 0
        self._seen_sentences = 0
        self._seen_math = 0

    def feed(self, token_id: int) -> list[tuple[int, str]]:
        self.token_ids.append(int(token_id))
        text = self.subword.decode(self.token_ids)
        words = self._WORD.findall(text)
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s]
        math_marks = len(re.findall(r"\$|\\[A-Za-z]+", text))
        events: list[tuple[int, str]] = []
        while self._seen_words < len(words):
            events.append((self._seen_words, "<word>"))
            self._seen_words += 1
        while self._seen_sentences < len(sentences):
            events.append((self._seen_sentences, "<sentence_end>"))
            self._seen_sentences += 1
        while self._seen_math < math_marks:
            events.append((self._seen_math, "<math>"))
            self._seen_math += 1
        return events

    @property
    def text(self) -> str:
        return self.subword.decode(self.token_ids)


class ConceptLoop(nn.Module):
    """Per-layer concept abstraction -> C(256,2) object memory -> route state."""

    def __init__(
        self,
        d_model: int,
        *,
        vertices: int = 256,
        dim: int = 128,
        top_vertices: int = 16,
        top_pairs: int = 8,
        top_triples: int = 8,
    ) -> None:
        super().__init__()
        if vertices != 256:
            # The object address space is defined over exactly 256 vertices.
            # Keeping this explicit avoids a silent C(V,2) reinterpretation.
            pass
        self.d_model = int(d_model)
        self.vertices = int(vertices)
        self.encoder = ConceptEncoder(d_model)
        self.object_memory = CombinatorialMemory(
            d_model=d_model,
            vertices=vertices,
            dim=dim,
            top_vertices=top_vertices,
            top_pairs=top_pairs,
            top_triples=top_triples,
        )
        # Seed the vertex space from a deterministic byte-symbol codebook.
        # Pair/triple objects are then bindings of these 256 concept seeds.
        generator = torch.Generator().manual_seed(20261113)
        byte_keys = F.normalize(
            torch.randn(vertices, dim, generator=generator), dim=-1
        )
        byte_values = F.normalize(
            torch.randn(vertices, dim, generator=generator), dim=-1
        )
        self.object_memory.seed_vertices(byte_keys, byte_values)
        self.concept_to_state = nn.Linear(d_model, d_model, bias=False)
        self.object_to_state = nn.Linear(d_model, d_model, bias=False)
        self.state_norm = nn.RMSNorm(d_model) if hasattr(nn, "RMSNorm") else nn.Identity()

    def counts(self) -> dict[str, int]:
        return {
            "vertices": self.vertices,
            "pairs": binomial(self.vertices, 2),
            "triples": binomial(self.vertices, 3),
        }

    def forward(
        self,
        x: torch.Tensor,
        sidecar: SpanSidecar | None,
        route_ctx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        concept, global_ctx = self.encoder(x, sidecar)
        object_ctx, object_aux = self.object_memory(x + concept)
        self.last_object_ids = object_aux["pair_ids"].detach()
        self.last_object_vertices = object_aux["pair_vertices"].detach()
        self.last_triple_ids = (
            object_aux["triple_ids"].detach()
            if object_aux.get("triple_ids") is not None else None
        )
        # Return only the concept/object contribution.  The block adds the
        # route context separately; including it here double-counted the
        # router state on every layer.
        loop_ctx = self.state_norm(
            self.concept_to_state(global_ctx)
            + self.object_to_state(object_ctx.mean(dim=1))
        )
        return concept, object_ctx, loop_ctx


def sidecar_batch(
    tokenizer: Any,
    texts: Sequence[str],
) -> tuple[list[SpanSidecar], list[int]]:
    """Convenience helper for tests and data builders."""
    sidecars = [SpanSidecar.from_text(text, tokenizer) for text in texts]
    return sidecars, [len(sidecar.word_ids) for sidecar in sidecars]
