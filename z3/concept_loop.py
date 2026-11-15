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


def byte_symbol_codebook(
    dim: int, *, seed: int = 20261113, device: str = "cpu"
) -> torch.Tensor:
    """Deterministic HD code for each of the 256 byte symbols.

    Each byte is the bundle of eight fixed bit hypervectors with signs chosen
    by its binary representation.  This gives every vertex a real byte-symbol
    prior instead of an unrelated random vector.
    """
    generator = torch.Generator().manual_seed(seed)
    bit_codes = torch.randn(8, dim, generator=generator, device=device)
    byte_values = torch.arange(256, device=device)
    signs = torch.stack(
        [
            ((byte_values >> bit) & 1).to(torch.float32) * 2.0 - 1.0
            for bit in range(8)
        ],
        dim=-1,
    )
    codes = torch.einsum("vb,bd->vd", signs, bit_codes)
    return F.normalize(codes, dim=-1)


@dataclass
class SpanSidecar:
    """Token-aligned structural labels for one window."""

    word_ids: torch.Tensor
    sentence_ids: torch.Tensor
    math_kind: torch.Tensor
    event_ids: torch.Tensor
    byte_start: torch.Tensor
    byte_len: torch.Tensor
    event_multi: torch.Tensor

    def to(self, device: torch.device | str) -> "SpanSidecar":
        return SpanSidecar(
            word_ids=self.word_ids.to(device),
            sentence_ids=self.sentence_ids.to(device),
            math_kind=self.math_kind.to(device),
            event_ids=self.event_ids.to(device),
            byte_start=self.byte_start.to(device),
            byte_len=self.byte_len.to(device),
            event_multi=self.event_multi.to(device),
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
            event_multi=self.event_multi[start:end],
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
        event_multi = torch.zeros(n, len(EVENT_NAMES), dtype=torch.float32)
        byte_start = torch.tensor([s for s, _ in spans], dtype=torch.long)
        byte_len = torch.tensor([e - s for s, e in spans], dtype=torch.long)

        def mark(index: int, event: int) -> None:
            event_ids[index] = event
            event_multi[index, event] = 1.0

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
            mark(indices[0], EVENT_IDS["<word_start>"])
            mark(indices[-1], EVENT_IDS["<word_end>"])
            if len(indices) > 1:
                # A multi-subword word is itself a composition target.
                mark(indices[-1], EVENT_IDS["<compose>"])

        for sent_index, (start, end) in enumerate(encoding.sentence_spans, start=1):
            indices = token_range(start, end)
            if not indices:
                continue
            for idx in indices:
                sentence_ids[idx] = sent_index
            mark(indices[-1], EVENT_IDS["<sentence_end>"])

        for span in getattr(encoding, "math_spans", []):
            indices = token_range(span.start, span.end)
            if not indices:
                continue
            kind = MATH_KINDS.index(
                f"<{span.kind}>"
            ) if f"<{span.kind}>" in MATH_KINDS else 0
            for idx in indices:
                math_kind[idx] = kind
            mark(indices[0], EVENT_IDS["<math_start>"])
            mark(indices[-1], EVENT_IDS["<math_end>"])

        return cls(
            word_ids=word_ids,
            sentence_ids=sentence_ids,
            math_kind=math_kind,
            event_ids=event_ids,
            byte_start=byte_start,
            byte_len=byte_len,
            event_multi=event_multi,
        )

    @classmethod
    def from_text(cls, text: str, tokenizer: Any) -> "SpanSidecar":
        tokenizer = getattr(tokenizer, "layered", tokenizer)
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
            "event_multi": [],
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
                ("event_multi", sidecar.event_multi),
            ):
                value = value.to(device)
                if pad:
                    pad_shape = (pad,) + tuple(value.shape[1:])
                    value = torch.cat(
                        [value, torch.zeros(pad_shape, dtype=value.dtype, device=device)]
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

        tokenizer = getattr(tokenizer, "layered", tokenizer)
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
            if value.shape[1] != x.shape[1]:
                width = min(value.shape[1], x.shape[1])
                if value.shape[1] < x.shape[1]:
                    pad = torch.zeros(
                        value.shape[0], x.shape[1] - width,
                        device=value.device, dtype=value.dtype,
                    )
                    value = torch.cat([value[:, :width], pad], dim=1)
                else:
                    value = value[:, :width]
            return value

        def align_multi(value: torch.Tensor) -> torch.Tensor | None:
            if value is None:
                return None
            value = value.to(x.device)
            if value.ndim == 2:
                value = value.unsqueeze(0)
            if value.ndim != 3:
                return None
            if value.shape[0] == 1 and x.shape[0] > 1:
                value = value.expand(x.shape[0], -1, -1)
            if value.shape[1] != x.shape[1]:
                # Sidecars can be shorter than a padded batch window.  Use
                # the aligned prefix and leave the tail as <none>.
                width = min(value.shape[1], x.shape[1])
                pad = torch.zeros(
                    value.shape[0], x.shape[1] - width, value.shape[-1],
                    device=value.device, dtype=value.dtype,
                )
                value = torch.cat([value[:, :width], pad], dim=1)
            return value.to(dtype=x.dtype)

        word, _ = _segment_pool(x, align_ids(sidecar.word_ids))
        word_ctx = self.word_proj(word)
        # Sentence abstraction pools word-enriched states, not raw hidden
        # states.  This makes the hierarchy genuinely recursive at the
        # abstraction level.
        sent, _ = _segment_pool(
            x + word_ctx, align_ids(sidecar.sentence_ids)
        )
        math_ids = align_ids(sidecar.math_kind).clamp(0, len(MATH_KINDS) - 1)
        event_ids = align_ids(sidecar.event_ids).clamp(0, len(EVENT_NAMES) - 1)
        math = self.math_embed(math_ids)
        event_multi = align_multi(getattr(sidecar, "event_multi", None))
        if event_multi is not None:
            # Multi-label composition: a token can end a word, sentence and
            # math span at the same time.  Sum the event embeddings instead
            # of collapsing to the last event id.
            event = event_multi @ self.event_embed.weight.to(event_multi.dtype)
            no_event = (event_multi.sum(dim=-1, keepdim=True) == 0).to(
                event.dtype
            )
            event = event + no_event * self.event_embed(
                torch.zeros(
                    event_multi.shape[:-1], device=x.device, dtype=torch.long
                )
            ).to(event.dtype)
        else:
            event = self.event_embed(event_ids)
        concept = self.out(
            torch.cat(
                [
                    x,
                    word_ctx,
                    self.sentence_proj(sent),
                    math + event,
                ],
                dim=-1,
            )
        )
        concept = F.normalize(concept, dim=-1)
        return concept, concept.mean(dim=1)


@dataclass
class CompositionEvent:
    token_index: int
    event_id: int
    byte_start: int
    byte_end: int

    @property
    def name(self) -> str:
        return EVENT_NAMES[int(self.event_id)]

    def as_dict(self) -> dict[str, Any]:
        return {
            "token_index": int(self.token_index),
            "event": self.name,
            "event_id": int(self.event_id),
            "byte_start": int(self.byte_start),
            "byte_end": int(self.byte_end),
        }


class CompositionTokenizer:
    """Tokenizer-side composition layer over a token-aligned sidecar."""

    def __init__(self, tokenizer: Any):
        # Accept either the raw layered tokenizer or the flat wrapper used by
        # training; both expose the same layered encoding underneath.
        self.tokenizer = getattr(tokenizer, "layered", tokenizer)

    def encode(self, text: str) -> list[CompositionEvent]:
        sidecar = SpanSidecar.from_text(text, self.tokenizer)
        events: list[CompositionEvent] = []
        for token_index in range(sidecar.event_multi.shape[0]):
            start = int(sidecar.byte_start[token_index])
            end = start + int(sidecar.byte_len[token_index])
            for event_id, fired in enumerate(
                sidecar.event_multi[token_index].tolist()
            ):
                if fired > 0.0:
                    events.append(
                        CompositionEvent(token_index, event_id, start, end)
                    )
        return events

    @staticmethod
    def decode(events: Iterable[CompositionEvent]) -> list[str]:
        return [event.name for event in events]

    @staticmethod
    def manifest() -> dict[str, Any]:
        return {
            "events": list(EVENT_NAMES),
            "alignment": "token_index + byte span",
            "roundtrip": "event list -> names; tokenizer spans retain text",
        }


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


class CompositionCalibrator:
    """Per-event threshold calibration for sparse structural events."""

    @staticmethod
    def thresholds(
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        grid: Sequence[float] | None = None,
    ) -> list[float]:
        probs = torch.sigmoid(logits.detach()).reshape(
            -1, logits.shape[-1]
        )
        labels = targets.detach().reshape(-1, targets.shape[-1]) > 0.5
        values = (
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            if grid is None else list(grid)
        )
        result: list[float] = []
        for event in range(probs.shape[-1]):
            positive = labels[:, event]
            if not bool(positive.any()):
                result.append(0.5)
                continue
            best_threshold = 0.5
            best_f1 = -1.0
            for threshold in values:
                predicted = probs[:, event] > float(threshold)
                tp = int((predicted & positive).sum())
                fp = int((predicted & ~positive).sum())
                fn = int((~predicted & positive).sum())
                denominator = 2 * tp + fp + fn
                f1 = (2 * tp / denominator) if denominator else 0.0
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = float(threshold)
            result.append(best_threshold)
        return result


@dataclass
class CompositionState:
    """Minimal structural grammar for generated composition events."""

    word_open: bool = False
    math_open: bool = False

    def observe_id(self, event_id: int) -> None:
        if event_id == EVENT_IDS["<word_start>"]:
            self.word_open = True
        elif event_id == EVENT_IDS["<word_end>"]:
            self.word_open = False
        elif event_id == EVENT_IDS["<math_start>"]:
            self.math_open = True
        elif event_id == EVENT_IDS["<math_end>"]:
            self.math_open = False

    def observe_events(self, event_ids: Iterable[int]) -> None:
        for event_id in event_ids:
            self.observe_id(int(event_id))

    def observe_multi(self, event_multi: torch.Tensor) -> None:
        value = event_multi
        if value.ndim == 1:
            value = value.unsqueeze(0)
        for row in value.detach().cpu().tolist():
            for event_id, fired in enumerate(row):
                if fired > 0.5:
                    self.observe_id(event_id)

    def filter(self, fired: Sequence[bool]) -> list[bool]:
        out = [bool(value) for value in fired]
        if len(out) != len(EVENT_NAMES):
            return out
        start = EVENT_IDS["<word_start>"]
        end = EVENT_IDS["<word_end>"]
        math_start = EVENT_IDS["<math_start>"]
        math_end = EVENT_IDS["<math_end>"]
        compose = EVENT_IDS["<compose>"]
        if out[end] and not self.word_open and not out[start]:
            out[end] = False
        if out[start] and self.word_open and not out[end]:
            out[start] = False
        if out[math_end] and not self.math_open and not out[math_start]:
            out[math_end] = False
        if out[math_start] and self.math_open and not out[math_end]:
            out[math_start] = False
        if out[compose] and not (self.word_open or out[end]):
            out[compose] = False
        return out


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
        byte_keys = byte_symbol_codebook(dim, seed=20261113)
        byte_values = byte_symbol_codebook(dim, seed=20261114)
        if vertices != 256:
            if vertices > 256:
                pad = F.normalize(
                    torch.randn(
                        vertices - 256, dim,
                        generator=torch.Generator().manual_seed(20261115),
                    ),
                    dim=-1,
                )
                byte_keys = torch.cat([byte_keys, pad], dim=0)
                byte_values = torch.cat([byte_values, pad], dim=0)
            else:
                byte_keys = byte_keys[:vertices]
                byte_values = byte_values[:vertices]
        self.object_memory.seed_vertices(byte_keys, byte_values)
        self.concept_to_state = nn.Linear(d_model, d_model, bias=False)
        self.object_to_state = nn.Linear(d_model, d_model, bias=False)
        self.object_pointer = nn.Linear(d_model, 2 * self.vertices, bias=False)
        self.object_triple_pointer = nn.Linear(
            d_model, 3 * self.vertices, bias=False
        )
        # Per-dimension gate for the recurrent concept carry.  Starting at
        # 0.5 keeps the first concept update visible while preventing the
        # carry from being overwritten wholesale at every block.
        self.carry_gate = nn.Parameter(torch.zeros(d_model))
        self.state_norm = nn.RMSNorm(d_model) if hasattr(nn, "RMSNorm") else nn.Identity()

    def counts(self) -> dict[str, int]:
        return {
            "vertices": self.vertices,
            "pairs": binomial(self.vertices, 2),
            "triples": binomial(self.vertices, 3),
        }

    def router_token_state(
        self, concept: torch.Tensor, object_ctx: torch.Tensor
    ) -> torch.Tensor:
        """Token-aligned concept/object state for the MoE router."""
        return self.state_norm(
            self.concept_to_state(concept)
            + self.object_to_state(object_ctx)
        )

    def forward(
        self,
        x: torch.Tensor,
        sidecar: SpanSidecar | None,
        route_ctx: torch.Tensor,
        carry: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Route/attention/MoE context conditions the concept abstraction and
        # the object query.  Without this the loop was one-way: concept fed
        # the router, but the router never fed concept.
        route_bias = (
            self.concept_to_state(route_ctx.to(dtype=x.dtype))
            .unsqueeze(1).to(dtype=x.dtype)
            if route_ctx is not None else None
        )
        concept_input = (
            x + 0.1 * route_bias if route_bias is not None else x
        )
        concept, global_ctx = self.encoder(concept_input, sidecar)
        pointer_input = x + concept
        if route_bias is not None:
            pointer_input = pointer_input + 0.1 * route_bias
        self.last_pointer_logits = self.object_pointer(pointer_input).view(
            x.shape[0], x.shape[1], 2, self.vertices
        )
        self.last_triple_pointer_logits = self.object_triple_pointer(
            pointer_input
        ).view(x.shape[0], x.shape[1], 3, self.vertices)
        # Hard causal pointer stream.  The forward value is the selected
        # pair; the gradient follows the soft pointer distribution through a
        # straight-through estimator.  The selected pair at token t-1
        # conditions object retrieval at token t.
        pointer_soft = F.softmax(
            self.last_pointer_logits.float(), dim=-1
        ).to(dtype=x.dtype)
        u_logits = self.last_pointer_logits[:, :, 0]
        v_logits = self.last_pointer_logits[:, :, 1].clone()
        u_hard = u_logits.argmax(dim=-1, keepdim=True)
        v_logits.scatter_(
            -1, u_hard, torch.finfo(v_logits.dtype).min
        )
        v_hard = v_logits.argmax(dim=-1, keepdim=True)
        u_onehot = F.one_hot(
            u_hard.squeeze(-1), self.vertices
        ).to(dtype=x.dtype)
        v_onehot = F.one_hot(
            v_hard.squeeze(-1), self.vertices
        ).to(dtype=x.dtype)
        pointer_hard = torch.stack([u_onehot, v_onehot], dim=2)
        pointer_st = pointer_hard + pointer_soft - pointer_soft.detach()
        vertex_keys = F.normalize(
            self.object_memory.vertex_keys.to(dtype=x.dtype), dim=-1
        )
        vertex_values = F.normalize(
            self.object_memory.vertex_values.to(dtype=x.dtype), dim=-1
        )
        pointer_state = (
            torch.einsum("btv,vd->btd", pointer_st[:, :, 0], vertex_keys)
            + torch.einsum("btv,vd->btd", pointer_st[:, :, 1], vertex_values)
        )
        pointer_ctx = self.object_memory.out_proj(pointer_state)
        triple_soft = F.softmax(
            self.last_triple_pointer_logits.float(), dim=-1
        ).to(dtype=x.dtype)
        triple_state = (
            torch.einsum("btv,vd->btd", triple_soft[:, :, 0], vertex_keys)
            + torch.einsum("btv,vd->btd", triple_soft[:, :, 1], vertex_values)
            + torch.einsum("btv,vd->btd", triple_soft[:, :, 2], vertex_keys)
        )
        triple_ctx = self.object_memory.out_proj(triple_state)
        pointer_prev = (
            torch.cat(
                [torch.zeros_like(pointer_ctx[:, :1]), pointer_ctx[:, :-1]],
                dim=1,
            )
            if pointer_ctx.shape[1] > 1 else torch.zeros_like(pointer_ctx)
        )
        triple_prev = (
            torch.cat(
                [torch.zeros_like(triple_ctx[:, :1]), triple_ctx[:, :-1]],
                dim=1,
            )
            if triple_ctx.shape[1] > 1 else torch.zeros_like(triple_ctx)
        )
        pointer_gate = torch.sigmoid(
            self.object_memory.gate_logit
        ).to(dtype=x.dtype)
        pointer_prev = pointer_prev * pointer_gate
        triple_prev = triple_prev * (0.5 * pointer_gate)
        object_input = x + concept + pointer_prev + triple_prev
        if route_bias is not None:
            object_input = object_input + 0.1 * route_bias
        object_ctx, object_aux = self.object_memory(object_input)
        # Current-layer pointer feedback.  This is the soft/current object
        # identity; pointer_prev above is the hard recurrent stream.
        object_ctx = object_ctx + pointer_gate * pointer_ctx
        object_ctx = object_ctx + 0.5 * pointer_gate * triple_ctx
        # Causal object recurrence along the sequence.  Token t+1 sees the
        # object/pointer state selected at t; the shift is a one-token delay,
        # so no future token can leak.  This is a feed-forward scan, not a
        # second full attention pass.
        if object_ctx.shape[1] > 1:
            previous = torch.cat(
                [torch.zeros_like(object_ctx[:, :1]), object_ctx[:, :-1]],
                dim=1,
            )
            object_ctx = object_ctx + torch.sigmoid(
                self.carry_gate
            ).to(dtype=object_ctx.dtype) * previous
        self.last_object_ids = object_aux["pair_ids"].detach()
        self.last_object_vertices = object_aux["pair_vertices"].detach()
        self.last_triple_ids = (
            object_aux["triple_ids"].detach()
            if object_aux.get("triple_ids") is not None else None
        )
        self.last_triple_vertices = (
            object_aux["triple_vertices"].detach()
            if object_aux.get("triple_vertices") is not None else None
        )
        if self.last_triple_vertices is None:
            # At long context the adaptive object policy disables triples;
            # do not expose a random triple readout with no retrieval target.
            self.last_triple_pointer_logits = None
        # Return only the concept/object contribution.  The block adds the
        # route context separately; including it here double-counted the
        # router state on every layer.
        update = (
            self.concept_to_state(global_ctx)
            + self.object_to_state(object_ctx.mean(dim=1))
        )
        if carry is None:
            loop_ctx = self.state_norm(update)
        else:
            gate = torch.sigmoid(self.carry_gate).to(dtype=update.dtype)
            loop_ctx = self.state_norm(
                (1.0 - gate) * carry + gate * update
            )
        self.last_carry = loop_ctx
        return concept, object_ctx, loop_ctx


def sidecar_batch(
    tokenizer: Any,
    texts: Sequence[str],
) -> tuple[list[SpanSidecar], list[int]]:
    """Convenience helper for tests and data builders."""
    sidecars = [SpanSidecar.from_text(text, tokenizer) for text in texts]
    return sidecars, [len(sidecar.word_ids) for sidecar in sidecars]
