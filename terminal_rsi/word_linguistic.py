#!/usr/bin/env python3
"""Rule-based linguistic candidate generators for word-level prediction.

This module implements the W4.1-W4.4 ideas from ``WORD_TOP5_CHECKLIST.md``:
a coarse causal tagger, a construction/skeleton phrase inventory, a centering
cache for entities/pronouns, and a dependency-free English morphology
generator.  ``LinguisticPrior`` fuses them into either a 4096-dim score vector
or a sparse ranked candidate list.

Everything is causal: a token only influences a prediction after it has been
observed.  There is no parser, no WordNet, and no learned model here.  The
methods are cheap symbolic approximations intended to add *candidates* and
small reranker features, not to replace the n-gram/head priors.

Approximations and expected contribution
----------------------------------------
``SimpleTagger``
    Closed-class word lists plus English suffix rules and a small irregular
    verb/adjective list.  It has no full morphological analyser and its
    accuracy on garden-path sentences is limited.  Expected contribution:
    gating and a broad P(word | tag) candidate source; useful at determiner,
    pronoun and auxiliary positions.

``ConstructionInventory``
    Mines 2-6 word contexts (phrase of 2-6 words plus successor) from the word
    stream.  A second, typed skeleton index replaces open-class words with
    ``<NOUN>``/``<VERB>``/``<ADJ>``/``<NUM>``/``<OTHER>`` slots, and an
    untyped ``<X>`` index generalises further.  This is not a semantic
    construction grammar: slots are POS-typed only, and skeletons are mined
    from adjacent tokens (sentence boundaries are only approximated by
    punctuation).  Expected contribution: high-precision candidates when a
    closed-class frame repeats with novel open-class fillers.  Build can be
    memory heavy for large corpora; use ``min_count=2`` and a smaller
    ``max_order`` there.

``CenteringCache``
    Heuristic centering: salience-decayed entity mentions, capitalized-name
    spans, and pronoun matching by number/gender.  There is no anaphora
    resolution or syntactic subject detection, only positional/gender/number
    cues.  Expected contribution: entity/name candidates at pronoun and
    determiner positions, plus name continuations; typically only a few
    percent of positions, but high value there.

``MorphologyCandidates``
    Suffix-stripping lemmas and rule-based inflection generation.  It can
    overgenerate (e.g. the ``-ed`` stem of ``hoped`` is ambiguous between
    ``hop`` and ``hope``) and does not know argument structure or derivational
    semantics.  Expected contribution: small-weight candidates for inflected
    forms, OOV surface variants, and lemma backoff.

Fusion
    Source scores are mixed with fixed weights (construction first, then
    centering/POS, then morphology) and renormalised.  Treat the result as a
    candidate *union* expert.  It is deliberately conservative: in particular
    unknown word ID 0 is excluded from generated candidates.

Integration with ``word_rerank_fast.py``
---------------------------------------
The source API matches the cache's ``sources_at`` shape::

    prior = LinguisticPrior(vocab_size=4096).build(train_words,
                                                  word_to_id=vocab)
    extra = prior.sources(history_words, top_k=100)
    # {name: [word_ids, ...]} for construction/pos/centering/morphology

``prior.query(history)`` additionally returns a 4096-vector, sparse ranked
IDs, per-source ranks and a construction match.  ``candidate_features`` emits
rank/score features that can be appended to the listwise reranker rows.  No
shared module is imported or modified.
"""
from __future__ import annotations

import heapq
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

DEFAULT_VOCAB_SIZE = 4096
UNK_ID = 0
TAGS = ("DET", "ADJ", "NOUN", "VERB", "PREP", "PRON", "CONJ", "NUM",
        "PUNCT", "OTHER")
TAG_INDEX = {tag: index for index, tag in enumerate(TAGS)}

_PUNCT_STRIP = "\"'`“”‘’()[]{}<>,.;:!?…"
_QUOTE_STRIP = "\"'`“”‘’()[]{}<>"
_WORD_ONLY = re.compile(r"^[A-Za-z]+(?:['’\-][A-Za-z]+)*$")
_NUMBER_RE = re.compile(r"^[+-]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][+-]?\d+)?%?$")

_NUMBER_WORDS = frozenset("""
zero one two three four five six seven eight nine ten eleven twelve thirteen
fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty
sixty seventy eighty ninety hundred thousand million billion trillion dozen
""".split())
_ORDINAL_WORDS = frozenset("""
first second third fourth fifth sixth seventh eighth ninth tenth eleventh
twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth
nineteenth twentieth thirtieth fortieth fiftieth sixtieth seventieth eightieth
ninetieth hundredth thousandth millionth
""".split())

_DET = frozenset("""
the a an this that these those some any every each no another both either
neither all few many much more most several such my your his her its our their
""".split())
_PRON = frozenset("""
i me you he him she it we us they them who whom whose which what whatever
whoever something anything nothing everything someone anyone everyone nobody
myself yourself himself herself itself ourselves yourselves themselves
mine yours hers ours theirs
""".split())
_PREP = frozenset("""
of in on at to for with from by about as into onto over under between among
through during before after above below against toward towards upon within
without across behind beyond near off per via despite except inside outside
since than like around along past plus minus versus unto amid beneath beside
besides throughout till
""".split())
_CONJ = frozenset("""
and or but nor so yet because although though while whereas if unless until
when where whether since after before as than that
""".split())
_PREP_CONJ = _PREP & _CONJ

_POSSESSIVE = frozenset("my your his her its our their".split())

_VERB_FORMS = frozenset("""
am is are was were be been being have has had having do does did done doing
can could will would shall should may might must ought
go goes went gone going make makes made making take takes took taken taking
come comes came coming see sees saw seen seeing know knows knew known knowing
get gets got gotten getting give gives gave given giving think thinks thought
thinking say says said saying tell tells told telling find finds found finding
feel feels felt feeling leave leaves left leaving put puts putting bring brings
brought bringing buy buys bought buying eat eats ate eaten eating run runs ran
running write writes wrote written writing read reads reading speak speaks spoke
spoken speaking meet meets met meeting pay pays paid paying sit sits sat sitting
stand stands stood standing understand understands understood understanding
hold holds held holding win wins won winning lose loses lost losing send sends
sent sending build builds built building become becomes became becoming begin
begins began begun beginning keep keeps kept keeping hear hears heard hearing
let lets letting mean means meant meaning set sets setting show shows showed
shown showing rise rises rose risen rising fall falls fell fallen falling grow
grows grew grown growing draw draws drew drawn drawing sleep sleeps slept
sleeping teach teaches taught teaching catch catches caught catching fight
fights fought fighting seek seeks sought seeking sell sells sold selling spend
spends spent spending wear wears wore worn wearing choose chooses chose chosen
choosing drive drives drove driven driving break breaks broke broken breaking
fly flies flew flown flying forget forgets forgot forgotten forgetting lie lies
lay lain lying die dies died dying
""".split())

# Small open-class exceptions so simple test sentences do not depend on a
# full lexicon.  The tagger is still primarily closed-class + suffix rules.
_ADJ_WORDS = frozenset("""
good bad big small large little old new young long short high low red blue green
black white brown yellow grey gray happy sad quick slow fast early late hard
easy right wrong same different other few many much more most some any all
first second last next own only best better worse great important possible
likely able sure certain true false full empty open closed free public private
real local social human main major minor common special strong weak dark light
hot cold warm cool beautiful ugly rich poor safe dangerous tall heavy tiny huge
bright quiet loud clean dirty friendly lovely lonely ugly silly likely daily
weekly monthly yearly wooden
""".split())
_NOUN_Y = frozenset("""
city story country family company money body party study history agency policy
energy memory theory quality activity community security ability university
library strategy economy industry technology
""".split())
_NOUN_ED = frozenset("""
bed shed seed need speed feed weed breed creed deed sled
""".split())

_ALL_CLOSED = _DET | _PRON | _PREP | _CONJ
_FUNCTION_WORDS = _ALL_CLOSED | frozenset("""
not no nor so too very just also then there here now only even still yet
""".split())
_COMMON_CAPS = frozenset("""
the a an this that these those i it he she they we you his her its our their
there here when where why how what who whom whose which and but or if because
although though while whereas for as at in on of to from by with chapter part
book section one two three four five six seven eight nine ten mr mrs ms dr sir
lord lady professor captain
""".split())


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def strip_token(word: str) -> str:
    """Strip surrounding punctuation, keeping internal apostrophes/hyphens."""
    return str(word).strip(_PUNCT_STRIP)


def word_core(word: str) -> str:
    """Lower-case form used for lexicon and suffix tests."""
    return strip_token(word).lower()


def _display(word: str) -> str:
    return str(word).strip(_PUNCT_STRIP)


def _is_punct_token(word: str) -> bool:
    return all(not ch.isalnum() for ch in str(word))


def is_sentence_end(word: str) -> bool:
    """Approximate sentence end.  ``Mr.`` is a known false positive."""
    token = str(word).strip().strip(_QUOTE_STRIP)
    return bool(token) and token[-1] in ".!?"


def _is_number_word(core: str) -> bool:
    if not core:
        return False
    if _NUMBER_RE.match(core):
        return True
    if re.match(r"^\d+(?:st|nd|rd|th)$", core):
        return True
    return core in _NUMBER_WORDS or core in _ORDINAL_WORDS


def _is_plural_noun(word: str) -> bool:
    core = word_core(word)
    return (len(core) > 2 and core.endswith("s")
            and not core.endswith(("ss", "us", "is", "ous")))


# ---------------------------------------------------------------------------
# 1. Coarse causal tagger
# ---------------------------------------------------------------------------

class SimpleTagger:
    """A tiny causal POS tagger over the coarse tag set ``TAGS``.

    It is a lexicon-plus-suffix tagger, not a statistical tagger.  ``tag`` is
    stateful and only uses the previous tag/word; ``tag_sequence`` resets the
    state first.  Punctuation is tagged ``PUNCT`` and also resets the
    sentence-start flag for the following token.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._prev_tag: str | None = None
        self._prev_core = ""
        self._sentence_start = True

    def tag_sequence(self, words: Iterable[str]) -> list[str]:
        if isinstance(words, str):
            words = words.split()
        self.reset()
        return [self.tag(word) for word in words]

    def tag(self, word: str) -> str:
        core = word_core(word)
        tag = self._classify(word, core, self._prev_tag)
        self._sentence_start = is_sentence_end(word)
        self._prev_tag = tag
        self._prev_core = core
        return tag

    def _classify(self, word: str, core: str,
                  prev: str | None) -> str:
        if _is_punct_token(word):
            return "PUNCT"
        if _is_number_word(core):
            return "NUM"

        # "that" is a determiner, complementizer, or pronoun depending on the
        # previous tag.  This is the only context-sensitive closed-class word
        # that matters frequently in prose.
        if core == "that":
            return "CONJ" if prev in ("VERB", "CONJ") else "DET"
        if core == "her":
            return "PRON" if prev in ("VERB", "PREP") else "DET"
        if core in _DET:
            return "DET"
        if core in _PRON:
            return "PRON"

        if core in _VERB_FORMS:
            return "VERB"
        if core in _ADJ_WORDS:
            return "ADJ"

        if core in _CONJ and core not in _PREP:
            return "CONJ"
        if core in _PREP:
            if core in _CONJ:
                # "after the rain" (PREP) vs "after we left" (CONJ).
                return ("PREP" if prev in
                        ("VERB", "NOUN", "ADJ", "PRON", "DET", "NUM",
                         "PREP") else "CONJ")
            return "PREP"
        if core in _CONJ:
            return "CONJ"

        # Inflectional suffixes.  Previous tag resolves the worst ambiguity
        # (plural noun vs. 3sg verb, participle vs. adjective).
        if core.endswith("ing") and len(core) > 4:
            if prev in ("DET", "ADJ", "NUM"):
                return "NOUN"
            return "VERB"
        if core.endswith("ed") and len(core) > 3:
            if prev in ("DET", "ADJ", "NUM"):
                return "ADJ"
            return "VERB"
        if core.endswith("ies") and len(core) > 4:
            return "VERB" if prev in ("PRON", "NOUN") else "NOUN"
        if core.endswith("es") and len(core) > 3:
            return "VERB" if prev in ("PRON", "NOUN") else "NOUN"
        if (core.endswith("s") and len(core) > 2
                and not core.endswith(("ss", "us", "is"))):
            return "VERB" if prev in ("PRON", "NOUN") else "NOUN"

        if core.endswith("ly") and len(core) > 3:
            return "OTHER"  # no ADV tag in the coarse set

        for suffix in ("ous", "ful", "less", "able", "ible", "ive",
                       "ical", "ic", "ish", "ant", "ent"):
            if core.endswith(suffix) and len(core) > len(suffix) + 2:
                return "ADJ"
        if core.endswith("y") and len(core) > 3 and core not in _NOUN_Y:
            return "ADJ"

        for suffix in ("tion", "sion", "ment", "ness", "ity", "ance",
                       "ence", "ship", "hood", "ism", "ist", "ology",
                       "graphy", "ure", "age", "dom", "th", "ee", "eer",
                       "ian"):
            if core.endswith(suffix) and len(core) > len(suffix) + 2:
                return "NOUN"

        if core.endswith(("er", "or")) and len(core) > 4:
            if prev in ("is", "are", "was", "were", "be", "been", "more",
                        "most", "than"):
                return "ADJ"
            if prev == "VERB":
                return "ADJ"
            return "NOUN"
        if core.endswith("est") and len(core) > 4:
            return "ADJ"
        if core.endswith(("ize", "ise", "ify")) and len(core) > 4:
            return "VERB"

        # Unknown-word fallbacks.  Capitalised words are treated as proper
        # nouns, and DET/ADJ/NUM/PREP contexts strongly imply a noun.
        if word and _display(word)[:1].isupper():
            return "NOUN"
        if prev in ("DET", "ADJ", "NUM", "PREP"):
            return "NOUN"
        if prev in ("PRON", "NOUN"):
            return "VERB"
        return "OTHER"


# ---------------------------------------------------------------------------
# 2. Construction inventory
# ---------------------------------------------------------------------------

_SLOT_CODES = {"NOUN": -1, "VERB": -2, "ADJ": -3, "NUM": -4, "OTHER": -5}
_SLOT_NAMES = {-1: "NOUN", -2: "VERB", -3: "ADJ", -4: "NUM", -5: "OTHER"}
_UNTYPED_SLOT = -6
_OPEN_CLASS = frozenset(_SLOT_CODES)


def _typed_key(ids: Sequence[int],
               tags: Sequence[str]) -> tuple[tuple[int, ...], bool]:
    key: list[int] = []
    has_slot = False
    for word_id, tag in zip(ids, tags):
        code = _SLOT_CODES.get(tag)
        if code is None:
            key.append(int(word_id))
        else:
            key.append(code)
            has_slot = True
    return tuple(key), has_slot


def _untyped_key(ids: Sequence[int],
                 tags: Sequence[str]) -> tuple[tuple[int, ...], bool]:
    key: list[int] = []
    has_slot = False
    for word_id, tag in zip(ids, tags):
        if tag in _OPEN_CLASS:
            key.append(_UNTYPED_SLOT)
            has_slot = True
        else:
            key.append(int(word_id))
    return tuple(key), has_slot


@dataclass(frozen=True)
class ConstructionMatch:
    """Longest matching construction and its next-word distribution."""

    order: int
    kind: str                       # exact | typed | untyped
    skeleton: tuple[str, ...]
    slot_types: tuple[str, ...]     # aligned, "" for literal positions
    next_ids: tuple[int, ...]
    probs: tuple[float, ...]
    support: int

    def distribution(self) -> dict[int, float]:
        return dict(zip(self.next_ids, self.probs))

    @property
    def slots(self) -> tuple[str, ...]:
        """Alias for ``slot_types`` (typed slot names, aligned)."""
        return self.slot_types

    @property
    def untyped_skeleton(self) -> tuple[str, ...]:
        """Skeleton with every typed slot collapsed to ``<X>``."""
        return tuple("<X>" if slot else token
                     for token, slot in zip(self.skeleton, self.slot_types))

    @property
    def typed_skeleton(self) -> tuple[str, ...]:
        """Alias for ``skeleton`` (typed slots preserved)."""
        return self.skeleton

    @property
    def slot_type_list(self) -> tuple[str, ...]:
        """Only the non-empty slot types, in left-to-right order."""
        return tuple(slot for slot in self.slot_types if slot)

    def top_k(self, k: int = 10) -> list[tuple[int, float]]:
        pairs = list(zip(self.next_ids, self.probs))[:max(int(k), 0)]
        total = sum(prob for _word, prob in pairs)
        if total <= 0.0:
            return []
        return [(word, prob / total) for word, prob in pairs]


class ConstructionInventory:
    """Mined exact 2-6-gram and typed/untyped skeleton constructions.

    Parameters
    ----------
    max_order:
        Maximum context length (2-6 in the intended use).
    min_count:
        Contexts seen fewer times are discarded after the counting pass.
    max_next:
        Maximum successors stored per context.
    max_contexts:
        Safety valve for large builds; the most frequent contexts per order
        are retained if the counting table exceeds this size.
    backoff_decay:
        Weight decay per order when ``next_distribution(backoff=True)``.
    """

    def __init__(self, max_order: int = 6, min_count: int = 1,
                 max_next: int = 12, max_contexts: int = 1_000_000,
                 backoff_decay: float = 0.5) -> None:
        if int(max_order) < 2:
            raise ValueError("max_order must be at least 2")
        self.max_order = int(max_order)
        self.min_count = max(1, int(min_count))
        self.max_next = max(1, int(max_next))
        self.max_contexts = None if max_contexts is None else int(max_contexts)
        self.backoff_decay = float(backoff_decay)
        self.reset()

    def reset(self) -> None:
        self.word_to_id: dict[str, int] = {}
        self.id_to_word: dict[int, str] = {}
        self.vocab_size = DEFAULT_VOCAB_SIZE
        self._canonical: dict[int, int] = {}
        self._canonical_text: dict[int, str] = {}
        self._exact: dict[int, dict[tuple[int, ...],
                                    tuple[tuple[int, ...],
                                          tuple[float, ...], int]]] = {}
        self._typed: dict[int, dict] = {}
        self._untyped: dict[int, dict] = {}
        self.order_contexts: dict[int, int] = {}
        self._built = False

    # -- construction ------------------------------------------------------

    def build(self, word_stream: Iterable[str | int], *,
              word_to_id: dict[str, int] | None = None,
              id_to_word: dict[int, str] | Sequence[str] | None = None,
              vocab_size: int = DEFAULT_VOCAB_SIZE,
              tags: Sequence[str] | None = None) -> "ConstructionInventory":
        ids, surfaces, id_to_word, word_to_id, vocab_size = _prepare_stream(
            word_stream, word_to_id=word_to_id, id_to_word=id_to_word,
            vocab_size=vocab_size)
        if tags is None and any(surfaces):
            tags = SimpleTagger().tag_sequence(surfaces)
        self.build_from_arrays(ids, tags=tags, id_to_word=id_to_word,
                               word_to_id=word_to_id, vocab_size=vocab_size)
        return self

    def build_from_arrays(self, ids: Sequence[int], *,
                          tags: Sequence[str] | None = None,
                          id_to_word: dict[int, str] | None = None,
                          word_to_id: dict[str, int] | None = None,
                          vocab_size: int = DEFAULT_VOCAB_SIZE
                          ) -> "ConstructionInventory":
        self.reset()
        ids = [int(word) for word in ids]
        self.id_to_word = dict(id_to_word or {})
        self.word_to_id = dict(word_to_id or {})
        self.vocab_size = max(int(vocab_size),
                              (max(ids) + 1) if ids else 1)
        self._build_canonical()
        canonical_ids = [self._canonical.get(word, word) for word in ids]
        if tags is not None and len(tags) != len(ids):
            tags = None
        orders = range(2, self.max_order + 1)

        exact_counts: dict[int, dict[tuple[int, ...], int]] = {
            order: {} for order in orders}
        typed_counts: dict[int, dict[tuple[int, ...], int]] = {
            order: {} for order in orders}
        untyped_counts: dict[int, dict[tuple[int, ...], int]] = {
            order: {} for order in orders}

        total = len(ids)
        for index in range(total):
            last_n = min(self.max_order, total - index - 1)
            for order in range(2, last_n + 1):
                context = tuple(canonical_ids[index:index + order])
                exact_counts[order][context] = (
                    exact_counts[order].get(context, 0) + 1)
                if tags is not None:
                    context_tags = tags[index:index + order]
                    typed, has_typed = _typed_key(context, context_tags)
                    if has_typed:
                        typed_counts[order][typed] = (
                            typed_counts[order].get(typed, 0) + 1)
                        untyped, has_untyped = _untyped_key(
                            context, context_tags)
                        if has_untyped:
                            untyped_counts[order][untyped] = (
                                untyped_counts[order].get(untyped, 0) + 1)

        for order in orders:
            exact_counts[order] = self._prune_counts(exact_counts[order])
            typed_counts[order] = self._prune_counts(typed_counts[order])
            untyped_counts[order] = self._prune_counts(untyped_counts[order])

        exact: dict[int, dict[tuple[int, ...], Counter[int]]] = {
            order: {} for order in orders}
        typed: dict[int, dict[tuple[int, ...], Counter[int]]] = {
            order: {} for order in orders}
        untyped: dict[int, dict[tuple[int, ...], Counter[int]]] = {
            order: {} for order in orders}

        for index in range(total):
            last_n = min(self.max_order, total - index - 1)
            for order in range(2, last_n + 1):
                context = tuple(canonical_ids[index:index + order])
                successor = ids[index + order]
                if context in exact_counts[order]:
                    _bump_count(exact[order], context, successor,
                                self.max_next)
                if tags is None:
                    continue
                context_tags = tags[index:index + order]
                typed_key, has_typed = _typed_key(context, context_tags)
                if has_typed and typed_key in typed_counts[order]:
                    _bump_count(typed[order], typed_key, successor,
                                self.max_next)
                untyped_key, has_untyped = _untyped_key(context, context_tags)
                if has_untyped and untyped_key in untyped_counts[order]:
                    _bump_count(untyped[order], untyped_key, successor,
                                self.max_next)

        for order in orders:
            self._exact[order] = self._finalise_rows(
                exact[order], exact_counts[order])
            self._typed[order] = self._finalise_rows(
                typed[order], typed_counts[order])
            self._untyped[order] = self._finalise_rows(
                untyped[order], untyped_counts[order])
            self.order_contexts[order] = len(self._exact[order])
        self._built = True
        return self

    def _build_canonical(self) -> None:
        """Map case/orthography variants to a representative context ID.

        Contexts are matched case-insensitively (so "The old man" and
        "the old man" share a construction) while successors are stored as
        raw vocabulary IDs.  This is an approximation: capitalization is
        useful for names, but splitting every sentence-initial word into a
        separate context costs far more recall than it buys.
        """
        by_core: dict[str, Counter[int]] = defaultdict(Counter)
        for raw, surface in self.id_to_word.items():
            core = word_core(surface)
            if core:
                by_core[core][int(raw)] += 1
        canonical: dict[int, int] = {}
        canonical_text: dict[int, str] = {}
        for core, counts in by_core.items():
            # Prefer a raw ID whose surface is already lower-case (the usual
            # query form), then the most frequent spelling, then the lowest
            # ID for determinism.
            def preference(item: tuple[int, int]) -> tuple[int, int, int]:
                raw, count = item
                surface = self.id_to_word.get(raw, "")
                is_lower = 1 if surface == surface.lower() else 0
                return (is_lower, count, -raw)
            representative = max(counts.items(), key=preference)[0]
            canonical_text[representative] = core
            for raw in counts:
                canonical[int(raw)] = int(representative)
        for raw in self.id_to_word:
            canonical.setdefault(int(raw), int(raw))
            canonical_text.setdefault(int(raw),
                                      self.id_to_word.get(int(raw), ""))
        self._canonical = canonical
        self._canonical_text = canonical_text

    def _prune_counts(
            self, table: dict[tuple[int, ...], int]
    ) -> dict[tuple[int, ...], int]:
        if self.min_count > 1:
            table = {key: count for key, count in table.items()
                     if count >= self.min_count}
        if (self.max_contexts is None
                or len(table) <= self.max_contexts):
            return table
        kept = heapq.nlargest(self.max_contexts, table.items(),
                              key=lambda item: item[1])
        return dict(kept)

    def _finalise_rows(
            self,
            rows: dict[tuple[int, ...], Counter[int]],
            counts: dict[tuple[int, ...], int],
    ) -> dict[tuple[int, ...], tuple[tuple[int, ...],
                                     tuple[float, ...], int]]:
        out: dict[tuple[int, ...], tuple[tuple[int, ...],
                                          tuple[float, ...], int]] = {}
        for context, row in rows.items():
            best = sorted(row.items(), key=lambda item: (-item[1],
                                                         item[0])
                          )[:self.max_next]
            kept_total = sum(count for _word, count in best)
            if kept_total <= 0:
                continue
            words = tuple(int(word) for word, _count in best)
            probs = tuple(float(count) / kept_total
                          for _word, count in best)
            out[context] = (words, probs, int(counts.get(context,
                                                         kept_total)))
        return out

    # -- querying ----------------------------------------------------------

    def longest_match(
            self,
            history: Iterable[str | int] | str | None = None, *,
            ids: Sequence[int] | None = None,
            surfaces: Sequence[str] | None = None,
            tags: Sequence[str] | None = None,
    ) -> ConstructionMatch | None:
        """Return the longest matching suffix (exact > typed > untyped)."""
        if ids is None:
            ids, surfaces = self._encode_history(history or [])
        else:
            ids = [int(word) for word in ids]
            if surfaces is None:
                surfaces = [self.id_to_word.get(int(word), "")
                            for word in ids]
        if tags is None and surfaces is not None and any(surfaces):
            tags = SimpleTagger().tag_sequence(surfaces)
        for order in range(min(self.max_order, len(ids)), 1, -1):
            match = self._match_at_order(
                order, list(ids), surfaces, tags)
            if match is not None:
                return match
        return None

    def _match_at_order(
            self, order: int, ids: list[int],
            surfaces: Sequence[str] | None,
            tags: Sequence[str] | None,
    ) -> ConstructionMatch | None:
        context = tuple(self._canonical.get(word, word)
                        for word in ids[-order:])
        if context in self._exact.get(order, {}):
            words, probs, support = self._exact[order][context]
            skeleton = self._decode_key(context)
            return ConstructionMatch(
                order=order, kind="exact", skeleton=skeleton,
                slot_types=tuple("" for _ in context),
                next_ids=words, probs=probs, support=support)
        if tags is None or len(tags) < order:
            return None
        context_tags = tags[-order:]
        typed_key, has_typed = _typed_key(context, context_tags)
        if has_typed and typed_key in self._typed.get(order, {}):
            words, probs, support = self._typed[order][typed_key]
            return ConstructionMatch(
                order=order, kind="typed",
                skeleton=self._decode_key(typed_key),
                slot_types=self._slot_types(typed_key),
                next_ids=words, probs=probs, support=support)
        untyped_key, has_untyped = _untyped_key(context, context_tags)
        if has_untyped and untyped_key in self._untyped.get(order, {}):
            words, probs, support = self._untyped[order][untyped_key]
            return ConstructionMatch(
                order=order, kind="untyped",
                skeleton=self._decode_key(untyped_key, untyped=True),
                slot_types=tuple(
                    "X" if token < 0 else "" for token in untyped_key),
                next_ids=words, probs=probs, support=support)
        return None

    def query(self, history: Iterable[str | int] | str,
              top_k: int = 10) -> list[tuple[int, float]]:
        """Top-k successors of the longest matching suffix."""
        match = self.longest_match(history)
        return [] if match is None else match.top_k(top_k)

    def query_distribution(self, history: Iterable[str | int] | str
                           ) -> dict[int, float]:
        """Successor distribution of the longest matching suffix."""
        match = self.longest_match(history)
        return {} if match is None else match.distribution()

    def query_top_k(self, history: Iterable[str | int] | str,
                    top_k: int = 10) -> list[tuple[int, float]]:
        """Alias for ``query`` (explicit top-k name)."""
        return self.query(history, top_k=top_k)

    def next_distribution(
            self,
            history: Iterable[str | int] | str | None = None,
            top_k: int = 12,
            backoff: bool = True,
            *,
            ids: Sequence[int] | None = None,
            surfaces: Sequence[str] | None = None,
            tags: Sequence[str] | None = None,
    ) -> list[tuple[int, float]]:
        """Next-word top-k, optionally interpolating shorter suffixes.

        The longest match dominates; each shorter order is down-weighted by
        ``backoff_decay ** distance``.  Returned probabilities are normalised
        over the returned top-k.
        """
        if ids is None:
            ids, surfaces = self._encode_history(history or [])
        else:
            ids = [int(word) for word in ids]
            if surfaces is None:
                surfaces = [self.id_to_word.get(int(word), "")
                            for word in ids]
        if tags is None and surfaces is not None and any(surfaces):
            tags = SimpleTagger().tag_sequence(surfaces)
        if not backoff:
            match = self.longest_match(
                ids=ids, surfaces=surfaces, tags=tags)
            return [] if match is None else match.top_k(top_k)

        matches: list[ConstructionMatch] = []
        for order in range(min(self.max_order, len(ids)), 1, -1):
            match = self._match_at_order(order, list(ids), surfaces, tags)
            if match is not None:
                matches.append(match)
        if not matches:
            return []
        longest = matches[0].order
        combined: dict[int, float] = defaultdict(float)
        for match in matches:
            weight = self.backoff_decay ** (longest - match.order)
            for word, prob in zip(match.next_ids, match.probs):
                combined[int(word)] += weight * float(prob)
        ranked = sorted(combined.items(), key=lambda item: (-item[1],
                                                           item[0]))
        ranked = ranked[:max(int(top_k), 0)]
        total = sum(score for _word, score in ranked)
        if total <= 0.0:
            return []
        return [(word, score / total) for word, score in ranked]

    def _decode_key(self, key: Sequence[int],
                    untyped: bool = False) -> tuple[str, ...]:
        out: list[str] = []
        for token in key:
            if token == _UNTYPED_SLOT:
                out.append("<X>")
            elif token < 0:
                out.append(f"<{_SLOT_NAMES.get(token, 'X')}>")
            else:
                out.append(self._canonical_text.get(
                    int(token), self.id_to_word.get(int(token),
                                                    str(int(token)))))
        return tuple(out)

    @staticmethod
    def _slot_types(key: Sequence[int]) -> tuple[str, ...]:
        types: list[str] = []
        for token in key:
            if token < 0:
                types.append(_SLOT_NAMES.get(token, "X"))
            else:
                types.append("")
        return tuple(types)

    def _encode_history(
            self, history: Iterable[str | int] | str
    ) -> tuple[list[int], list[str]]:
        if isinstance(history, str):
            tokens: list[str | int] = history.split()
        else:
            tokens = list(history)
        ids: list[int] = []
        surfaces: list[str] = []
        for token in tokens:
            if isinstance(token, (int, np.integer)):
                word_id = int(token)
                surface = self.id_to_word.get(word_id, "")
            else:
                surface = str(token)
                word_id = self._lookup_id(surface)
            ids.append(word_id)
            surfaces.append(surface)
        return ids, surfaces

    def _lookup_id(self, surface: str) -> int:
        for candidate in (surface, _display(surface), word_core(surface),
                          _display(surface).capitalize()):
            if candidate in self.word_to_id:
                return int(self.word_to_id[candidate])
        return UNK_ID


# ---------------------------------------------------------------------------
# 3. Centering cache
# ---------------------------------------------------------------------------

_PRONOUN_INFO = {
    "he": ("m", "sg"), "him": ("m", "sg"), "his": ("m", "sg"),
    "himself": ("m", "sg"),
    "she": ("f", "sg"), "her": ("f", "sg"), "hers": ("f", "sg"),
    "herself": ("f", "sg"),
    "it": ("n", "sg"), "its": ("n", "sg"), "itself": ("n", "sg"),
    "they": (None, "pl"), "them": (None, "pl"), "their": (None, "pl"),
    "theirs": (None, "pl"), "themselves": (None, "pl"),
    "we": (None, "pl"), "us": (None, "pl"), "our": (None, "pl"),
    "ours": (None, "pl"), "ourselves": (None, "pl"),
    "i": (None, "sg"), "me": (None, "sg"), "my": (None, "sg"),
    "mine": (None, "sg"), "myself": (None, "sg"),
    "you": (None, "both"), "your": (None, "both"), "yours": (None, "both"),
    "yourself": (None, "both"), "yourselves": (None, "both"),
}
_FEMALE_NAMES = frozenset("""
alice mary jane susan karen nancy lisa betty margaret sandra ashley emily
emma sarah jessica laura helen anna julia maria elizabeth
""".split())
_MALE_NAMES = frozenset("""
john james robert michael william david richard joseph thomas charles
christopher daniel matthew anthony mark donald steven paul andrew joshua
kenneth kevin brian george edward ronald alice
""".split()) - {"alice"}
_FEMALE_TITLES = frozenset("mrs ms miss lady queen princess mother sister "
                           "daughter wife aunt".split())
_MALE_TITLES = frozenset("mr sir lord king prince father brother son husband "
                         "uncle".split())


@dataclass
class Entity:
    """One tracked discourse entity (approximate centering record)."""

    surface: str
    head: str
    word_id: int | None
    gender: str | None
    number: str
    score: float
    last_pos: int
    count: int
    tokens: tuple[str, ...] = ()


@dataclass(frozen=True)
class EntityCandidate:
    """A word-level candidate proposed by centering."""

    surface: str
    word_id: int | None
    score: float
    source: str
    kind: str                  # entity | pronoun | name_continuation
    gender: str | None = None
    number: str = "sg"


@dataclass
class CenteringState:
    entities: list[Entity]
    last_token: str
    last_tag: str
    last_pronoun: str | None
    tags: tuple[str, ...]


class CenteringCache:
    """Heuristic entity/pronoun tracking with recency and salience.

    Salience decays geometrically per token; subject-ish and repeated mentions
    receive a bonus.  Pronoun resolution is limited to number/gender
    compatibility.  No parser or coreference model is used.
    """

    def __init__(self, decay: float = 0.94, max_entities: int = 64,
                 pronoun_bonus: float = 0.5) -> None:
        self.decay = float(decay)
        self.max_entities = int(max_entities)
        self.pronoun_bonus = float(pronoun_bonus)
        self.word_to_id: dict[str, int] = {}
        self.id_to_word: dict[int, str] = {}
        self.name_lexicon: Counter[str] = Counter()
        self.name_display: dict[str, str] = {}
        self.name_pairs: dict[str, Counter[str]] = defaultdict(Counter)
        self._stream: list[str] = []
        self._tagger = SimpleTagger()

    # -- building the name lexicon ----------------------------------------

    def build(self, word_stream: Iterable[str | int], *,
              word_to_id: dict[str, int] | None = None,
              id_to_word: dict[int, str] | Sequence[str] | None = None,
              vocab_size: int = DEFAULT_VOCAB_SIZE) -> "CenteringCache":
        _ids, surfaces, id_to_word, word_to_id, _vocab = _prepare_stream(
            word_stream, word_to_id=word_to_id, id_to_word=id_to_word,
            vocab_size=vocab_size)
        self.word_to_id = dict(word_to_id)
        self.id_to_word = dict(id_to_word)
        self.build_from_words(surfaces)
        # Keep the observed stream so ``entities()`` and ``observed`` remain
        # useful after a batch build.  ``LinguisticPrior`` calls
        # ``build_from_words`` directly and therefore does not retain the
        # full training surface list twice.
        self._stream = list(surfaces)
        return self

    def build_from_words(self, words: Sequence[str]) -> "CenteringCache":
        self.name_lexicon.clear()
        self.name_display.clear()
        self.name_pairs.clear()
        if not words:
            return self
        tags = self._tagger.tag_sequence(words)
        flags = [self._name_like(words, tags, index)
                 for index in range(len(words))]
        for index, is_name in enumerate(flags):
            if not is_name:
                continue
            display = _display(words[index])
            core = display.lower()
            self.name_lexicon[core] += 1
            self.name_display.setdefault(core, display)
            if index + 1 < len(words) and flags[index + 1]:
                nxt = _display(words[index + 1])
                self.name_pairs[core][nxt] += 1
        return self

    def _name_like(self, words: Sequence[str], tags: Sequence[str],
                   index: int) -> bool:
        display = _display(words[index])
        if not display or not display[:1].isupper():
            return False
        core = display.lower()
        if core in _COMMON_CAPS or core in _ALL_CLOSED:
            return False
        if tags[index] != "NOUN" and core not in self.name_lexicon:
            return False
        if core in self.name_lexicon:
            return True
        if index > 0 and not is_sentence_end(words[index - 1]):
            return True
        # Sentence-initial capitalized token.  Accept a known name, a
        # multi-token capitalized span, or an otherwise unknown capitalized
        # noun.  Common sentence-initial words were filtered above.
        if index + 1 < len(words):
            nxt = _display(words[index + 1])
            if (nxt and nxt[:1].isupper()
                    and nxt.lower() not in _COMMON_CAPS):
                return True
        return True

    # -- streaming helpers -------------------------------------------------

    def reset(self) -> None:
        self._stream.clear()

    def observe(self, word: str | int) -> "CenteringCache":
        if isinstance(word, (int, np.integer)):
            self._stream.append(self.id_to_word.get(int(word), ""))
        else:
            self._stream.append(str(word))
        return self

    @property
    def observed(self) -> tuple[str, ...]:
        return tuple(self._stream)

    def entities(self) -> list[Entity]:
        return list(self.analyze(self._stream).entities)

    def candidate_ids(self, history: Iterable[str | int] | str,
                      top_k: int = 12) -> list[int]:
        return [candidate.word_id for candidate
                in self.candidates(history, top_k=top_k)
                if candidate.word_id is not None]

    def candidate_surfaces(self, history: Iterable[str | int] | str,
                           top_k: int = 12) -> list[str]:
        return [candidate.surface
                for candidate in self.candidates(history, top_k=top_k)]

    # -- analysis and candidates ------------------------------------------

    def analyze(self, history: Iterable[str | int] | str) -> CenteringState:
        words = self._as_words(history)
        if not words:
            return CenteringState([], "", "PUNCT", None, ())
        tags = self._tagger.tag_sequence(words)
        name_flags = [self._name_like(words, tags, index)
                      for index in range(len(words))]
        entities: dict[str, Entity] = {}
        last_pronoun: str | None = None

        def decay() -> None:
            for entity in entities.values():
                entity.score *= self.decay

        def mention(surface: str, head: str, tokens: tuple[str, ...],
                    position: int) -> None:
            key = head.lower()
            role = 1.0
            if position == 0 or is_sentence_end(words[position - 1]):
                role += 0.15
            if position + 1 < len(tags) and tags[position + 1] == "VERB":
                role += 0.20
            if position > 0 and tags[position - 1] == "DET":
                role += 0.05
            entity = entities.get(key)
            if entity is None:
                is_proper = (bool(tokens)
                             and _display(tokens[0])[:1].isupper())
                number = ("sg" if is_proper else
                          ("pl" if _is_plural_noun(head) else "sg"))
                entity = Entity(
                    surface=surface, head=head,
                    word_id=self._lookup_id(head),
                    gender=_name_gender(head, tokens),
                    number=number,
                    score=0.0, last_pos=position, count=0, tokens=tokens)
                entities[key] = entity
            entity.score += role * (1.0 + 0.10 * math.log1p(entity.count))
            entity.score = min(entity.score, 3.0)
            entity.count += 1
            entity.last_pos = position
            if len(tokens) > 1:
                entity.tokens = tokens
                entity.surface = surface

        index = 0
        while index < len(words):
            decay()
            tag = tags[index]
            if name_flags[index]:
                end = index
                while end + 1 < len(words) and name_flags[end + 1]:
                    end += 1
                tokens = tuple(_display(word) for word in
                               words[index:end + 1])
                mention(" ".join(tokens), tokens[-1], tokens, index)
                index = end + 1
                continue
            core = word_core(words[index])
            info = _PRONOUN_INFO.get(core)
            if info is not None:
                last_pronoun = core
                if entities:
                    gender, number = info
                    best = max(
                        entities.values(),
                        key=lambda entity: self._pronoun_score(
                            entity, gender, number))
                    if best.gender is None and gender is not None:
                        best.gender = gender
                    if best.number not in ("sg", "pl"):
                        best.number = number
                index += 1
                continue
            if tag == "NOUN":
                start = index
                cursor = index - 1
                while cursor >= 0 and index - cursor <= 2:
                    if tags[cursor] == "DET":
                        start = cursor
                        break
                    if tags[cursor] in ("ADJ", "NUM"):
                        cursor -= 1
                        continue
                    break
                tokens = tuple(_display(word) for word in
                               words[start:index + 1])
                mention(" ".join(tokens), _display(words[index]), tokens,
                        index)
            index += 1

        ordered = sorted(entities.values(),
                         key=lambda entity: (-entity.score,
                                             -entity.last_pos))
        ordered = ordered[:self.max_entities]
        return CenteringState(
            entities=ordered,
            last_token=word_core(words[-1]),
            last_tag=tags[-1],
            last_pronoun=last_pronoun if last_pronoun ==
            word_core(words[-1]) else None,
            tags=tuple(tags))

    def candidates(self, history: Iterable[str | int] | str,
                   top_k: int = 12) -> list[EntityCandidate]:
        state = self.analyze(history)
        if not state.entities:
            return []
        words = self._as_words(history)
        last = word_core(words[-1]) if words else ""
        boundary = bool(words) and is_sentence_end(words[-1])
        pronoun_info = _PRONOUN_INFO.get(last)
        found: dict[str, EntityCandidate] = {}

        def offer(surface: str, score: float, kind: str,
                  entity: Entity | None = None,
                  gender: str | None = None,
                  number: str = "sg") -> None:
            display = _display(surface)
            if not display:
                return
            key = display.lower()
            if key == last:
                score *= 0.40
            candidate = EntityCandidate(
                surface=display,
                word_id=self._lookup_id(display),
                score=float(score), source="centering", kind=kind,
                gender=gender, number=number)
            previous = found.get(key)
            if previous is None or candidate.score > previous.score:
                found[key] = candidate

        if pronoun_info is not None:
            gender, number = pronoun_info
            for entity in state.entities:
                match = self._pronoun_score(entity, gender, number)
                score = entity.score * (0.8 + 0.5 * match)
                offer(entity.head, score, "pronoun_resolution", entity,
                      entity.gender, entity.number)
                if (len(entity.tokens) > 1 and entity.tokens
                        and entity.tokens[0][:1].isupper()):
                    # Referring to a multi-token name often uses its first
                    # token ("Alice" for "Alice Smith").
                    offer(entity.tokens[0], score * 0.95,
                          "pronoun_resolution", entity,
                          entity.gender, entity.number)
        else:
            multiplier = 0.55
            if state.last_tag in ("DET", "ADJ"):
                multiplier = 0.95
            elif state.last_tag == "PREP":
                multiplier = 0.80
            elif boundary:
                multiplier = 0.70
            elif state.last_tag == "VERB":
                multiplier = 0.55
            for entity in state.entities:
                offer(entity.head, entity.score * multiplier, "entity",
                      entity, entity.gender, entity.number)
                if (len(entity.tokens) > 1 and entity.tokens
                        and entity.tokens[0][:1].isupper()):
                    offer(entity.tokens[0], entity.score * multiplier * 0.8,
                          "entity", entity, entity.gender, entity.number)

        # Learned name continuation: "Alice" -> "Smith".
        pair = self.name_pairs.get(last)
        if pair:
            for display, count in pair.most_common(3):
                offer(display, 1.0 + 0.1 * math.log1p(count),
                      "name_continuation")

        # Realising a salient entity as a pronoun is a weak but useful
        # candidate at sentence boundaries and after a name.
        if boundary or state.last_tag in ("NOUN", "VERB"):
            for pronoun in ("he", "she", "it", "they"):
                gender, number = _PRONOUN_INFO[pronoun]
                best = max(
                    (self._pronoun_score(entity, gender, number)
                     for entity in state.entities), default=0.0)
                if best > 0.0:
                    offer(pronoun, 0.18 * best, "pronoun",
                          gender=gender, number=number)

        ranked = sorted(found.values(), key=lambda candidate:
                        (-candidate.score, candidate.surface.lower()))
        return ranked[:max(int(top_k), 0)]

    def _as_words(self, history: Iterable[str | int] | str) -> list[str]:
        if isinstance(history, str):
            return history.split()
        words: list[str] = []
        for token in history:
            if isinstance(token, (int, np.integer)):
                words.append(self.id_to_word.get(int(token), ""))
            else:
                words.append(str(token))
        return words

    def _lookup_id(self, surface: str) -> int | None:
        for candidate in (surface, _display(surface), word_core(surface),
                          _display(surface).capitalize()):
            if candidate in self.word_to_id:
                return int(self.word_to_id[candidate])
        return None

    @staticmethod
    def _pronoun_score(entity: Entity, gender: str | None,
                       number: str | None) -> float:
        score = entity.score
        if number == "pl":
            score *= 1.35 if entity.number == "pl" else 0.50
        elif number == "sg":
            score *= 1.10 if entity.number == "sg" else 0.50
        elif number == "both":
            score *= 0.95
        if gender:
            if entity.gender == gender:
                score *= 1.45
            elif entity.gender is None:
                score *= 0.85
            else:
                score *= 0.25
        return score


def _name_gender(head: str, tokens: Sequence[str]) -> str | None:
    lower = head.lower()
    if lower in _FEMALE_NAMES:
        return "f"
    if lower in _MALE_NAMES:
        return "m"
    for token in tokens:
        name = word_core(token)
        if name in _FEMALE_NAMES:
            return "f"
        if name in _MALE_NAMES:
            return "m"
        if name in ("mr", "sir", "lord", "king", "prince"):
            return "m"
        if name in ("mrs", "ms", "miss", "lady", "queen", "princess"):
            return "f"
    if lower in _FEMALE_TITLES:
        return "f"
    if lower in _MALE_TITLES:
        return "m"
    return None


# ---------------------------------------------------------------------------
# 4. Morphology candidates
# ---------------------------------------------------------------------------

_IRREGULAR_VERBS = {
    "be": {"base": "be", "3sg": "is", "past": "was", "pp": "been",
           "gerund": "being"},
    "have": {"base": "have", "3sg": "has", "past": "had", "pp": "had",
             "gerund": "having"},
    "do": {"base": "do", "3sg": "does", "past": "did", "pp": "done",
           "gerund": "doing"},
    "go": {"base": "go", "3sg": "goes", "past": "went", "pp": "gone",
           "gerund": "going"},
    "make": {"base": "make", "3sg": "makes", "past": "made",
             "pp": "made", "gerund": "making"},
    "take": {"base": "take", "3sg": "takes", "past": "took",
             "pp": "taken", "gerund": "taking"},
    "come": {"base": "come", "3sg": "comes", "past": "came",
             "pp": "come", "gerund": "coming"},
    "see": {"base": "see", "3sg": "sees", "past": "saw", "pp": "seen",
            "gerund": "seeing"},
    "know": {"base": "know", "3sg": "knows", "past": "knew",
             "pp": "known", "gerund": "knowing"},
    "get": {"base": "get", "3sg": "gets", "past": "got",
            "pp": "gotten", "gerund": "getting"},
    "give": {"base": "give", "3sg": "gives", "past": "gave",
             "pp": "given", "gerund": "giving"},
    "think": {"base": "think", "3sg": "thinks", "past": "thought",
              "pp": "thought", "gerund": "thinking"},
    "say": {"base": "say", "3sg": "says", "past": "said", "pp": "said",
            "gerund": "saying"},
    "tell": {"base": "tell", "3sg": "tells", "past": "told",
             "pp": "told", "gerund": "telling"},
    "find": {"base": "find", "3sg": "finds", "past": "found",
             "pp": "found", "gerund": "finding"},
    "feel": {"base": "feel", "3sg": "feels", "past": "felt",
             "pp": "felt", "gerund": "feeling"},
    "leave": {"base": "leave", "3sg": "leaves", "past": "left",
              "pp": "left", "gerund": "leaving"},
    "bring": {"base": "bring", "3sg": "brings", "past": "brought",
              "pp": "brought", "gerund": "bringing"},
    "buy": {"base": "buy", "3sg": "buys", "past": "bought",
            "pp": "bought", "gerund": "buying"},
    "eat": {"base": "eat", "3sg": "eats", "past": "ate", "pp": "eaten",
            "gerund": "eating"},
    "run": {"base": "run", "3sg": "runs", "past": "ran", "pp": "run",
            "gerund": "running"},
    "write": {"base": "write", "3sg": "writes", "past": "wrote",
              "pp": "written", "gerund": "writing"},
    "speak": {"base": "speak", "3sg": "speaks", "past": "spoke",
              "pp": "spoken", "gerund": "speaking"},
    "meet": {"base": "meet", "3sg": "meets", "past": "met", "pp": "met",
             "gerund": "meeting"},
    "pay": {"base": "pay", "3sg": "pays", "past": "paid", "pp": "paid",
            "gerund": "paying"},
    "sit": {"base": "sit", "3sg": "sits", "past": "sat", "pp": "sat",
            "gerund": "sitting"},
    "stand": {"base": "stand", "3sg": "stands", "past": "stood",
              "pp": "stood", "gerund": "standing"},
    "hold": {"base": "hold", "3sg": "holds", "past": "held",
             "pp": "held", "gerund": "holding"},
    "win": {"base": "win", "3sg": "wins", "past": "won", "pp": "won",
            "gerund": "winning"},
    "lose": {"base": "lose", "3sg": "loses", "past": "lost",
             "pp": "lost", "gerund": "losing"},
    "send": {"base": "send", "3sg": "sends", "past": "sent",
             "pp": "sent", "gerund": "sending"},
    "build": {"base": "build", "3sg": "builds", "past": "built",
              "pp": "built", "gerund": "building"},
    "become": {"base": "become", "3sg": "becomes", "past": "became",
               "pp": "become", "gerund": "becoming"},
    "begin": {"base": "begin", "3sg": "begins", "past": "began",
              "pp": "begun", "gerund": "beginning"},
    "keep": {"base": "keep", "3sg": "keeps", "past": "kept",
             "pp": "kept", "gerund": "keeping"},
    "hear": {"base": "hear", "3sg": "hears", "past": "heard",
             "pp": "heard", "gerund": "hearing"},
    "mean": {"base": "mean", "3sg": "means", "past": "meant",
             "pp": "meant", "gerund": "meaning"},
    "show": {"base": "show", "3sg": "shows", "past": "showed",
             "pp": "shown", "gerund": "showing"},
    "grow": {"base": "grow", "3sg": "grows", "past": "grew",
             "pp": "grown", "gerund": "growing"},
    "draw": {"base": "draw", "3sg": "draws", "past": "drew",
             "pp": "drawn", "gerund": "drawing"},
    "sleep": {"base": "sleep", "3sg": "sleeps", "past": "slept",
              "pp": "slept", "gerund": "sleeping"},
    "teach": {"base": "teach", "3sg": "teaches", "past": "taught",
              "pp": "taught", "gerund": "teaching"},
    "catch": {"base": "catch", "3sg": "catches", "past": "caught",
              "pp": "caught", "gerund": "catching"},
    "fight": {"base": "fight", "3sg": "fights", "past": "fought",
              "pp": "fought", "gerund": "fighting"},
    "sell": {"base": "sell", "3sg": "sells", "past": "sold",
             "pp": "sold", "gerund": "selling"},
    "spend": {"base": "spend", "3sg": "spends", "past": "spent",
              "pp": "spent", "gerund": "spending"},
    "wear": {"base": "wear", "3sg": "wears", "past": "wore",
             "pp": "worn", "gerund": "wearing"},
    "choose": {"base": "choose", "3sg": "chooses", "past": "chose",
               "pp": "chosen", "gerund": "choosing"},
    "drive": {"base": "drive", "3sg": "drives", "past": "drove",
              "pp": "driven", "gerund": "driving"},
    "break": {"base": "break", "3sg": "breaks", "past": "broke",
              "pp": "broken", "gerund": "breaking"},
    "fly": {"base": "fly", "3sg": "flies", "past": "flew",
            "pp": "flown", "gerund": "flying"},
    "forget": {"base": "forget", "3sg": "forgets", "past": "forgot",
               "pp": "forgotten", "gerund": "forgetting"},
    "lie": {"base": "lie", "3sg": "lies", "past": "lay", "pp": "lain",
            "gerund": "lying"},
    "die": {"base": "die", "3sg": "dies", "past": "died", "pp": "died",
            "gerund": "dying"},
    "read": {"base": "read", "3sg": "reads", "past": "read",
             "pp": "read", "gerund": "reading"},
    "put": {"base": "put", "3sg": "puts", "past": "put", "pp": "put",
            "gerund": "putting"},
    "set": {"base": "set", "3sg": "sets", "past": "set", "pp": "set",
            "gerund": "setting"},
    "let": {"base": "let", "3sg": "lets", "past": "let", "pp": "let",
            "gerund": "letting"},
    "rise": {"base": "rise", "3sg": "rises", "past": "rose",
             "pp": "risen", "gerund": "rising"},
    "fall": {"base": "fall", "3sg": "falls", "past": "fell",
             "pp": "fallen", "gerund": "falling"},
    "seek": {"base": "seek", "3sg": "seeks", "past": "sought",
             "pp": "sought", "gerund": "seeking"},
    "hit": {"base": "hit", "3sg": "hits", "past": "hit", "pp": "hit",
            "gerund": "hitting"},
    "cut": {"base": "cut", "3sg": "cuts", "past": "cut", "pp": "cut",
            "gerund": "cutting"},
    "shut": {"base": "shut", "3sg": "shuts", "past": "shut", "pp": "shut",
             "gerund": "shutting"},
}

_IRREGULAR_NOUNS = {
    "children": ("child", ("plural",)),
    "men": ("man", ("plural",)),
    "women": ("woman", ("plural",)),
    "people": ("person", ("plural",)),
    "feet": ("foot", ("plural",)),
    "teeth": ("tooth", ("plural",)),
    "mice": ("mouse", ("plural",)),
    "geese": ("goose", ("plural",)),
    "indices": ("index", ("plural",)),
}
_IRREGULAR_ADJECTIVES = {
    "better": ("good", ("comparative",)),
    "best": ("good", ("superlative",)),
    "worse": ("bad", ("comparative",)),
    "worst": ("bad", ("superlative",)),
    "further": ("far", ("comparative",)),
    "furthest": ("far", ("superlative",)),
    "farther": ("far", ("comparative",)),
    "farthest": ("far", ("superlative",)),
    "less": ("little", ("comparative",)),
    "least": ("little", ("superlative",)),
    "more": ("many", ("comparative",)),
    "most": ("many", ("superlative",)),
}
_IRREGULAR_PLURALS = {
    lemma: surface
    for surface, (lemma, _features) in _IRREGULAR_NOUNS.items()
}
_IRREGULAR_EXTRA_SURFACES = {
    "am": ("be", ("present", "sg")),
    "are": ("be", ("present", "pl")),
    "were": ("be", ("past", "pl")),
}
_IRREGULAR_DEGREE = {
    "good": {"comparative": "better", "superlative": "best"},
    "bad": {"comparative": "worse", "superlative": "worst"},
    "far": {"comparative": "further", "superlative": "furthest"},
    "little": {"comparative": "less", "superlative": "least"},
    "many": {"comparative": "more", "superlative": "most"},
    "much": {"comparative": "more", "superlative": "most"},
}

_COMMON_VERBS = frozenset(_IRREGULAR_VERBS) | frozenset("""
walk look open smile jump talk play work call ask answer help love like live
move need want use try start stop plan visit watch listen cry study carry worry
hurry enjoy clean close dance change create decide describe improve include
provide receive require achieve believe produce divide remove save smile bake
wake name hope value argue
""".split())
_E_RESTORE_VERBS = frozenset("""
make take give have come move love live write drive use hope like bake wake save
name smile dance change choose lose raise close argue value create produce
decide divide include provide achieve believe receive improve remove require
describe become
""".split())
_COMMON_ADJECTIVES = _ADJ_WORDS | frozenset("""
normal local social national natural central special general recent political
public private foreign available financial commercial traditional cultural
""".split())


@dataclass(frozen=True)
class MorphAnalysis:
    """Best rule-based lemma analysis for one surface form."""

    surface: str
    lemma: str
    suffix: str
    features: tuple[str, ...]
    confidence: float
    alternatives: tuple[str, ...] = ()


@dataclass(frozen=True)
class MorphCandidate:
    """An inflected/derived form proposed by morphology."""

    surface: str
    lemma: str
    score: float
    feature: str
    source_word: str


class MorphologyCandidates:
    """Rule-based English suffix/lemma analysis and inflection generation.

    No external lexicon/WordNet is used.  ``variants`` intentionally
    overgenerates plausible forms; ``candidates`` ranks them with a coarse
    grammatical-context heuristic.
    """

    def __init__(self, max_candidates: int = 16) -> None:
        self.max_candidates = int(max_candidates)
        self._tagger = SimpleTagger()

    # -- analysis ----------------------------------------------------------

    def analyses(self, word: str) -> list[MorphAnalysis]:
        core = word_core(word)
        if not core or not _WORD_ONLY.match(core):
            return []
        if core in _FUNCTION_WORDS and core not in _IRREGULAR_VERBS:
            return []
        options: list[MorphAnalysis] = []

        def add(lemma: str, suffix: str, features: tuple[str, ...],
                confidence: float) -> None:
            lemma = lemma.lower()
            if not lemma or not _WORD_ONLY.match(lemma) or len(lemma) < 2:
                return
            options.append(MorphAnalysis(
                surface=core, lemma=lemma, suffix=suffix,
                features=features, confidence=float(confidence)))

        if core in _IRREGULAR_VERBS:
            add(core, "", ("base", "irregular"), 1.0)
        if core in _IRREGULAR_NOUNS:
            lemma, features = _IRREGULAR_NOUNS[core]
            add(lemma, "", features, 1.0)
        if core in _IRREGULAR_ADJECTIVES:
            lemma, features = _IRREGULAR_ADJECTIVES[core]
            add(lemma, "", features, 1.0)
        for lemma, forms in _IRREGULAR_VERBS.items():
            for feature, surface in forms.items():
                if feature != "base" and surface == core:
                    add(lemma, "", (feature,), 1.0)
        if core in _IRREGULAR_EXTRA_SURFACES:
            lemma, features = _IRREGULAR_EXTRA_SURFACES[core]
            add(lemma, "", features, 1.0)

        if core.endswith("ying") and len(core) > 5:
            add(core[:-4] + "ie", "ying", ("gerund",), 0.80)
        if core.endswith("ies") and len(core) > 4:
            add(core[:-3] + "y", "ies", ("plural",), 0.90)
            add(core[:-1], "s", ("plural",), 0.30)
        if core.endswith("ied") and len(core) > 4:
            add(core[:-3] + "y", "ied", ("past", "pp"), 0.90)
        if core.endswith("ves") and len(core) > 4:
            add(core[:-3] + "f", "ves", ("plural",), 0.65)
            add(core[:-3] + "fe", "ves", ("plural",), 0.55)
        if (core.endswith("es") and len(core) > 3
                and core.endswith(("ses", "xes", "zes", "ches", "shes",
                                   "oes"))):
            stem = core[:-2]
            add(stem, "es", ("plural",), 0.75)
            if stem.endswith("s"):
                add(stem + "e", "es", ("plural",), 0.80)
            else:
                add(stem + "e", "es", ("plural",), 0.30)
        elif (core.endswith("s") and len(core) > 2
              and not core.endswith(("ss", "us", "is"))):
            add(core[:-1], "s", ("plural",), 0.70)

        if core.endswith("ed") and len(core) > 3:
            stem = core[:-2]
            add(stem, "ed", ("past", "pp"), 0.70)
            if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
                add(stem[:-1], "ed", ("past", "pp"), 0.90)
            if stem + "e" in _E_RESTORE_VERBS:
                add(stem + "e", "ed", ("past", "pp"), 0.90)
            else:
                add(stem + "e", "ed", ("past", "pp"), 0.20)

        if core.endswith("ing") and len(core) > 4:
            stem = core[:-3]
            add(stem, "ing", ("gerund",), 0.70)
            if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
                add(stem[:-1], "ing", ("gerund",), 0.90)
            if stem + "e" in _E_RESTORE_VERBS:
                add(stem + "e", "ing", ("gerund",), 0.90)
            else:
                add(stem + "e", "ing", ("gerund",), 0.20)

        if core.endswith("est") and len(core) > 4:
            stem = core[:-3]
            add(stem, "est", ("superlative",), 0.70)
            if stem.endswith("i"):
                add(stem[:-1] + "y", "est", ("superlative",), 0.85)
            if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
                add(stem[:-1], "est", ("superlative",), 0.85)
            if stem + "e" in _COMMON_ADJECTIVES:
                add(stem + "e", "est", ("superlative",), 0.85)
        if core.endswith("er") and len(core) > 4:
            stem = core[:-2]
            add(stem, "er", ("comparative", "agent"), 0.60)
            if stem.endswith("i"):
                add(stem[:-1] + "y", "er", ("comparative", "agent"), 0.85)
            if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
                add(stem[:-1], "er", ("comparative", "agent"), 0.75)
            if stem + "e" in _COMMON_ADJECTIVES | _E_RESTORE_VERBS:
                add(stem + "e", "er", ("comparative", "agent"), 0.80)

        if core.endswith("ly") and len(core) > 4:
            stem = core[:-2]
            add(stem, "ly", ("adverb",), 0.75)
            if stem.endswith("i"):
                add(stem[:-1] + "y", "ly", ("adverb",), 0.85)
            if stem.endswith("al"):
                add(stem[:-2], "ally", ("adverb",), 0.55)
        if core.endswith("ness") and len(core) > 5:
            stem = core[:-4]
            add(stem, "ness", ("noun_derived",), 0.75)
            if stem.endswith("i"):
                add(stem[:-1] + "y", "ness", ("noun_derived",), 0.85)
        if core.endswith("ment") and len(core) > 5:
            add(core[:-4], "ment", ("noun_derived",), 0.70)
            add(core[:-4] + "e", "ment", ("noun_derived",), 0.55)
        if core.endswith("ion") and len(core) > 5:
            stem = core[:-3]
            add(stem, "ion", ("noun_derived",), 0.55)
            add(stem + "e", "ion", ("noun_derived",), 0.50)
            if stem.endswith("t"):
                add(stem[:-1] + "ss", "ion", ("noun_derived",), 0.25)
        if core.endswith("able") and len(core) > 5:
            add(core[:-4], "able", ("adjective_derived",), 0.65)
            add(core[:-4] + "e", "able", ("adjective_derived",), 0.50)
        if core.endswith("ible") and len(core) > 5:
            add(core[:-4], "ible", ("adjective_derived",), 0.55)
        if core.endswith("ful") and len(core) > 4:
            add(core[:-3], "ful", ("adjective_derived",), 0.75)
        if core.endswith("less") and len(core) > 5:
            add(core[:-4], "less", ("adjective_derived",), 0.75)
        if core.endswith("ous") and len(core) > 4:
            add(core[:-3], "ous", ("adjective_derived",), 0.70)
        if core.endswith("ive") and len(core) > 4:
            add(core[:-3], "ive", ("adjective_derived",), 0.65)
            add(core[:-3] + "e", "ive", ("adjective_derived",), 0.45)
        if core.endswith("ity") and len(core) > 5:
            stem = core[:-3]
            add(stem, "ity", ("noun_derived",), 0.60)
            add(stem + "e", "ity", ("noun_derived",), 0.50)
        if core.endswith("al") and len(core) > 4:
            add(core[:-2], "al", ("adjective_derived",), 0.45)
        if core.endswith("ic") and len(core) > 4:
            add(core[:-2], "ic", ("adjective_derived",), 0.50)
            add(core[:-2] + "e", "ic", ("adjective_derived",), 0.35)
        if core.endswith("ish") and len(core) > 4:
            add(core[:-3], "ish", ("adjective_derived",), 0.55)
        if core.endswith(("ize", "ise")) and len(core) > 5:
            add(core[:-3], core[-3:], ("verb_derived",), 0.65)
            add(core[:-3] + "e", core[-3:], ("verb_derived",), 0.30)
        if core.endswith("ify") and len(core) > 5:
            add(core[:-3], "ify", ("verb_derived",), 0.60)
            add(core[:-3] + "e", "ify", ("verb_derived",), 0.45)
        if core.endswith("ship") and len(core) > 5:
            add(core[:-4], "ship", ("noun_derived",), 0.65)
        if core.endswith("hood") and len(core) > 5:
            add(core[:-4], "hood", ("noun_derived",), 0.65)
        if core.endswith("dom") and len(core) > 5:
            add(core[:-3], "dom", ("noun_derived",), 0.55)

        # Strip duplicates and rank by confidence.  A bare word is a valid
        # base analysis, but a productive suffix analysis wins.
        unique: dict[tuple[str, tuple[str, ...]], MorphAnalysis] = {}
        for option in options:
            key = (option.lemma, option.features)
            previous = unique.get(key)
            if previous is None or option.confidence > previous.confidence:
                unique[key] = option
        ranked = sorted(unique.values(),
                        key=lambda item: (-item.confidence, item.lemma))
        if not ranked:
            ranked = [MorphAnalysis(
                surface=core, lemma=core, suffix="",
                features=("base",), confidence=0.40)]
        return ranked

    def analyze(self, word: str) -> MorphAnalysis | None:
        options = self.analyses(word)
        return options[0] if options else None

    # -- generation --------------------------------------------------------

    def variants(self, word: str, include_derived: bool = True) -> list[str]:
        """All plausible lemma/inflection variants of ``word``."""
        analysis = self.analyze(word)
        if analysis is None:
            return []
        lemma = analysis.lemma
        out = [lemma]
        for form in ("plural", "3sg", "past", "pp", "gerund",
                     "comparative", "superlative", "adverb"):
            out.extend(self._inflect(lemma, form))
        if include_derived:
            out.extend(self._derived(lemma))
        out.append(word_core(word))
        seen: set[str] = set()
        ordered: list[str] = []
        for surface in out:
            if surface and surface not in seen:
                seen.add(surface)
                ordered.append(surface)
        return ordered

    def candidates(self, history: Iterable[str | int] | str,
                   top_k: int = 12) -> list[MorphCandidate]:
        """Propose next-word forms conditioned on the observed context.

        Two signals are combined: variants of the immediately previous word
        (word-form repetition/backoff) and inflections of recent content-word
        lemmas that fit the grammatical slot after the previous word.
        """
        words = self._as_words(history)
        if not words:
            return []
        tags = self._tagger.tag_sequence(words)
        cores = [word_core(word) for word in words]
        last = cores[-1]
        last_tag = tags[-1]
        recent: list[tuple[float, str, str]] = []
        for offset in range(len(cores) - 1, max(-1, len(cores) - 25), -1):
            core = cores[offset]
            if (tag_in := tags[offset]) in ("NOUN", "VERB", "ADJ") and \
                    core not in _FUNCTION_WORDS:
                recency = math.exp(-(len(cores) - 1 - offset) / 12.0)
                recent.append((recency, core, tag_in))

        found: dict[str, MorphCandidate] = {}

        def offer(surface: str, score: float, feature: str,
                  source_word: str, lemma: str) -> None:
            surface = word_core(surface)
            if not surface or not _WORD_ONLY.match(surface):
                return
            candidate = MorphCandidate(
                surface=surface, lemma=lemma, score=float(score),
                feature=feature, source_word=source_word)
            previous = found.get(surface)
            if previous is None or candidate.score > previous.score:
                found[surface] = candidate

        last_analysis = self.analyze(last)
        if last_analysis is not None:
            # Only inflectional forms related to the last word; derived
            # forms stay available via ``variants`` but would swamp a
            # top-k next-word list with low-probability nouns/adverbs.
            late_score = (0.55 if last not in _FUNCTION_WORDS
                          and last not in _VERB_FORMS else 0.30)
            for form in self._last_word_forms(last, last_tag):
                score = 0.45 if form == "base" else late_score
                for variant in self._inflect(last_analysis.lemma, form):
                    offer(variant, score, f"last_word_{form}", last,
                          last_analysis.lemma)

        trigger = self._trigger_forms(last, last_tag,
                                      bool(words) and
                                      is_sentence_end(words[-1]))
        if trigger:
            for recency, core, tag in recent:
                analysis = self.analyze(core)
                if analysis is None:
                    continue
                for form, weight in trigger.items():
                    if not self._form_allowed(form, tag):
                        continue
                    for variant in self._inflect(analysis.lemma, form):
                        offer(variant, 0.60 * recency * weight,
                              f"context_{form}", core, analysis.lemma)

        ranked = sorted(found.values(), key=lambda candidate:
                        (-candidate.score, candidate.surface))
        if ranked:
            best = ranked[0].score
            if best > 0:
                ranked = [MorphCandidate(
                    surface=item.surface, lemma=item.lemma,
                    score=item.score / best, feature=item.feature,
                    source_word=item.source_word) for item in ranked]
        return ranked[:max(int(top_k), 0)]

    def candidate_surfaces(self, history: Iterable[str | int] | str,
                           top_k: int = 12) -> list[str]:
        """Surface-string view of ``candidates`` for simple callers."""
        return [candidate.surface
                for candidate in self.candidates(history, top_k=top_k)]

    @staticmethod
    def _last_word_forms(last: str, last_tag: str) -> tuple[str, ...]:
        if last_tag == "NOUN":
            return ("plural", "base")
        if last_tag == "VERB":
            return ("3sg", "past", "pp", "gerund", "base")
        if last_tag == "ADJ":
            return ("comparative", "superlative", "adverb", "base")
        if last_tag == "NUM":
            return ("base",)
        return ("plural", "past", "gerund", "base")

    def _trigger_forms(self, last: str, last_tag: str,
                       boundary: bool = False) -> dict[str, float]:
        if last == "to":
            return {"base": 1.0}
        if last in ("have", "has", "had", "having"):
            return {"pp": 1.0}
        if last in ("be", "am", "is", "are", "was", "were", "been", "being"):
            return {"gerund": 0.70, "pp": 0.80}
        if last in ("do", "does", "did", "can", "could", "will", "would",
                    "shall", "should", "may", "might", "must"):
            return {"base": 1.0}
        if last == "more":
            return {"comparative": 1.0}
        if last == "most":
            return {"superlative": 1.0}
        if last in ("many", "few", "several", "both", "two", "three",
                    "four", "five", "six", "seven", "eight", "nine",
                    "ten"):
            return {"plural": 0.90}
        if last_tag == "DET":
            return {"base": 0.85, "plural": 0.20}
        if last_tag == "NUM":
            return {"plural": 0.90}
        if last_tag == "PRON":
            if last in ("he", "she", "it"):
                return {"3sg": 0.90}
            if last in ("they", "we", "i", "you"):
                return {"base": 0.70, "plural": 0.30}
        if last_tag == "ADJ":
            return {"base": 0.70, "plural": 0.25}
        if last_tag == "PREP":
            return {"base": 0.70, "gerund": 0.20}
        if last_tag == "NOUN":
            return {"3sg": 0.55} if not _is_plural_noun(last) else {
                "base": 0.55}
        if last_tag == "VERB":
            return {"base": 0.50}
        if boundary:
            return {"base": 0.50}
        return {"base": 0.40}

    @staticmethod
    def _form_allowed(form: str, tag: str) -> bool:
        if tag == "NOUN":
            return form in ("plural", "base")
        if tag == "VERB":
            return form in ("3sg", "past", "pp", "gerund", "base")
        if tag == "ADJ":
            return form in ("comparative", "superlative", "adverb", "base")
        return True

    def _inflect(self, lemma: str, form: str) -> list[str]:
        lemma = word_core(lemma)
        if not lemma or not _WORD_ONLY.match(lemma):
            return []
        if lemma in _IRREGULAR_VERBS and form in _IRREGULAR_VERBS[lemma]:
            value = _IRREGULAR_VERBS[lemma][form]
            return [value]
        if form == "base":
            return [lemma]
        if form == "plural":
            if lemma in _IRREGULAR_PLURALS:
                return [_IRREGULAR_PLURALS[lemma]]
            return _regular_plural(lemma)
        if form == "3sg":
            return _regular_3sg(lemma)
        if form in ("past", "pp"):
            return _regular_past(lemma)
        if form == "gerund":
            return _regular_ing(lemma)
        if form in ("comparative", "superlative"):
            if lemma in _IRREGULAR_DEGREE:
                return [_IRREGULAR_DEGREE[lemma][form]]
            return _regular_degree(lemma, form)
        if form == "adverb":
            return _regular_adverb(lemma)
        if form == "agent":
            return _regular_agent(lemma)
        return []

    def _derived(self, lemma: str) -> list[str]:
        out: list[str] = []
        if lemma in _COMMON_ADJECTIVES or lemma.endswith(("ous", "ful",
                                                          "less", "ive")):
            out.append("un" + lemma)
        if lemma.endswith("y") and len(lemma) > 3:
            out.append(lemma[:-1] + "iness")
        else:
            out.append(lemma + "ness")
        out.append(lemma + "ment")
        out.append(lemma + "er")
        adverb = _regular_adverb(lemma)
        out.append(adverb[0] if adverb else lemma + "ly")
        out.append(lemma + "able")
        if lemma.endswith("y") and len(lemma) > 3:
            out.append(lemma[:-1] + "ily")
        return [word for word in out if word]

    @staticmethod
    def _as_words(history: Iterable[str | int] | str) -> list[str]:
        if isinstance(history, str):
            return history.split()
        return [str(token) for token in history]


def _regular_plural(lemma: str) -> list[str]:
    if lemma.endswith("y") and len(lemma) > 2 and lemma[-2] not in "aeiou":
        return [lemma[:-1] + "ies"]
    if lemma.endswith(("s", "x", "z", "ch", "sh")):
        return [lemma + "es"]
    if lemma.endswith("f"):
        return [lemma[:-1] + "ves", lemma + "s"]
    if lemma.endswith("fe"):
        return [lemma[:-2] + "ves", lemma + "s"]
    return [lemma + "s"]


def _regular_3sg(lemma: str) -> list[str]:
    if lemma in ("have", "be", "do", "go"):
        return {
            "have": ["has"], "be": ["is"], "do": ["does"], "go": ["goes"],
        }[lemma]
    return _regular_plural(lemma)


def _is_cvc(stem: str) -> bool:
    if len(stem) < 3:
        return False
    vowels = "aeiou"
    return (stem[-1] not in vowels and stem[-1] not in "wxy"
            and stem[-2] in vowels and stem[-3] not in vowels)


def _regular_past(lemma: str) -> list[str]:
    if lemma.endswith("e"):
        return [lemma + "d"]
    if lemma.endswith("y") and len(lemma) > 2 and lemma[-2] not in "aeiou":
        return [lemma[:-1] + "ied"]
    if _is_cvc(lemma) and len(lemma) <= 4:
        return [lemma + lemma[-1] + "ed"]
    return [lemma + "ed"]


def _regular_ing(lemma: str) -> list[str]:
    if lemma.endswith("ie"):
        return [lemma[:-2] + "ying"]
    if lemma.endswith("e") and not lemma.endswith("ee"):
        return [lemma[:-1] + "ing"]
    if _is_cvc(lemma) and len(lemma) <= 4:
        return [lemma + lemma[-1] + "ing"]
    return [lemma + "ing"]


def _regular_degree(lemma: str, form: str) -> list[str]:
    suffix = "er" if form == "comparative" else "est"
    if lemma.endswith("y") and len(lemma) > 2 and lemma[-2] not in "aeiou":
        return [lemma[:-1] + "i" + suffix]
    if lemma.endswith("e"):
        return [lemma + ("r" if form == "comparative" else "st")]
    if _is_cvc(lemma) and len(lemma) <= 4:
        return [lemma + lemma[-1] + suffix]
    return [lemma + suffix]


def _regular_adverb(lemma: str) -> list[str]:
    if lemma.endswith("y") and len(lemma) > 2 and lemma[-2] not in "aeiou":
        return [lemma[:-1] + "ily"]
    if lemma.endswith("le") and len(lemma) > 3:
        return [lemma[:-1] + "y"]
    if lemma.endswith("ic"):
        return [lemma + "ally"]
    return [lemma + "ly"]


def _regular_agent(lemma: str) -> list[str]:
    if lemma.endswith("e"):
        return [lemma + "r"]
    if _is_cvc(lemma) and len(lemma) <= 4:
        return [lemma + lemma[-1] + "er"]
    return [lemma + "er"]


# ---------------------------------------------------------------------------
# 5. Fusion
# ---------------------------------------------------------------------------

@dataclass
class LinguisticQuery:
    """Result of combining the linguistic experts for one history."""

    ids: list[int]
    scores: np.ndarray
    sources: dict[str, list[int]]
    source_scores: dict[int, dict[str, float]]
    details: dict = field(default_factory=dict)

    @property
    def vector(self) -> np.ndarray:
        return self.scores

    @property
    def shape(self) -> tuple[int, ...]:
        return self.scores.shape

    @property
    def dtype(self):
        return self.scores.dtype

    @property
    def candidates(self) -> list[int]:
        """Sparse ranked candidate IDs (alias for ``ids``)."""
        return self.ids

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.scores, dtype=dtype)

    def __len__(self) -> int:
        return len(self.scores)

    def __getitem__(self, index):
        return self.scores[index]

    def top_k(self, k: int = 10) -> list[tuple[int, float]]:
        limit = max(int(k), 0)
        return [(word, float(self.scores[word]))
                for word in self.ids[:limit]
                if 0 <= word < len(self.scores)]

    def candidate_ids(self) -> list[int]:
        return list(self.ids)

    def score(self, word_id: int) -> float:
        if 0 <= int(word_id) < len(self.scores):
            return float(self.scores[int(word_id)])
        return 0.0

    def as_dict(self) -> dict:
        return {
            "ids": list(self.ids),
            "sources": {name: list(ids)
                        for name, ids in self.sources.items()},
            "details": dict(self.details),
        }


_DEFAULT_WEIGHTS = {
    "construction": 1.00,
    "centering": 0.65,
    "pos": 0.55,
    "morphology": 0.35,
    "unigram": 0.03,
}


class LinguisticPrior:
    """Fuse tagger, constructions, centering and morphology.

    ``build(word_stream)`` accepts a stream of surface strings, or integer
    word IDs.  For full functionality pass ``word_to_id``/``id_to_word`` (a
    4096-entry vocabulary is the intended use).  With integer IDs and no
    mapping only exact construction matching and a unigram fallback are
    available.

    ``query(history)`` returns a :class:`LinguisticQuery` with a
    ``vocab_size``-dimensional score vector (4096 by default), sparse ranked
    IDs, and per-source ranked lists.  ``sources`` has the same shape as the
    candidate-source dictionary in ``word_rerank_fast.py``.
    """

    def __init__(self, vocab_size: int = DEFAULT_VOCAB_SIZE,
                 max_order: int = 6, min_count: int = 1,
                 max_next: int = 12, source_top_k: int = 100,
                 weights: dict[str, float] | None = None) -> None:
        self.vocab_size = int(vocab_size)
        self.source_top_k = int(source_top_k)
        self.weights = dict(_DEFAULT_WEIGHTS)
        if weights:
            self.weights.update({key: float(value)
                                 for key, value in weights.items()})
        self.tagger = SimpleTagger()
        self.constructions = ConstructionInventory(
            max_order=max_order, min_count=min_count, max_next=max_next)
        self.centering = CenteringCache()
        self.morphology = MorphologyCandidates()
        self.word_to_id: dict[str, int] = {}
        self.id_to_word: dict[int, str] = {}
        self.word_tag: dict[int, str] = {}
        self._tag_rows: dict[str, list[tuple[int, float]]] = {}
        self._tag_start: Counter[str] = Counter()
        self._tag_bigram: dict[tuple[str, str], Counter[str]] = defaultdict(
            Counter)
        self._tag_trigram: dict[tuple[str, str, str],
                                Counter[str]] = defaultdict(Counter)
        self._tag_unigram: Counter[str] = Counter()
        self._word_by_tag: dict[str, Counter[int]] = defaultdict(Counter)
        self._unigram: Counter[int] = Counter()
        self._built = False

    # -- build -------------------------------------------------------------

    def build(self, word_stream: Iterable[str | int], *,
              word_to_id: dict[str, int] | None = None,
              id_to_word: dict[int, str] | Sequence[str] | None = None,
              vocab_size: int | None = None,
              max_order: int | None = None,
              min_count: int | None = None,
              max_next: int | None = None) -> "LinguisticPrior":
        if max_order is not None:
            if int(max_order) != self.constructions.max_order:
                self.constructions = ConstructionInventory(
                    max_order=int(max_order),
                    min_count=(self.constructions.min_count
                               if min_count is None else int(min_count)),
                    max_next=(self.constructions.max_next
                              if max_next is None else int(max_next)))
        if min_count is not None:
            self.constructions.min_count = max(1, int(min_count))
        if max_next is not None:
            self.constructions.max_next = max(1, int(max_next))

        target_vocab = self.vocab_size if vocab_size is None else int(
            vocab_size)
        ids, surfaces, id_to_word, word_to_id, effective = _prepare_stream(
            word_stream, word_to_id=word_to_id, id_to_word=id_to_word,
            vocab_size=target_vocab)
        self.vocab_size = max(self.vocab_size, effective)
        self.word_to_id = dict(word_to_id)
        self.id_to_word = dict(id_to_word)

        tags: list[str] | None = None
        if surfaces and any(surfaces):
            tags = self.tagger.tag_sequence(surfaces)

        self.constructions.build_from_arrays(
            ids, tags=tags, id_to_word=self.id_to_word,
            word_to_id=self.word_to_id, vocab_size=self.vocab_size)
        self.centering.build_from_words(surfaces)
        self.centering.word_to_id = dict(self.word_to_id)
        self.centering.id_to_word = dict(self.id_to_word)
        self._build_pos(ids, surfaces, tags)
        self._unigram.clear()
        self._unigram.update(int(word) for word in ids)
        self._built = True
        return self

    def _build_pos(self, ids: Sequence[int], surfaces: Sequence[str],
                   tags: Sequence[str] | None) -> None:
        self._tag_start.clear()
        self._tag_bigram.clear()
        self._tag_trigram.clear()
        self._tag_unigram.clear()
        self._word_by_tag.clear()
        self._tag_rows.clear()
        self.word_tag.clear()
        if tags is None or not tags:
            return
        start = True
        for index, tag in enumerate(tags):
            self._tag_unigram[tag] += 1
            self._word_by_tag[tag][int(ids[index])] += 1
            if start:
                self._tag_start[tag] += 1
                start = False
            if index + 1 < len(tags):
                self._tag_bigram[tag][tags[index + 1]] += 1
            if index + 2 < len(tags):
                self._tag_trigram[(tag, tags[index + 1])][
                    tags[index + 2]] += 1
            if surfaces and is_sentence_end(surfaces[index]):
                start = True

        max_per_tag = 64
        for tag, row in self._word_by_tag.items():
            total = sum(row.values())
            if total <= 0:
                continue
            self._tag_rows[tag] = [
                (int(word), count / total)
                for word, count in row.most_common(max_per_tag)
                if int(word) != UNK_ID]
        votes: dict[int, Counter[str]] = defaultdict(Counter)
        for tag, row in self._word_by_tag.items():
            for word, count in row.items():
                votes[int(word)][tag] += count
        self.word_tag = {
            word: max(tag_counts.items(),
                      key=lambda item: (item[1], -TAG_INDEX.get(item[0], 99))
                      )[0]
            for word, tag_counts in votes.items()}

    # -- query -------------------------------------------------------------

    def _encode_history(
            self, history: Iterable[str | int] | str
    ) -> tuple[list[int], list[str]]:
        if isinstance(history, str):
            tokens: list[str | int] = history.split()
        else:
            tokens = list(history)
        ids: list[int] = []
        surfaces: list[str] = []
        for token in tokens:
            if isinstance(token, (int, np.integer)):
                word_id = int(token)
                ids.append(word_id)
                surfaces.append(self.id_to_word.get(word_id, ""))
            else:
                surface = str(token)
                ids.append(self._lookup_id(surface))
                surfaces.append(surface)
        return ids, surfaces

    def _lookup_id(self, surface: str) -> int:
        for candidate in (surface, _display(surface), word_core(surface),
                          _display(surface).capitalize()):
            if candidate in self.word_to_id:
                return int(self.word_to_id[candidate])
        return UNK_ID

    def _pos_distribution(self, surfaces: Sequence[str],
                          top_tags: int = 3,
                          per_tag: int = 32) -> dict[int, float]:
        if not self._tag_rows or not surfaces or not any(surfaces):
            return {}
        tags = self.tagger.tag_sequence(surfaces)
        if not tags or is_sentence_end(surfaces[-1]):
            tag_scores = self._normalise_counter(self._tag_start)
        else:
            last = tags[-1]
            prev = tags[-2] if len(tags) >= 2 else "<s>"
            scores: dict[str, float] = defaultdict(float)
            trigram_row = self._tag_trigram.get((prev, last))
            if trigram_row:
                total = sum(trigram_row.values())
                for tag, count in trigram_row.items():
                    scores[tag] += 0.62 * count / total
            bigram_row = self._tag_bigram.get(last)
            if bigram_row:
                total = sum(bigram_row.values())
                for tag, count in bigram_row.items():
                    scores[tag] += 0.28 * count / total
            if self._tag_unigram:
                total = sum(self._tag_unigram.values())
                for tag, count in self._tag_unigram.items():
                    scores[tag] += 0.10 * count / total
            if not scores:
                scores.update(self._normalise_counter(self._tag_start))
            tag_scores = dict(scores)
        distributed: dict[int, float] = defaultdict(float)
        for tag, probability in sorted(tag_scores.items(),
                                       key=lambda item: (-item[1], item[0])
                                       )[:top_tags]:
            for word, word_probability in self._tag_rows.get(
                    tag, [])[:per_tag]:
                distributed[int(word)] += float(probability) * word_probability
        return dict(distributed)

    @staticmethod
    def _normalise_counter(counter: Counter[str]) -> dict[str, float]:
        total = sum(counter.values())
        if total <= 0:
            return {}
        return {key: value / total for key, value in counter.items()}

    def _rank_map(self, scores: dict[int, float],
                  limit: int | None = None) -> list[tuple[int, float]]:
        ranked = sorted(((int(word), float(score))
                         for word, score in scores.items()
                         if int(word) != UNK_ID
                         and int(word) < self.vocab_size),
                        key=lambda item: (-item[1], item[0]))
        if limit is not None:
            ranked = ranked[:max(int(limit), 0)]
        return ranked

    def query(self, history: Iterable[str | int] | str = (),
              top_k: int | None = 100) -> LinguisticQuery:
        """Combine all experts for one causal history.

        ``history`` may contain surface strings or integer word IDs.
        ``top_k`` only truncates the sparse ``ids`` list; the returned
        ``scores`` vector still holds every generated candidate.
        """
        ids, surfaces = self._encode_history(history)
        tags = self.tagger.tag_sequence(surfaces) if any(surfaces) else []

        construction_match = None
        construction: dict[int, float] = {}
        if ids:
            construction_match = self.constructions.longest_match(
                ids=ids, surfaces=surfaces, tags=tags)
            if construction_match is not None:
                # Include the longest match plus shorter-order backoff for a
                # slightly wider candidate set.
                for word, probability in self.constructions.next_distribution(
                        ids=ids, surfaces=surfaces, tags=tags,
                        top_k=max(self.source_top_k, 1), backoff=True):
                    construction[int(word)] = float(probability)

        pos = self._pos_distribution(surfaces) if any(surfaces) else {}

        centering: dict[int, float] = {}
        if any(surfaces):
            for candidate in self.centering.candidates(
                    surfaces, top_k=max(self.source_top_k, 1)):
                if candidate.word_id is None:
                    continue
                word = int(candidate.word_id)
                centering[word] = max(centering.get(word, 0.0),
                                      float(candidate.score))

        morphology: dict[int, float] = {}
        if any(surfaces):
            for candidate in self.morphology.candidates(
                    surfaces, top_k=max(self.source_top_k, 1)):
                word = self._lookup_id(candidate.surface)
                if word == UNK_ID:
                    continue
                morphology[word] = max(morphology.get(word, 0.0),
                                       float(candidate.score))

        # Normalise each source before weighted mixing so that a source with
        # hundreds of tiny scores cannot overwhelm a peaked source.
        construction = self._unit_max(construction)
        pos = self._unit_max(pos)
        centering = self._unit_max(centering)
        morphology = self._unit_max(morphology)

        source_maps = {
            "construction": construction,
            "pos": pos,
            "centering": centering,
            "morphology": morphology,
        }
        combined: dict[int, float] = defaultdict(float)
        source_scores: dict[int, dict[str, float]] = defaultdict(dict)
        for source, mapping in source_maps.items():
            weight = float(self.weights.get(source, 0.0))
            for word, score in mapping.items():
                if word <= 0 or word >= self.vocab_size:
                    continue
                contribution = weight * float(score)
                combined[word] += contribution
                source_scores[word][source] = float(score)

        # Unigram backoff keeps the vector non-empty on cold start and adds a
        # tiny amount of tail coverage.  It is not a fifth linguistic expert.
        unigram: dict[int, float] = {}
        if self._unigram:
            top = self._unigram.most_common(16)
            total = sum(count for _word, count in top)
            if total > 0:
                unigram = {int(word): count / total
                           for word, count in top if int(word) > 0}
                # Only let the unigram tail enter the mixture when the
                # linguistic experts produced a very small candidate set;
                # the source list is still exposed for diagnostics.
                if len(combined) < 8:
                    for word, score in unigram.items():
                        if word < self.vocab_size:
                            combined[word] += self.weights["unigram"] * score
                            source_scores[word]["unigram"] = float(score)

        if not combined:
            return LinguisticQuery(
                ids=[], scores=np.zeros(self.vocab_size, dtype=np.float32),
                sources={name: [] for name in source_maps},
                source_scores={},
                details={"tags": tags,
                         "construction": construction_match,
                         "next_tags": []})

        # Renormalise the sparse mixture to max 1.0.
        best = max(combined.values())
        if best > 0:
            for word in list(combined):
                combined[word] /= best

        ranked = sorted(combined.items(), key=lambda item: (-item[1],
                                                           item[0]))
        ranked = [(int(word), float(score)) for word, score in ranked
                  if 0 < int(word) < self.vocab_size]

        scores = np.zeros(self.vocab_size, dtype=np.float32)
        for word, score in ranked:
            scores[word] = score

        sources: dict[str, list[int]] = {}
        for source, mapping in source_maps.items():
            sources[source] = [word for word, _score
                               in self._rank_map(
                                   mapping, limit=self.source_top_k)]
        if unigram:
            sources["unigram"] = [word for word, _score
                                  in self._rank_map(
                                      unigram, limit=self.source_top_k)]

        keep = len(ranked) if top_k is None else min(len(ranked),
                                                     max(int(top_k), 0))
        ids_out = [word for word, _score in ranked[:keep]]
        next_tags = sorted(
            ((tag, probability) for tag, probability in
             self._tag_distribution(surfaces).items()),
            key=lambda item: (-item[1], item[0]))[:5] if any(surfaces) else []
        details = {
            "tags": tags,
            "construction": construction_match,
            "next_tags": next_tags,
            "centering": [(candidate.surface, candidate.kind,
                            candidate.score)
                          for candidate in self.centering.candidates(
                              surfaces, top_k=8)] if any(surfaces) else [],
            "morphology": [(candidate.surface, candidate.feature,
                             candidate.score)
                           for candidate in self.morphology.candidates(
                               surfaces, top_k=8)] if any(surfaces) else [],
        }
        return LinguisticQuery(
            ids=ids_out, scores=scores, sources=sources,
            source_scores={word: dict(values)
                           for word, values in source_scores.items()},
            details=details)

    def _tag_distribution(self, surfaces: Sequence[str]) -> dict[str, float]:
        if not surfaces or not any(surfaces):
            return {}
        tags = self.tagger.tag_sequence(surfaces)
        if not tags or is_sentence_end(surfaces[-1]):
            return self._normalise_counter(self._tag_start)
        last = tags[-1]
        prev = tags[-2] if len(tags) >= 2 else "<s>"
        scores: dict[str, float] = defaultdict(float)
        trigram_row = self._tag_trigram.get((prev, last))
        if trigram_row:
            total = sum(trigram_row.values())
            for tag, count in trigram_row.items():
                scores[tag] += 0.62 * count / total
        bigram_row = self._tag_bigram.get(last)
        if bigram_row:
            total = sum(bigram_row.values())
            for tag, count in bigram_row.items():
                scores[tag] += 0.28 * count / total
        if self._tag_unigram:
            total = sum(self._tag_unigram.values())
            for tag, count in self._tag_unigram.items():
                scores[tag] += 0.10 * count / total
        return dict(scores) or self._normalise_counter(self._tag_start)

    @staticmethod
    def _unit_max(scores: dict[int, float]) -> dict[int, float]:
        if not scores:
            return {}
        best = max(scores.values())
        if best <= 0:
            return {}
        return {int(word): float(score) / best
                for word, score in scores.items()}

    # -- integration helpers ----------------------------------------------

    def query_vector(self, history: Iterable[str | int] | str = ()) -> np.ndarray:
        """Return only the 4096-dim score vector."""
        return self.query(history).scores

    def query_sparse(self, history: Iterable[str | int] | str = (),
                     top_k: int = 100) -> list[tuple[int, float]]:
        """Return only the sparse ranked candidate list."""
        return self.query(history, top_k=top_k).top_k(top_k)

    def sources(self, history: Iterable[str | int] | str = (),
                top_k: int = 100) -> dict[str, list[int]]:
        """Per-source ranked IDs, compatible with ``sources_at``."""
        result = self.query(history, top_k=top_k)
        return {name: ids[:max(int(top_k), 0)]
                for name, ids in result.sources.items()}

    def candidate_features(
            self, history: Iterable[str | int] | str,
            candidates: Sequence[int],
            query: LinguisticQuery | None = None) -> list[dict[str, float]]:
        """Rank/score features for an external listwise reranker."""
        result = query if query is not None else self.query(history)
        rank = {word: index for index, word in enumerate(result.ids)}
        source_ranks = {
            name: {word: index for index, word in enumerate(ids)}
            for name, ids in result.sources.items()}
        rows: list[dict[str, float]] = []
        for candidate in candidates:
            word = int(candidate)
            row: dict[str, float] = {
                "ling_score": result.score(word),
                "ling_rank": float(rank.get(word, -1)),
                "ling_rank_inv": (1.0 / (rank[word] + 1)
                                  if word in rank else 0.0),
            }
            values = result.source_scores.get(word, {})
            for source in ("construction", "pos", "centering",
                           "morphology", "unigram"):
                source_rank = source_ranks.get(source, {}).get(word)
                row[f"{source}_rank"] = float(
                    source_rank if source_rank is not None else -1)
                row[f"{source}_rank_inv"] = (
                    1.0 / (source_rank + 1)
                    if source_rank is not None else 0.0)
                row[f"{source}_score"] = float(values.get(source, 0.0))
            tag = self.word_tag.get(word)
            for known_tag in TAGS:
                row[f"tag_{known_tag}"] = 1.0 if tag == known_tag else 0.0
            rows.append(row)
        return rows


# ---------------------------------------------------------------------------
# Stream/vocabulary helpers
# ---------------------------------------------------------------------------

def _prepare_stream(
        word_stream: Iterable[str | int], *,
        word_to_id: dict[str, int] | None = None,
        id_to_word: dict[int, str] | Sequence[str] | None = None,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> tuple[list[int], list[str], dict[int, str], dict[str, int], int]:
    """Normalise a word stream to IDs, surfaces, and bidirectional vocab."""
    if isinstance(word_stream, str):
        items: list = word_stream.split()
    else:
        items = list(word_stream)
    if not items:
        return [], [], {}, {}, int(vocab_size)

    first = items[0]
    if isinstance(first, (int, np.integer)):
        ids = [int(word) for word in items]
        if id_to_word is None:
            id_map: dict[int, str] = {}
        elif isinstance(id_to_word, dict):
            id_map = {int(key): str(value)
                      for key, value in id_to_word.items()}
        else:
            id_map = {index: str(value)
                      for index, value in enumerate(id_to_word)}
        surfaces = [id_map.get(word, "") for word in ids]
        if word_to_id is None:
            word_map = {value: int(key) for key, value in id_map.items()}
        else:
            word_map = {str(key): int(value)
                        for key, value in word_to_id.items()}
    else:
        surfaces = [str(word) for word in items]
        if word_to_id is None:
            counts = Counter(surfaces)
            limit = max(int(vocab_size) - 1, 0)
            kept = [word for word, _count in counts.most_common(limit)]
            word_map = {word: index + 1
                        for index, word in enumerate(kept)}
        else:
            word_map = {str(key): int(value)
                        for key, value in word_to_id.items()}
        ids = [int(word_map.get(word, UNK_ID)) for word in surfaces]
        if id_to_word is None:
            id_map = {identifier: word for word, identifier in word_map.items()}
        elif isinstance(id_to_word, dict):
            id_map = {int(key): str(value)
                      for key, value in id_to_word.items()}
        else:
            id_map = {index: str(value)
                      for index, value in enumerate(id_to_word)}

    maximum = max(ids) if ids else 0
    effective = max(int(vocab_size), maximum + 1)
    return ids, surfaces, id_map, word_map, effective


def _bump_count(rows: dict[tuple[int, ...], Counter[int]],
                context: tuple[int, ...], successor: int,
                max_next: int) -> None:
    row = rows.get(context)
    if row is None:
        row = Counter()
        rows[context] = row
    row[int(successor)] += 1
    # Keep the per-context table bounded while streaming; final selection is
    # done by ``_finalise_rows``.
    if len(row) > max_next * 8:
        rows[context] = Counter(dict(row.most_common(max_next * 4)))


__all__ = [
    "TAGS", "DEFAULT_VOCAB_SIZE", "UNK_ID",
    "SimpleTagger", "ConstructionInventory", "ConstructionMatch",
    "CenteringCache", "CenteringState", "Entity", "EntityCandidate",
    "MorphologyCandidates", "MorphAnalysis", "MorphCandidate",
    "LinguisticPrior", "LinguisticQuery",
]
