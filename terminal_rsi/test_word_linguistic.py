"""Unit tests for the rule-based linguistic word experts."""
from __future__ import annotations

import unittest

import numpy as np

from terminal_rsi.word_linguistic import (
    TAGS, CenteringCache, ConstructionInventory, LinguisticPrior,
    MorphologyCandidates, SimpleTagger,
)

STREAM = """
Alice opened the old door.
She looked at the old door.
The old man walked into the room.
The old man walked into the garden.
The young man walked into the room.
Alice smiled at the young man.
The old man sat down.
""".split()

NAME_STREAM = """
Alice Smith opened the door.
Alice Smith smiled.
Alice Smith went home.
""".split()


def build_vocab(words):
    return {word: index + 1
            for index, word in enumerate(sorted(set(words)))}


class SimpleTaggerTests(unittest.TestCase):
    def test_closed_class_and_suffixes(self):
        words = "The quick brown fox jumps over the lazy dog .".split()
        self.assertEqual(
            SimpleTagger().tag_sequence(words),
            ["DET", "ADJ", "ADJ", "NOUN", "VERB", "PREP",
             "DET", "ADJ", "NOUN", "PUNCT"])

    def test_irregular_verbs_and_numbers(self):
        words = "She bought two red apples and ate them .".split()
        self.assertEqual(
            SimpleTagger().tag_sequence(words),
            ["PRON", "VERB", "NUM", "ADJ", "NOUN", "CONJ",
             "VERB", "PRON", "PUNCT"])
        self.assertEqual(
            SimpleTagger().tag_sequence("two 2026 first .".split()),
            ["NUM", "NUM", "NUM", "PUNCT"])

    def test_plural_vs_third_person(self):
        self.assertEqual(SimpleTagger().tag_sequence(["the", "walks"]),
                         ["DET", "NOUN"])
        self.assertEqual(SimpleTagger().tag_sequence(["they", "walks"]),
                         ["PRON", "VERB"])

    def test_tag_is_causal(self):
        tagger = SimpleTagger()
        self.assertEqual(tagger.tag("fox"), "OTHER")
        tagger.reset()
        self.assertEqual(tagger.tag("the"), "DET")
        self.assertEqual(tagger.tag("fox"), "NOUN")

    def test_tag_set_is_complete(self):
        self.assertEqual(len(TAGS), 10)
        self.assertIn("PUNCT", TAGS)
        self.assertIn("OTHER", TAGS)

    def test_string_sentence_input(self):
        self.assertEqual(SimpleTagger().tag_sequence("the dog"),
                         ["DET", "NOUN"])


class ConstructionInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = ConstructionInventory(
            max_order=6, min_count=1).build(STREAM)
        cls.walked = cls.inventory.word_to_id["walked"]
        cls.sat = cls.inventory.word_to_id["sat"]

    def test_exact_longest_suffix(self):
        match = self.inventory.longest_match(["the", "old", "man"])
        self.assertIsNotNone(match)
        self.assertEqual(match.order, 3)
        self.assertEqual(match.kind, "exact")
        self.assertEqual(match.next_ids[0], self.walked)
        self.assertIn(self.sat, match.distribution())
        self.assertEqual(match.top_k(2)[0][0], self.walked)

    def test_capitalisation_is_folded_for_context(self):
        lower = self.inventory.longest_match(["the", "old", "man"])
        upper = self.inventory.longest_match(["The", "old", "man"])
        self.assertIsNotNone(upper)
        self.assertEqual(upper.order, lower.order)
        self.assertEqual(upper.next_ids[0], lower.next_ids[0])

    def test_typed_skeleton_matches_novel_adjective(self):
        match = self.inventory.longest_match(["the", "tall", "man"])
        self.assertIsNotNone(match)
        self.assertEqual(match.order, 3)
        self.assertEqual(match.kind, "typed")
        self.assertIn("ADJ", match.slot_types)
        self.assertIn("NOUN", match.slot_types)
        self.assertIn(self.walked, [word for word, _prob in match.top_k(4)])
        self.assertAlmostEqual(sum(match.probs), 1.0, places=6)

    def test_untyped_x_skeleton_fallback(self):
        match = self.inventory.longest_match(["the", "quickly", "man"])
        self.assertIsNotNone(match)
        self.assertEqual(match.kind, "untyped")
        self.assertIn("<X>", match.untyped_skeleton)
        self.assertIn("X", match.slot_types)

    def test_min_count_filters_singleton_contexts(self):
        # All tokens are closed-class, so the skeleton index has no slots and
        # this isolates the exact-context count filter.
        inventory = ConstructionInventory(
            max_order=4, min_count=2).build(
                "the a and . the a or .".split())
        self.assertIsNone(inventory.longest_match(["a", "and"]))
        match = inventory.longest_match(["the", "a"])
        self.assertIsNotNone(match)
        self.assertEqual(match.kind, "exact")
        self.assertEqual(set(match.next_ids),
                         {inventory.word_to_id["and"],
                          inventory.word_to_id["or"]})

    def test_shorter_suffix_still_matches(self):
        match = self.inventory.longest_match(["old", "man"])
        self.assertIsNotNone(match)
        self.assertEqual(match.order, 2)
        self.assertEqual(match.kind, "exact")

    def test_string_history_and_distribution(self):
        match = self.inventory.longest_match("the old man")
        self.assertIsNotNone(match)
        self.assertEqual(match.order, 3)
        distribution = self.inventory.query_distribution("the old man")
        self.assertEqual(distribution[match.next_ids[0]],
                         match.probs[0])

    def test_query_and_backoff_distribution(self):
        top = self.inventory.query(["the", "old", "man"], top_k=2)
        self.assertEqual(len(top), 2)
        self.assertEqual(top[0][0], self.walked)
        self.assertAlmostEqual(sum(score for _word, score in top), 1.0)
        backoff = self.inventory.next_distribution(
            ["the", "tall", "man"], top_k=5, backoff=True)
        self.assertIn(self.walked, [word for word, _score in backoff])
        self.assertAlmostEqual(sum(score for _word, score in backoff), 1.0)

    def test_six_word_context_is_mined(self):
        inventory = ConstructionInventory(
            max_order=6, min_count=1).build(
                "a b c d e f g h".split())
        match = inventory.longest_match(["a", "b", "c", "d", "e", "f"])
        self.assertIsNotNone(match)
        self.assertEqual(match.order, 6)
        self.assertEqual(match.next_ids[0],
                         inventory.word_to_id["g"])


class CenteringCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocab = build_vocab(STREAM + NAME_STREAM)
        cls.cache = CenteringCache().build(STREAM, word_to_id=cls.vocab)
        cls.names = CenteringCache().build(NAME_STREAM,
                                           word_to_id=cls.vocab)

    def test_entities_track_names_and_nouns(self):
        state = self.cache.analyze(
            ["Alice", "opened", "the", "door.", "She"])
        heads = [entity.head.lower() for entity in state.entities]
        self.assertIn("alice", heads)
        self.assertIn("door", heads)
        self.assertEqual(state.entities[0].head.lower(), "alice")

    def test_pronoun_resolves_to_salient_name(self):
        candidates = self.cache.candidates(
            ["Alice", "opened", "the", "door.", "She"], top_k=5)
        self.assertTrue(candidates)
        top = candidates[0]
        self.assertEqual(top.surface.lower(), "alice")
        self.assertEqual(top.kind, "pronoun_resolution")
        self.assertEqual(top.gender, "f")
        self.assertEqual(top.word_id, self.vocab["Alice"])

    def test_entity_candidate_after_determiner(self):
        candidates = self.cache.candidates(
            ["Alice", "opened", "the"], top_k=5)
        surfaces = [candidate.surface.lower() for candidate in candidates]
        self.assertIn("alice", surfaces)
        self.assertTrue(any(candidate.word_id is not None
                            for candidate in candidates))

    def test_multiword_name_continuation(self):
        candidates = self.names.candidates(["Alice"], top_k=5)
        self.assertTrue(candidates)
        self.assertEqual(candidates[0].surface.lower(), "smith")
        self.assertEqual(candidates[0].kind, "name_continuation")
        self.assertEqual(candidates[0].word_id, self.vocab["Smith"])

    def test_streaming_observation(self):
        cache = CenteringCache()
        cache.reset()
        cache.observe("Alice").observe("ran").observe(".")
        self.assertEqual(cache.observed, ("Alice", "ran", "."))
        heads = [entity.head.lower() for entity in cache.entities()]
        self.assertIn("alice", heads)


class MorphologyCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.morph = MorphologyCandidates()

    def test_lemma_analysis(self):
        expected = {
            "running": "run",
            "stories": "story",
            "apples": "apple",
            "walked": "walk",
            "making": "make",
            "boxes": "box",
            "houses": "house",
            "children": "child",
            "better": "good",
        }
        for word, lemma in expected.items():
            analysis = self.morph.analyze(word)
            self.assertIsNotNone(analysis, word)
            self.assertEqual(analysis.lemma, lemma, word)

    def test_variants_include_inflections(self):
        apple_variants = self.morph.variants("apple")
        self.assertIn("apple", apple_variants)
        self.assertIn("apples", apple_variants)
        run_variants = self.morph.variants("run")
        for form in ("run", "runs", "ran", "running"):
            self.assertIn(form, run_variants)
        self.assertIn("children", self.morph.variants("child"))

    def test_context_conditioned_plural(self):
        candidates = self.morph.candidates(["two", "apple"], top_k=8)
        surfaces = [candidate.surface for candidate in candidates]
        self.assertIn("apples", surfaces)

    def test_context_conditioned_participle(self):
        candidates = self.morph.candidates(
            ["they", "walk", "have"], top_k=12)
        surfaces = [candidate.surface for candidate in candidates]
        self.assertIn("walked", surfaces)

    def test_context_conditioned_base_after_to(self):
        candidates = self.morph.candidates(
            ["she", "run", "to"], top_k=8)
        surfaces = [candidate.surface for candidate in candidates]
        self.assertIn("run", surfaces)

    def test_context_conditioned_third_person(self):
        candidates = self.morph.candidates(
            ["the", "dog", "runs", "the", "boy"], top_k=12)
        surfaces = [candidate.surface for candidate in candidates]
        self.assertIn("runs", surfaces)

    def test_scores_are_sorted(self):
        candidates = self.morph.candidates(
            ["they", "walk", "have"], top_k=12)
        scores = [candidate.score for candidate in candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))


class LinguisticPriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior = LinguisticPrior(vocab_size=4096).build(
            STREAM, min_count=1, max_order=6)
        cls.walked = cls.prior.word_to_id["walked"]
        cls.sat = cls.prior.word_to_id["sat"]
        cls.alice = cls.prior.word_to_id["Alice"]

    def test_vector_shape_and_top_candidate(self):
        result = self.prior.query(["the", "old", "man"])
        self.assertEqual(result.scores.shape, (4096,))
        self.assertEqual(result.scores.dtype, np.float32)
        self.assertEqual(np.asarray(result).shape, (4096,))
        self.assertEqual(result.ids[0], self.walked)
        self.assertEqual(result.top_k(1)[0][0], self.walked)
        self.assertGreater(result.score(self.walked), 0.0)

    def test_sources_shape_and_construction_membership(self):
        sources = self.prior.sources(["the", "tall", "man"], top_k=10)
        for name in ("construction", "pos", "centering", "morphology"):
            self.assertIn(name, sources)
            for word in sources[name]:
                self.assertGreaterEqual(word, 0)
                self.assertLess(word, 4096)
        self.assertIn(self.walked, sources["construction"])

    def test_typed_construction_feeds_vector(self):
        result = self.prior.query(["the", "tall", "man"])
        self.assertGreater(result.score(self.walked), 0.0)
        match = result.details["construction"]
        self.assertIsNotNone(match)
        self.assertEqual(match.kind, "typed")

    def test_centering_candidate_in_fusion(self):
        result = self.prior.query(
            ["Alice", "opened", "the", "door.", "She"])
        self.assertIn(self.alice, result.sources["centering"])
        self.assertGreater(result.score(self.alice), 0.0)

    def test_morphology_candidate_in_fusion(self):
        result = self.prior.query(["they", "walk", "have"])
        self.assertIn(self.walked, result.sources["morphology"])
        self.assertGreater(result.score(self.walked), 0.0)

    def test_id_history_works(self):
        history = [self.prior.word_to_id[word]
                   for word in ("the", "old", "man")]
        result = self.prior.query(history)
        self.assertEqual(result.ids[0], self.walked)

    def test_empty_history_is_cold_start(self):
        result = self.prior.query([])
        self.assertEqual(result.scores.shape, (4096,))
        self.assertGreaterEqual(len(result.ids), 1)
        self.assertIn("unigram", result.sources)
        self.assertAlmostEqual(float(result.scores.max()), 1.0, places=5)

    def test_candidate_features(self):
        rows = self.prior.candidate_features(
            ["the", "old", "man"],
            [self.walked, self.sat, 0])
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["construction_rank"], 0.0)
        self.assertEqual(rows[0]["construction_rank_inv"], 1.0)
        self.assertAlmostEqual(rows[0]["ling_score"], 1.0, places=5)
        for row in rows:
            self.assertIn("pos_rank_inv", row)
            self.assertIn("tag_VERB", row)

    def test_query_vector_helper(self):
        vector = self.prior.query_vector(["the", "old", "man"])
        self.assertEqual(vector.shape, (4096,))
        self.assertEqual(int(vector.argmax()), self.walked)


if __name__ == "__main__":
    unittest.main()
