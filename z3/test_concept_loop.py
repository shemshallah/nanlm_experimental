#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from concept_loop import (
    CompositionVocabulary,
    ConceptEncoder,
    ConceptLoop,
    IncrementalCompositionDecoder,
    SpanSidecar,
)
from layered_tokenizer import LayeredTokenizer
from v3_model import V3Config, V3LM


TEXT = (
    "The quick brown fox jumps. "
    "A second sentence has $E=mc^2$ and 123."
)


class ConceptLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = LayeredTokenizer(
            prefer_native=False, sentence_max_chars=128
        )

    def test_aligned_sidecar(self):
        sidecar = SpanSidecar.from_text(TEXT, self.tokenizer)
        n = len(sidecar.word_ids)
        self.assertGreater(n, 5)
        self.assertEqual(sidecar.byte_start.shape, (n,))
        self.assertEqual(sidecar.event_ids.shape, (n,))
        self.assertTrue((sidecar.word_ids >= 0).all())
        self.assertTrue((sidecar.sentence_ids >= 0).all())
        self.assertGreater(int(sidecar.word_ids.max()), 0)
        self.assertGreater(int(sidecar.sentence_ids.max()), 0)
        self.assertGreater(int(sidecar.math_kind.max()), 0)
        self.assertIsNotNone(sidecar.token_bytes)
        self.assertEqual(sidecar.token_bytes.shape[0], n)
        self.assertEqual(sidecar.token_bytes.shape[1], 48)
        valid_bytes = sidecar.token_bytes[sidecar.token_bytes >= 0]
        self.assertTrue((valid_bytes <= 255).all())

    def test_sidecar_from_token_ids(self):
        encoded = self.tokenizer.encode(TEXT, add_document_frame=False)
        from_ids = SpanSidecar.from_token_ids(
            encoded.subword.ids, self.tokenizer
        )
        from_text = SpanSidecar.from_text(TEXT, self.tokenizer)
        self.assertEqual(len(from_ids.word_ids), len(from_text.word_ids))
        self.assertTrue(torch.equal(from_ids.word_ids, from_text.word_ids))
        self.assertTrue(torch.equal(from_ids.sentence_ids, from_text.sentence_ids))
        self.assertTrue(torch.equal(from_ids.math_kind, from_text.math_kind))

    def test_composition_vocabulary_roundtrip(self):
        events = ["<word_start>", "<word_end>", "<sentence_end>", "<compose>"]
        ids = CompositionVocabulary.encode(events)
        self.assertEqual(CompositionVocabulary.decode(ids), events)
        self.assertEqual(CompositionVocabulary.manifest()["size"], 7)

    def test_sidecar_collate(self):
        a = SpanSidecar.from_text("One short sentence.", self.tokenizer)
        b = SpanSidecar.from_text("A much longer second sentence here.", self.tokenizer)
        batch = SpanSidecar.collate([a, b])
        self.assertEqual(batch.word_ids.shape[0], 2)
        self.assertEqual(
            batch.word_ids.shape[1],
            max(a.word_ids.numel(), b.word_ids.numel()),
        )
        self.assertIsNotNone(batch.token_bytes)
        self.assertEqual(
            batch.token_bytes.shape,
            (2, batch.word_ids.shape[1], 48),
        )

    def test_concept_encoder_and_loop(self):
        sidecar = SpanSidecar.from_text(TEXT, self.tokenizer)
        encoder = ConceptEncoder(32)
        x = torch.randn(1, len(sidecar.word_ids), 32)
        concept, global_ctx = encoder(x, sidecar)
        self.assertEqual(concept.shape, x.shape)
        self.assertEqual(global_ctx.shape, (1, 32))
        self.assertTrue(torch.isfinite(concept).all())

        loop = ConceptLoop(
            d_model=32, vertices=256, dim=16,
            top_vertices=8, top_pairs=4, top_triples=4,
        )
        route = torch.zeros(1, 32)
        concept, object_ctx, loop_ctx = loop(x, sidecar, route)
        self.assertEqual(concept.shape, x.shape)
        self.assertEqual(object_ctx.shape, x.shape)
        self.assertEqual(loop_ctx.shape, (1, 32))
        self.assertTrue(torch.isfinite(loop_ctx).all())
        self.assertEqual(loop.counts()["pairs"], 32640)
        self.assertEqual(loop.counts()["triples"], 2763520)
        self.assertEqual(loop.last_object_ids.shape[0], 1)
        self.assertTrue(
            (loop.last_object_ids < 32640).all()
        )

    def test_model_sidecar_feedback_and_generation(self):
        encoded = self.tokenizer.encode(TEXT, add_document_frame=False)
        sidecar = SpanSidecar.from_token_ids(
            encoded.subword.ids, self.tokenizer
        )
        cfg = V3Config(
            vocab_size=64060, ctx=32, max_ctx=32, d_model=32,
            n_layers=2, n_heads=4, n_kv_heads=2, swiglu=64,
            window=16, global_every=2, moe_layers=1, n_experts=8,
            topk=2, shared_experts=1, expert_hidden=64, mtp=1,
            factorized_embedding=True, embedding_rank=16,
            use_combinatorial_memory=False, use_concept_loop=True,
            composition_head=True, moe_hd_dim=64,
            concept_loop_dim=32, concept_top_vertices=8,
            concept_top_pairs=4, concept_top_triples=2,
            dropout=0.0, use_memory=False,
        )
        model = V3LM(cfg)
        tokens = torch.tensor(
            [encoded.subword.ids], dtype=torch.long
        )
        logits_a, _mtp, aux_a = model(tokens, sidecar=sidecar)
        logits_b, _mtp, aux_b = model(tokens, sidecar=None)
        self.assertEqual(
            tuple(aux_a["composition_logits"].shape),
            (1, tokens.size(1), 7),
        )
        self.assertFalse(torch.allclose(logits_a, logits_b))
        loss = logits_a.float().logsumexp(-1).mean()
        loss.backward()
        generated, events = model.generate(
            tokens[:, :4], max_new_tokens=2, temperature=0.8, seed=4
        )
        self.assertEqual(generated.shape, (1, 6))
        self.assertEqual(len(events), 2)
        generated2, payload = model.generate(
            tokens[:, :4], max_new_tokens=2, temperature=0.8, seed=4,
            tokenizer=self.tokenizer,
        )
        self.assertEqual(generated2.shape, (1, 6))
        self.assertIn("model", payload)
        self.assertIn("tokenizer", payload)
        self.assertIn("objects", payload)
        self.assertEqual(len(payload["objects"]), 2)
        self.assertTrue(
            all(0 <= object_id < 32640 for object_id in payload["objects"][0]["pairs"])
        )
        self.assertIn("triples", payload["objects"][0])
        self.assertTrue(
            all(0 <= object_id < 2763520 for object_id in payload["objects"][0]["triples"])
        )

    def test_incremental_composition(self):
        encoded = self.tokenizer.encode(TEXT, add_document_frame=False)
        decoder = IncrementalCompositionDecoder(self.tokenizer.subword)
        events = []
        for token_id in encoded.subword.ids:
            events.extend(decoder.feed(token_id))
        self.assertEqual(decoder.text, TEXT)
        self.assertTrue(any(kind == "<word>" for _, kind in events))
        self.assertTrue(any(kind == "<sentence_end>" for _, kind in events))
        self.assertTrue(any(kind == "<math>" for _, kind in events))


if __name__ == "__main__":
    unittest.main(verbosity=2)
