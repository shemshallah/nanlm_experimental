"""R62: exact causal cross-window FA2 stream, replay and model integration."""
from dataclasses import asdict
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from book_successor_stream import CausalBookSuccessor
from pair_successor_memory import PairSuccessorTrust,retrieve_pair_successors
from poincare_frontier import Config,MirrorLM


class PersistentBookMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @staticmethod
    def sample():
        torch.manual_seed(620)
        vocab=torch.tensor([32,65,66,67,68,69])
        observed=vocab[torch.randint(0,len(vocab),(48,))]
        candidates=torch.tensor([0,1,2,3,254,255,256,257,258,510,765,1020])
        pair_ids=candidates[torch.randint(0,len(candidates),(48,3))]
        return observed,pair_ids

    def test_chunk_parity_with_frozen_vectorized_oracle_all_modes(self):
        seen,pairs=self.sample()
        for mode in CausalBookSuccessor.VALID_MODES:
            for horizon in (1,4,12,48):
                for control in ("exact","rotated"):
                    with self.subTest(mode=mode,h=horizon,control=control):
                        kwargs=dict(horizon=horizon,key_mode=mode,value_control=control)
                        oracle=retrieve_pair_successors(pairs[None],seen[None],
                                                        engine="vectorized",**kwargs)
                        stream=CausalBookSuccessor(**kwargs)
                        stream.begin_book("book-17")
                        chunks=[stream.read_chunk("book-17",seen[a:b],pairs[a:b])
                                for a,b in ((0,9),(9,25),(25,48))]
                        for i,expected in enumerate(oracle):
                            actual=torch.cat([chunk[i] for chunk in chunks],dim=0)
                            torch.testing.assert_close(actual,expected[0],rtol=0,atol=1e-6)

    def test_query_only_negative_controls(self):
        seen,pairs=self.sample()
        for mode in ("pair_byte","head_byte","hierarchical"):
            for addr,byte in ((torch.arange(32640).roll(137),None),
                              (None,torch.arange(257).roll(1)),
                              (torch.arange(32640).roll(137),torch.arange(257).roll(1))):
                kwargs=dict(horizon=16,key_mode=mode,query_map=addr,
                            query_byte_map=byte,value_control="exact")
                expected=retrieve_pair_successors(
                    pairs[None],seen[None],engine="vectorized",**kwargs)
                stream=CausalBookSuccessor(**kwargs)
                stream.begin_book(17)
                actual=stream.read_chunk(17,seen,pairs)
                for x,y in zip(actual,expected):
                    torch.testing.assert_close(x,y[0],rtol=0,atol=1e-6)

    def test_snapshot_replay_and_book_barrier(self):
        seen,pairs=self.sample()
        stream=CausalBookSuccessor(horizon=64,key_mode="hierarchical")
        stream.begin_book("novel-A")
        first=stream.read_chunk("novel-A",seen[:19],pairs[:19])
        snapshot=stream.state_dict()
        remaining=stream.read_chunk("novel-A",seen[19:],pairs[19:])
        restored=CausalBookSuccessor(horizon=64,key_mode="hierarchical")
        restored.load_state_dict(snapshot)
        after=restored.read_chunk("novel-A",seen[19:],pairs[19:])
        for x,y in zip(remaining,after):
            self.assertTrue(torch.equal(x,y))
        with self.assertRaisesRegex(ValueError,"Begin"):
            restored.read_chunk("novel-B",seen[:2],pairs[:2])
        restored.begin_book("novel-B")
        empty=restored.read_chunk("novel-B",seen[:1],pairs[:1])
        self.assertEqual(empty[0].sum().item(),0)
        self.assertEqual(restored.state_dict()["book_id"],"novel-B")
        self.assertEqual(first[0].shape,(19,257))

    def test_suffix_causality_and_long_window_memory_beyond_chunk(self):
        seen,pairs=self.sample()
        stream=CausalBookSuccessor(horizon=48,key_mode="hierarchical")
        stream.begin_book("one")
        original=stream.read_chunk("one",seen,pairs)
        changed=seen.clone()
        changed[30:]=98
        changed_pairs=pairs.clone()
        changed_pairs[30:]=32500
        replay=CausalBookSuccessor(horizon=48,key_mode="hierarchical")
        replay.begin_book("one")
        altered=replay.read_chunk("one",changed,changed_pairs)
        for x,y in zip(original,altered):
            self.assertTrue(torch.equal(x[:30],y[:30]))
        # A historical match from the first 20 positions is read in a
        # one-position second chunk; no context-window reconstruction occurs.
        special=torch.tensor([65,66]+[90]*40+[65,66])
        pair_ids=torch.full((44,1),5000,dtype=torch.long)
        ids=CausalBookSuccessor(horizon=64,key_mode="pair_byte")
        ids.begin_book("long")
        ids.read_chunk("long",special[:43],pair_ids[:43])
        memory=ids.read_chunk("long",special[43:],pair_ids[43:])
        self.assertGreater(memory[1][0].item(),0)
        self.assertGreater(memory[0][0,90].item(),0)

    def test_external_book_memory_reaches_existing_trainable_joint_rail(self):
        seen=torch.tensor([65,66]+[90]*38+[65,68,90,90],dtype=torch.long)
        pairs=torch.full((44,1),254,dtype=torch.long)
        stream=CausalBookSuccessor(horizon=64,key_mode="pair_byte")
        stream.begin_book("author-one")
        evidence=stream.read_chunk("author-one",seen,pairs)
        # On t=40, x[t]=65 matches u=0 in the long prefix but not u>=8.
        self.assertGreater(evidence[1][40].item(),0)
        logits=torch.zeros(1,44,257)
        trust=PairSuccessorTrust(conditional_content=True,joint_content=True)
        trust.joint_key_mode="pair_byte"
        with torch.no_grad():
            trust.joint_weight.copy_(torch.tensor([1.,.2,0.,0.]))
        local,_=trust(logits,pairs[None],seen[None])
        outside,_=trust(logits,pairs[None],seen[None],
                        joint_memory_override=tuple(e[None] for e in evidence))
        self.assertFalse(torch.equal(local[:,40],outside[:,40]))
        self.assertTrue(torch.equal(local[...,32],outside[...,32]))
        self.assertTrue(torch.equal(local[...,256],outside[...,256]))
        mask=torch.ones(257,dtype=torch.bool)
        mask[[32,256]]=False
        self.assertTrue(torch.allclose(
            torch.logsumexp(local[...,mask],dim=-1),
            torch.logsumexp(outside[...,mask],dim=-1),atol=1e-6))
        target=seen.roll(-1)[None]
        loss=torch.nn.functional.cross_entropy(outside[:,:-1].reshape(-1,257),
                                                target[:,:-1].reshape(-1))
        loss.backward()
        self.assertGreater(trust.joint_weight.grad.abs().sum().item(),0)
        with self.assertRaisesRegex(ValueError,"shape"):
            trust(logits,pairs[None],seen[None],
                  joint_memory_override=(evidence[0],evidence[1],evidence[2]))

    def test_model_forward_accepts_external_history_without_state_changes(self):
        cfg=Config(dim=16,layers=2,heads=2,pairs=4,topk_pairs=1,
                   expert_hidden=16,context=16,moe_every=2,symbiotic_rank=8,
                   micro_meta=True,simplex_transfer=True,dispatch_mode="gather",
                   balance_weight=0.,pair_successor_copy=True,
                   pair_successor_conditional=True,pair_successor_joint=True)
        torch.manual_seed(621)
        model=MirrorLM(cfg).eval()
        with torch.no_grad():
            model.pair_successor.joint_weight.copy_(torch.tensor([1.,.2,0.,0.]))
        seen=torch.tensor([[65,66,65,66,65,66,65,66]],dtype=torch.long)
        copy=torch.zeros(1,8,257)
        copy[...,66]=1.
        cov=torch.ones(1,8)
        pos=torch.ones(1,8)
        before={name:x.clone() for name,x in model.state_dict().items()}
        model(seen,joint_memory_override=(copy,cov,pos))
        for name,x in before.items():
            self.assertTrue(torch.equal(x,model.state_dict()[name]))
        restored=MirrorLM(Config(**asdict(model.cfg))).eval()
        restored.load_state_dict(model.state_dict(),strict=True)


if __name__=="__main__":
    unittest.main()
