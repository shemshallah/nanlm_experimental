"""R74 factual-vs-corrupted same-address successor readout tests."""
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pair_successor_memory import PairSuccessorTrust
from book_stream_eval import evaluate_causal_book
from poincare_frontier import Config,MirrorLM


class R74Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(74)
        self.m=PairSuccessorTrust(conditional_content=True,joint_content=True,
                                  future_content=True)
        self.base=torch.randn(1,4,257)
        e=torch.zeros_like(self.base)
        n=torch.zeros_like(self.base)
        e[...,65]=.8;e[...,66]=.2
        n[...,66]=.7;n[...,67]=.3
        cov=torch.full((1,4),.6)
        hit=torch.full((1,4),3.)
        self.e=(e,cov,hit)
        self.n=(n,cov.clone(),hit.clone())

    def test_zero_parity_old_weights_preserved(self):
        before={k:v.clone() for k,v in self.m.state_dict().items()}
        self.m.enable_counterfactual_content()
        self.assertTrue(torch.equal(self.base,
                         self.m.counterfactual_forward(self.base,self.e,self.n)))
        for key,value in before.items():
            self.assertTrue(torch.equal(value,self.m.state_dict()[key]))

    def test_mass_and_space_eos_protected(self):
        self.m.enable_counterfactual_content()
        with torch.no_grad(): self.m.counterfactual_weight.fill_(.8)
        out=self.m.counterfactual_forward(self.base,self.e,self.n)
        self.assertGreater((out-self.base).abs().max().item(),.01)
        for b in (32,256):self.assertTrue(torch.equal(out[...,b],self.base[...,b]))
        mask=torch.ones(257,dtype=torch.bool);mask[[32,256]]=False
        self.assertTrue(torch.allclose(torch.logsumexp(out[...,mask],-1),
                                      torch.logsumexp(self.base[...,mask],-1),atol=1e-6))

    def test_invalid_negative_rejected(self):
        self.m.enable_counterfactual_content()
        bad=(self.n[0],self.n[1]*.9,self.n[2])
        with self.assertRaisesRegex(ValueError,'matching control locations'):
            self.m.counterfactual_forward(self.base,self.e,bad)

    def test_gradients_update_only_new_weight(self):
        p=self.m.enable_counterfactual_content()
        p.data.fill_(.2)
        loss=torch.nn.functional.cross_entropy(
            self.m.counterfactual_forward(self.base,self.e,self.n).flatten(0,1),
            torch.full((4,),65,dtype=torch.long))
        loss.backward()
        self.assertGreater(p.grad.abs().sum().item(),0)

    def test_book_evaluator_parity_with_neutral_weight(self):
        cfg=Config(dim=16,layers=2,heads=2,pairs=4,topk_pairs=1,
                   expert_hidden=16,context=16,moe_every=2,symbiotic_rank=8,
                   micro_meta=True,simplex_transfer=True,dispatch_mode='gather',
                   balance_weight=0.,pair_successor_copy=True,
                   pair_successor_conditional=True,pair_successor_joint=True,
                   pair_successor_future=True)
        model=MirrorLM(cfg).eval()
        model.pair_successor.enable_counterfactual_content()
        tokens=torch.tensor(([65,66,67,32,68,69,70,32]*5),dtype=torch.long)
        old=evaluate_causal_book(model,tokens,'R74-toy',context=16,chunk=4,
                                 score_start=8,score_positions=16)
        new=evaluate_causal_book(model,tokens,'R74-toy',context=16,chunk=4,
                       score_start=8,score_positions=16,counterfactual_readout=True)
        for key in ('ce_1','ce_2','ce_3','top5_1','top5_2','top5_3'):
            self.assertAlmostEqual(old['exact'][key],new['exact'][key],places=7)


if __name__=='__main__':unittest.main()
