"""CPU optimization invariants on the existing paired MoE contract."""
import unittest
import torch
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from poincare_frontier import Config,MirrorLM,MirrorPairMoE,_rope_trig,update_growth_bias

class CPURound29(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_batched_gram_hyperbolic_distance_matches_original(self):
        torch.manual_seed(29)
        x=MirrorPairMoE.to_ball(torch.randn(43,16))
        c=MirrorPairMoE.to_ball(torch.randn(5,16))
        ours=MirrorPairMoE.hyperbolic_distance(x,c)
        q2=x.square().sum(-1,keepdim=True)
        c2=c.square().sum(-1)[None,:]
        diff=(x[:,None]-c[None]).square().sum(-1)
        reference=torch.acosh((1+2*diff/((1-q2).clamp_min(1e-6)*(1-c2).clamp_min(1e-6))).clamp_min(1+1e-6))
        self.assertTrue(torch.allclose(ours,reference,atol=1e-4,rtol=1e-5))

    def test_rope_cache_immutable_reuse(self):
        a=_rope_trig(128,16,torch.device('cpu'),torch.float32)
        b=_rope_trig(128,16,torch.device('cpu'),torch.float32)
        self.assertTrue(a[0] is b[0] and a[1] is b[1])

    def test_growth_bias_is_continuous_when_rebased(self):
        model=MirrorLM(Config(dim=32,heads=4,layers=2,pairs=8,topk_pairs=2,
                              expert_hidden=16,context=16))
        growth={'from_pairs':4,'to_pairs':8,'start_step':100,'warmup_steps':1000,'initial_bias':-12.}
        update_growth_bias(model,growth,200)
        old=model.blocks[1].ff.pair_bias[4:].clone()
        shorter=dict(growth,start_step=200,warmup_steps=100,initial_bias=float(old[0]))
        update_growth_bias(model,shorter,200)
        self.assertTrue(torch.equal(old,model.blocks[1].ff.pair_bias[4:]))
        update_growth_bias(model,shorter,300)
        self.assertTrue(torch.equal(torch.zeros_like(old),model.blocks[1].ff.pair_bias[4:]))

if __name__=='__main__':unittest.main()

class ContextMixParse(unittest.TestCase):
    def test_context_mixture_probabilities(self):
        from poincare_frontier import parse_context_mix
        lengths, probs = parse_context_mix('128:3,512:1,4096:2',4096)
        self.assertEqual(lengths,[128,512,4096])
        self.assertAlmostEqual(sum(probs),1.)
        self.assertAlmostEqual(probs[0],.5)
        for value in ('1024:0','8192:1','512:1,512:1','512:-1','512:nan'):
            with self.assertRaises(ValueError):parse_context_mix(value,4096)
