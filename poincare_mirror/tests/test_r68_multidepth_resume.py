"""R68 must preserve R66's appended optimizer order on checkpoint recovery."""
from dataclasses import asdict
from pathlib import Path
import sys
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from poincare_frontier import Config,MirrorLM,load_v25_compatible
from train_r66_causal_book import cache_causal_book,train_joint_only
from train_r68_multidepth import restore_joint_optimizer,audit_inherited


class HistoricalAdamOrderTests(unittest.TestCase):
    def test_append_then_recover_last_joint_moment(self):
        torch.set_num_threads(1)
        torch.manual_seed(68)
        cfg=Config(dim=16,layers=2,heads=2,pairs=4,topk_pairs=1,
                   expert_hidden=16,context=16,moe_every=2,symbiotic_rank=8,
                   micro_meta=True,simplex_transfer=True,dispatch_mode='gather',
                   balance_weight=0.,pair_successor_copy=True,
                   pair_successor_conditional=True)
        original=MirrorLM(cfg).eval()
        optim=torch.optim.AdamW(original.parameters(),lr=.001)
        original.enable_joint_pair_successor(optim,key_mode='hierarchical')
        tokens=torch.tensor(([65,32,66,67,68,32,65,66]*6)+[256],dtype=torch.long)
        records=cache_causal_book(original,tokens,'TRAIN:0',context=16,
                                  chunk=4,warmup=8,positions=16,history=128)
        train_joint_only(original,optim,records,updates=3,lr=.003)
        state={'model':original.state_dict(),'optimizer':optim.state_dict()}
        new_cfg=asdict(original.cfg)
        new_cfg['pair_successor_space_rank_guard']=True
        restored=MirrorLM(Config(**new_cfg)).eval()
        load_v25_compatible(restored,state['model'])
        resumed=restore_joint_optimizer(restored,state)
        joint=restored.pair_successor.joint_weight
        self.assertEqual(tuple(resumed.state[joint]['exp_avg'].shape),(4,))
        self.assertTrue(torch.equal(resumed.state[joint]['exp_avg'],
            state['optimizer']['state'][state['optimizer']['param_groups'][0]['params'][-1]]['exp_avg']))
        records2=cache_causal_book(restored,tokens,'TRAIN:1',context=16,
                                   chunk=4,warmup=8,positions=16,history=128)
        train_joint_only(restored,resumed,records2,updates=2,lr=.003)
        audit=audit_inherited(state,restored,resumed)
        self.assertTrue(audit['trained_joint_parameter_only'])


if __name__=='__main__':
    unittest.main()
