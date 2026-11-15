"""Actor/critic checkpoint protection, geometric feedback and training smoke."""
import copy
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from poincare_frontier import Config,MirrorLM
from poincare_rsi_adapter.torch_adapter import Batch,attach,restore,critic_warmup,sft_loss,ppo_loss,candidate


def config():
    return Config(dim=32,layers=2,heads=4,pairs=4,topk_pairs=2,expert_hidden=16,
                  context=16,moe_every=2,hierarchy_rank=8,symbiotic_rank=8,
                  micro_meta=True,simplex_transfer=True,reciprocal_scales=True)


class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_migration_zero_drift_and_adam_preservation(self):
        torch.manual_seed(1)
        ids=torch.randint(0,256,(1,9))
        original=MirrorLM(config()).eval()
        optimizer=torch.optim.AdamW(original.parameters())
        original.loss_with_future(ids)['loss'].backward();optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        saved={'model':copy.deepcopy(original.state_dict()),
               'optimizer':copy.deepcopy(optimizer.state_dict()),'step':7}
        with torch.no_grad(): reference=original(ids)[0]
        model=MirrorLM(config()).eval(); new_opt=torch.optim.AdamW(model.parameters())
        bridge=restore(model,new_opt,saved,32)
        with torch.no_grad(): migrated,values=bridge(ids)
        self.assertTrue(torch.equal(reference,migrated))
        self.assertTrue(torch.equal(values,torch.zeros_like(values)))
        self.assertEqual(len(new_opt.state),len(optimizer.state))
        for key in saved['optimizer']['state']:
            for k,v in saved['optimizer']['state'][key].items():
                self.assertTrue(torch.equal(v,new_opt.state_dict()['state'][key][k]))
        result=candidate(bridge,new_opt,saved,'RSR-1',1,0)
        self.assertEqual(result['step'],7)
        self.assertEqual(len(result['optimizer']['param_groups']),2)
        self.assertFalse(result['poincare_rsi_legacy']['promoted'])
        self.assertEqual(len(saved['optimizer']['param_groups']),1)

    def test_masked_sft_and_critic_warmup(self):
        torch.manual_seed(2)
        model=MirrorLM(config());opt=torch.optim.AdamW(model.parameters())
        ids=torch.randint(0,256,(1,9))
        model.loss_with_future(ids)['loss'].backward();opt.step()
        opt.zero_grad(set_to_none=True)
        old_weights=copy.deepcopy(model.state_dict())
        old_state=copy.deepcopy(opt.state_dict()['state'])
        bridge=attach(model,opt,32)
        mask=torch.zeros_like(ids,dtype=torch.bool);mask[:,[2,5,8]]=True
        done=torch.zeros_like(mask);done[:,8]=True
        reward=torch.zeros_like(ids,dtype=torch.float);reward[:,8]=1
        turns=torch.arange(ids.shape[1])[None].long()
        b=Batch(ids,ids.clone(),mask,turns,reward,done,torch.zeros_like(reward))
        losses=critic_warmup(bridge,opt,b,torch.ones_like(reward),4)
        self.assertEqual(len(losses),4)
        for k,v in old_weights.items(): self.assertTrue(torch.equal(v,model.state_dict()[k]),k)
        for ident,state in old_state.items():
            for k,v in state.items(): self.assertTrue(torch.equal(v,opt.state_dict()['state'][ident][k]))
        original_sft=sft_loss(bridge,b)
        other_labels=b.labels.clone();other_labels[~mask]=256
        other=Batch(ids,other_labels,mask,turns,reward,done,b.old_logprobs)
        self.assertTrue(torch.equal(original_sft,sft_loss(bridge,other)))
        original_sft.backward()
        self.assertTrue(all(p.grad is None for p in bridge.critic.parameters()))
        opt.zero_grad(set_to_none=True)
        loss,stats=ppo_loss(bridge,b)
        self.assertEqual(stats['generated'],3)
        self.assertEqual(stats['observations'],6)
        loss.backward()
        self.assertGreater(float(bridge.critic.weight.grad.abs().sum()),0.)

    def test_pilot_train_and_report(self):
        from scripts.poincare_rsi_legacy_pilot import run
        report=run(sft_steps=4,ppo_steps=4)
        self.assertEqual(report['verified_replay_count'],6)
        self.assertEqual(report['critic_warmup_updates'],4)
        self.assertEqual(report['ppo_updates'],4)
        self.assertTrue(report['promotion'].startswith('FORBIDDEN:'))
        for name in ('metrics_before','metrics_after_rsr','metrics_after_rpr'):
            self.assertIn('top1_percent',report[name])
            self.assertIn('top5_percent',report[name])
            self.assertIn('bits_per_byte',report[name])
            self.assertEqual(report[name]['bytes_per_token'],1.)


if __name__=='__main__': unittest.main()
