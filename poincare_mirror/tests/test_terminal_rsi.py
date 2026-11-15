"""Pure-Python verification gates for Terminal-RSI adaptation."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from terminal_rsi.core import *


class CoreTests(unittest.TestCase):
    def test_fdr_spr_and_p90(self):
        c=Calibration.from_history({'f':list(range(1,11))})
        self.assertEqual(c.divisor('f'),9)
        self.assertEqual(c.divisor('unknown'),82)
        bad=Outcome('f',False,6,8,True)
        good=Outcome('f',True,6,8,True)
        self.assertAlmostEqual(fdr_reward(bad,c),6/9)
        self.assertLess(spr_reward(bad,c),1)
        self.assertGreaterEqual(spr_reward(good,c),1)
        self.assertEqual(fdr_reward(Outcome('f',False,6,8,False),c),0)

    def test_frozen_family_adaptation(self):
        c=Calibration({'a':40,'b':8})
        n=c.adapted(high_failure=['a'],long_low=['a','b'])
        self.assertEqual((n.divisor('a'),n.divisor('b')),(80,2))
        self.assertEqual(c.divisor('a'),40)

    def test_reward_rejects_invalid_checks(self):
        with self.assertRaises(ValueError): Outcome('f',False,10,9,True)
        with self.assertRaises(ValueError): Calibration({'f':0})

    def test_same_turn_and_cross_turn_advantage(self):
        a=[Action(0,0),Action(0,0),Action(1,0,2,True)]
        result=tb_gae(a,.5,.3)
        self.assertEqual(result.advantages,(.3,.3,2.))
        same=tb_gae([Action(0,0),Action(0,0),Action(0,0,1,True)],.5,.1)
        self.assertEqual(same.advantages,(1.,1.,1.))

    def test_chunk_equivalence(self):
        a=[Action(0,.2,.1),Action(0,.3,0),Action(1,-.1,.4),Action(2,.2,0),Action(2,.3,1,True)]
        full=tb_gae(a,.87,.62)
        for cut in range(1,len(a)):
            partial=tb_gae(a[:cut],.87,.62,Continuation(a[cut].value,full.advantages[cut],a[cut].turn_id))
            for x,y in zip(partial.advantages,full.advantages): self.assertAlmostEqual(x,y)
        with self.assertRaises(ValueError): tb_gae(a[:-1])

    def test_public_replay_and_critic_handoff(self):
        donor=Trajectory('task','shell','donor','1',(Event('user','help',0),Event('assistant','fix',1)),True)
        target=Trajectory('task','shell','target','2',donor.events,True)
        def verify(t): return Receipt(t.task_id,t.run_id,t.harness_id,t.digest(),True)
        result=adopt_rewrite(donor,lambda task,actions: target,verify)
        self.assertEqual(result.loss_mask,(False,True))
        with self.assertRaises(ValueError):
            adopt_rewrite(donor,lambda *args: donor,verify)
        with self.assertRaises(ValueError):
            adopt_rewrite(Trajectory('task','shell','d','1',(Event('assistant','secret',0,'private'),),True),
                          lambda *args:target,verify)
        with self.assertRaises(ValueError):
            adopt_rewrite(donor,lambda *args:target,lambda t: Receipt('task','2','target','0'*64,True))
        base=Stage('BASE','a'*64,'b'*64,'c'*64)
        rsr=Stage('RSR-1','d'*64,'b'*64,'e'*64,verified_sft=2)
        check_transition(base,rsr)
        check_transition(rsr,Stage('RPR-1','f'*64,'e'*64,'f'*64,critic_warmup=4))
        with self.assertRaises(ValueError): check_transition(base,Stage('RSR-1','f'*64,'f'*64,'f'*64,verified_sft=2))

    def test_promotion_fail_closed_on_nonspace(self):
        base=Gate(.5,3.2,3.1,4.2,4.5,.05,100,2.65,.3,.6)
        good=Gate(.51,3.1,3.05,4.1,4.4,.06,100,2.6,.31,.61)
        self.assertTrue(promote(base,good)[0])
        bad=Gate(.52,3.1,3.05,4.1,4.4,.06,100,2.68,.31,.61)
        self.assertIn('nonspace_ce',promote(base,bad)[1])
        wrong=Gate(.52,3.1,3.05,4.1,4.4,-.01,100,2.6,.31,.61)
        self.assertIn('history_control',promote(base,wrong)[1])


if __name__=='__main__': unittest.main()
