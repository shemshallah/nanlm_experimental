"""Causal byte-proposal ranking contracts; no pretrained donor is assumed."""
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from terminal_rsi.byte_specialist.causal_echo import CausalEcho
from terminal_rsi.byte_specialist.contextual import ContextualResidual
from terminal_rsi.byte_specialist.feedback_core import FeedbackRSI
from terminal_rsi.byte_specialist.residual import ResidualPolicy
from terminal_rsi.byte_specialist.trustgate import TrustGate
from terminal_rsi.byte_specialist.command_scoring import (
    CausalRSIPredictor, LowBPCRanker, BPCRankedProposer, causal_bpc,
    command_bytes, observed_context)
from terminal_rsi.planner import TerminalPlan, run_plan
from terminal_rsi.terminal_agent import FeedbackPolicy


class FixedDonor:
    def __init__(self,p):
        self.p=p
        self.events=[]
    def predict(self):
        self.events.append('predict')
        return self.p.copy()
    def observe(self,byte):
        self.events.append(('observe',byte))


class CommandScoringTests(unittest.TestCase):
    def model(self):
        return FeedbackRSI(CausalEcho(TrustGate(
            ContextualResidual(ResidualPolicy()))))

    def test_bytewise_probability_is_recorded_before_observation(self):
        uniform=np.ones(256,np.float64)/256
        donor=FixedDonor(uniform)
        predictor=CausalRSIPredictor(self.model(),donor,b'prior')
        payload=command_bytes(('echo','hello'))
        self.assertAlmostEqual(causal_bpc(predictor,payload),8.,places=10)
        self.assertEqual(len(donor.events),2*len(payload))
        self.assertEqual(donor.events[0],'predict')
        self.assertEqual(donor.events[1],('observe',payload[0]))
        with self.assertRaisesRegex(ValueError,'not pending'):
            predictor.observe(12)

    def test_low_bpc_model_can_rank_but_cannot_verify_own_actions(self):
        p=np.ones(256,np.float64)
        p[ord('a')]=1000.
        p/=p.sum()
        model=self.model()
        ranker=LowBPCRanker(model,lambda prefix:FixedDonor(p))
        choices=((sys.executable,'-c',"print('zzzz')"),
                 (sys.executable,'-c',"print('aaaa')"))
        scores=ranker.score('Print the requested value',(),choices)
        self.assertLess(scores[1]['bpc'],scores[0]['bpc'])
        class Base:
            def propose(self,goal,results):
                return choices
        proposed=BPCRankedProposer(Base(),ranker)
        self.assertEqual(proposed.propose('goal',())[0],choices[1])
        plan=TerminalPlan.parse({
            'plan_id':'byte-rank','goal':'Print aaaa',
            'steps':[{'family':'first-try',
                      'choices':[[sys.executable,'-c',"print('failed')"]],
                      'expected_stdout':'aaaa','max_generated':2}]})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);actor=FeedbackPolicy()
            final=run_plan(plan,actor,root,root/'actor.json',proposer=proposed)
            self.assertEqual(final['status'],'succeeded')
            self.assertEqual(final['attempts'],2)
            self.assertEqual(final['records'][0]['source'],'preplanned')
            self.assertEqual(final['records'][1]['source'],'proposed')
            self.assertTrue(final['records'][1]['verified_success'])
            self.assertEqual(actor.total_updates,0)

    def test_observed_context_contains_no_unobserved_target(self):
        seen={'argv':['python3','-V'],
              'execution':{'returncode':1,'stdout':'old stdout',
                           'stderr':'old stderr'}}
        context=observed_context('Goal',(seen,))
        self.assertIn(b'old stdout',context)
        self.assertIn(b'old stderr',context)
        self.assertNotIn(b'future target',context)


if __name__=='__main__':unittest.main()
