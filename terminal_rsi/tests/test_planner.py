"""End-to-end multi-step, retry, learned recovery and crash-resume tests."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

from terminal_rsi.planner import TerminalPlan, run_plan
from terminal_rsi.terminal_agent import FeedbackPolicy


class PlanTests(unittest.TestCase):
    def plan(self):
        return TerminalPlan.parse({
            'plan_id':'two-phase-task',
            'goal':'Write an observed file and retrieve the exact value',
            'steps':[
                {'family':'create-output',
                 'choices':[[sys.executable,'-c',
                             "from pathlib import Path; Path('state.txt').write_text('hello'); print('written')"]],
                 'expected_stdout':'written'},
                {'family':'read-output',
                 'choices':[[sys.executable,'-c',"print('not hello')"],
                            [sys.executable,'-c',
                             "from pathlib import Path; print(Path('state.txt').read_text())"]],
                 'expected_stdout':'hello','max_attempts':2}
            ]})

    def test_feedback_replanning_checkpoint_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); checkpoint=root/'combined.json'
            model=FeedbackPolicy()
            first=run_plan(self.plan(),model,root,checkpoint,
                           train=True,max_commands=2)
            self.assertEqual((first['status'],first['completed_steps']),('in_progress',1))
            self.assertTrue(first['credit_pending'])
            self.assertEqual(first['attempts'],2)
            self.assertFalse(first['records'][-1]['verified_success'])
            self.assertEqual(first['records'][-1]['choice'],0)
            self.assertEqual(model.total_updates,2)
            restored=FeedbackPolicy.load(checkpoint)
            self.assertEqual(restored.state(),model.state())
            final=run_plan(self.plan(),restored,root,checkpoint,train=True)
            self.assertTrue(final['resumed'])
            self.assertEqual(final['status'],'succeeded')
            self.assertEqual(final['completed_steps'],2)
            self.assertEqual(final['attempts'],3)
            self.assertEqual(final['successful_attempts'],2)
            self.assertEqual(final['records'][-1]['choice'],1)
            self.assertFalse(final['credit_pending'])
            self.assertEqual(len(final['turn_aware_advantages']),3)
            self.assertEqual(restored.total_updates,3)
            again=run_plan(self.plan(),restored,root,checkpoint,train=False)
            self.assertEqual(again['attempts'],3)  # complete plan cannot rerun
            self.assertEqual(restored.total_updates,3)

    def test_plan_digest_prevents_cross_episode_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); checkpoint=root/'checkpoint.json'
            model=FeedbackPolicy()
            run_plan(self.plan(),model,root,checkpoint,train=False,max_commands=1)
            altered=TerminalPlan.parse({
                'plan_id':'two-phase-task','goal':'Different objective',
                'steps':[{'family':'new','choices':[[sys.executable,'-c',
                          "print('new')"]],'expected_stdout':'new'}]})
            with self.assertRaisesRegex(ValueError,'different plan/root'):
                run_plan(altered,FeedbackPolicy.load(checkpoint),root,checkpoint)
            different_root=root/'different';different_root.mkdir()
            with self.assertRaisesRegex(ValueError,'different plan/root'):
                run_plan(self.plan(),FeedbackPolicy.load(checkpoint),
                         different_root,checkpoint)

    def test_stop_after_exhausting_candidates(self):
        plan=TerminalPlan.parse({
            'plan_id':'fail','goal':'Unsolvable within supplied candidates',
            'steps':[{'family':'bad','choices':[[sys.executable,'-c',"print('no')"]],
                      'expected_stdout':'yes'}]})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);checkpoint=root/'combined.json';model=FeedbackPolicy()
            result=run_plan(plan,model,root,checkpoint,train=False)
            self.assertEqual(result['status'],'failed')
            self.assertEqual(result['attempts'],1)
            self.assertEqual(model.total_updates,0)
            self.assertEqual(run_plan(plan,FeedbackPolicy.load(checkpoint),
                                      root,checkpoint)['attempts'],1)

    def test_one_optimizer_continues_across_new_episodes(self):
        first=TerminalPlan.parse({
            'plan_id':'first','goal':'Establish command preference',
            'steps':[{'family':'shared-choice',
                      'choices':[[sys.executable,'-c',"print('no')"],
                                 [sys.executable,'-c',"print('yes')"]],
                      'expected_stdout':'yes'}]})
        second=TerminalPlan.parse({
            'plan_id':'second','goal':'Reuse previously learned preference',
            'steps':[{'family':'shared-choice',
                      'choices':[[sys.executable,'-c',"print('no')"],
                                 [sys.executable,'-c',"print('yes')"]],
                      'expected_stdout':'yes'}]})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);checkpoint=root/'state.json';model=FeedbackPolicy()
            initial=run_plan(first,model,root,checkpoint,train=True)
            self.assertEqual(initial['status'],'succeeded')
            self.assertEqual(initial['attempts'],2)
            prev=model.total_updates
            resumed=FeedbackPolicy.load(checkpoint)
            next_run=run_plan(second,resumed,root,checkpoint,train=True)
            self.assertEqual(next_run['status'],'succeeded')
            self.assertEqual(next_run['attempts'],1)
            self.assertEqual(next_run['records'][0]['choice'],1)
            self.assertEqual(resumed.total_updates,prev+1)
            self.assertEqual(len(resumed.episode_history),1)
            self.assertEqual(resumed.episode_history[0]['plan_id'],'first')

    def test_new_proposals_use_only_observed_failures_and_resume(self):
        plan=TerminalPlan.parse({
            'plan_id':'proposed-fix','goal':'Find the correct terminal output',
            'steps':[{'family':'attempt-fix',
                      'choices':[[sys.executable,'-c',"print('wrong')"]],
                      'expected_stdout':'fixed','max_generated':2}]})
        class ErrorDrivenProposer:
            calls=0
            def propose(self,goal,observed):
                self.calls+=1
                self_check=observed[-1]['execution']['stdout'].strip()
                if self_check!='wrong':raise AssertionError('No relevant failure observed')
                return ((sys.executable,'-c',"print('almost')"),
                        (sys.executable,'-c',"print('fixed')"))
        class MustNotCallAgain:
            def propose(self,*args):
                raise AssertionError('Restart must recover saved proposals')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);checkpoint=root/'policy.json'
            proposer=ErrorDrivenProposer();model=FeedbackPolicy()
            first=run_plan(plan,model,root,checkpoint,train=True,
                           proposer=proposer,max_commands=2)
            self.assertEqual(first['status'],'in_progress')
            self.assertEqual([x['source'] for x in first['records']],
                             ['preplanned','proposed'])
            self.assertEqual(proposer.calls,1)
            recovered=FeedbackPolicy.load(checkpoint)
            result=run_plan(plan,recovered,root,checkpoint,train=True,
                            proposer=MustNotCallAgain())
            self.assertEqual(result['status'],'succeeded')
            self.assertEqual(result['attempts'],3)
            self.assertEqual(result['records'][-1]['source'],'proposed')
            self.assertTrue(result['records'][-1]['verified_success'])
            self.assertEqual(recovered.total_updates,3)


if __name__=='__main__':unittest.main()
