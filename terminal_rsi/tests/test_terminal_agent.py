"""Real-process smoke tests for the bounded Terminal-RSI feedback loop."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

from terminal_rsi.terminal_agent import CommandTask, FeedbackPolicy, attempt, execute


class LiveTerminalTests(unittest.TestCase):
    def task(self):
        return CommandTask.parse({
            'task_id':'choose-observed-command','family':'probe-family',
            'choices':[[sys.executable,'-c','print("wrong")'],
                       [sys.executable,'-c','print("ready")']],
            'expected_stdout':'ready', 'expected_code':0})

    def test_real_process_feedback_learns_and_persists(self):
        task=self.task();policy=FeedbackPolicy()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            initial=policy.probabilities(task)
            first=attempt(task,policy,root,train=True)
            second=attempt(task,policy,root,train=True)
            self.assertFalse(first['verified_success'])
            self.assertTrue(second['verified_success'])
            self.assertTrue(first['policy_updated'])
            self.assertEqual(policy.total_updates,2)
            self.assertGreater(policy.probabilities(task)[1],initial[1])
            saved=root/'agent-checkpoint.json';policy.save(saved)
            restored=FeedbackPolicy.load(saved)
            self.assertEqual(restored.state(),policy.state())
            self.assertEqual(restored.choose(task)[0],1)
            previous=restored.rows[task.family]['step']
            read_only=attempt(task,restored,root,train=False)
            self.assertTrue(read_only['verified_success'])
            self.assertFalse(read_only['policy_updated'])
            self.assertEqual(restored.rows[task.family]['step'],previous)
            self.assertNotEqual(read_only['receipt_digest'],second['receipt_digest'])
            damaged=json.loads(saved.read_text())
            damaged['state']['total_updates']=999
            saved.write_text(json.dumps(damaged))
            with self.assertRaisesRegex(ValueError,'integrity'):
                FeedbackPolicy.load(saved)

    def test_rejects_shell_strings_and_frozen_task_shape(self):
        with self.assertRaisesRegex(ValueError,'argument arrays'):
            CommandTask.parse({'task_id':'x','family':'x',
                               'choices':['echo ready'],'expected_stdout':'ready'})
        actor=FeedbackPolicy()
        actor.probabilities(self.task())
        another=CommandTask.parse({'task_id':'x','family':'probe-family',
                                   'choices':[['echo','ready']], 'expected_stdout':'ready'})
        with self.assertRaisesRegex(ValueError,'Changing command count'):
            actor.choose(another)

    def test_missing_command_and_timeout_are_observed_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing=execute(Path(tmp),('command-definitely-does-not-exist',))
            self.assertIsNone(missing['returncode'])
            self.assertTrue(missing['stderr'])
            timeout=execute(Path(tmp),(sys.executable,'-c','import time;time.sleep(3)'),
                            timeout=.05)
            self.assertTrue(timeout['timed_out'])
            self.assertNotEqual(timeout['returncode'],0)


if __name__=='__main__':unittest.main()
