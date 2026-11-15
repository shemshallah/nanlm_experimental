"""Known-answer tests; no FA1/FA2 imports or fixtures."""
import hashlib
from pathlib import Path
import unittest

import numpy as np

from terminal_rsi.core import (Action, Continuation, TaskGate,
                               preliminary_task_gate, tb_gae)
from terminal_rsi.standalone import (
    Actor, FAMILIES, Task, measure, run, task_split, verified_answer, verified_rewrite,
)
from terminal_rsi.benchmark import compare


class IsolatedTests(unittest.TestCase):
    def test_verifier_known_answers(self):
        self.assertEqual(verified_answer('COPY_A 2 7'), '2')
        self.assertEqual(verified_answer('COPY_B 2 7'), '7')
        self.assertEqual(verified_answer('NEXT_A 9 0'), '0')
        self.assertEqual(verified_answer('PREV_B 0 0'), '9')
        self.assertEqual(verified_answer('DOUBLE_A 7 1'), '4')
        self.assertEqual(verified_answer('NINE_B 1 7'), '2')
        with self.assertRaises(ValueError): verified_answer('RUN rm -rf')

    def test_task_disjointness_and_full_coverage(self):
        train, dev = task_split(2026)
        self.assertEqual((len(train), len(dev)), (800, 200))
        self.assertEqual(len({t.task_id for t in train + dev}), 1000)
        self.assertEqual({t.family for t in train}, set(FAMILIES))
        self.assertEqual({t.family for t in dev}, set(FAMILIES))
        self.assertEqual((train, dev), task_split(2026))

    def test_verified_replay_has_new_harness(self):
        self.assertEqual(verified_rewrite(Task('NEXT_B', 7, 9), 12), '0')

    def test_numeric_metrics_uniform_actor(self):
        actor = Actor(2)
        actor.weights.fill(0.)
        tasks = (Task('COPY_A', 0, i) for i in range(10))
        measured = measure(actor, tuple(tasks))
        self.assertEqual(measured['tasks'], 10)
        self.assertEqual(measured['top1'], 1.)
        self.assertEqual(measured['top5'], 1.)
        self.assertAlmostEqual(measured['bits_per_action'], np.log2(10), places=5)

    def test_critic_handoff_and_chunk_credit(self):
        sequence = (Action(0, .1, 0.), Action(0, .2, 0.), Action(1, .2, 1., True))
        full = tb_gae(sequence)
        chunk = tb_gae(sequence[:1], continuation=Continuation(sequence[1].value,
                          full.advantages[1], sequence[1].turn_id))
        self.assertAlmostEqual(chunk.advantages[0], full.advantages[0])

    def test_terminal_only_preliminary_gate(self):
        baseline = TaskGate(.2, .4, 200, 10)
        improved = TaskGate(.25, .43, 200, 10)
        self.assertEqual(preliminary_task_gate(baseline, improved), (True, ()))
        dropped = TaskGate(.25, .3, 190, 9)
        passed, failures = preliminary_task_gate(baseline, dropped)
        self.assertFalse(passed)
        self.assertIn('process_reward_regression', failures)
        self.assertIn('family_coverage', failures)
        self.assertIn('verified_task_coverage', failures)

    def test_five_stages_optimizer_continuity_and_no_promotion(self):
        report = run(seed=2026, sft_updates=12, rpr_updates=8, include_checkpoint=True)
        self.assertEqual([x['stage'] for x in report['stages']],
                         ['BASE','RSR-1','RPR-1','RSR-2','RPR-2'])
        self.assertEqual(report['verified_replays'], 24)
        self.assertEqual(report['adam_updates'], 40)
        self.assertEqual(report['heldout_tasks'], 200)
        self.assertFalse(report['benchmark_eligible'])
        self.assertFalse(report['candidate_promoted'])
        self.assertIsNone(report['terminal_bench_success'])
        checkpoint = report['candidate_checkpoint']
        self.assertEqual(checkpoint['actor']['adam_step'], 40)
        self.assertFalse(checkpoint['promoted'])
        self.assertEqual(len(checkpoint['actor']['adam_m']), len(FAMILIES)*21)

    def test_no_fa_dependencies_in_standalone_modules(self):
        root = Path(__file__).resolve().parents[1]
        source = '\n'.join(p.read_text() for p in (root / '__init__.py',
                       root / 'core.py', root / 'standalone.py', root / 'run.py',
                       root / 'benchmark.py'))
        for forbidden in ('import poincare', 'from poincare',
                          'import fa1', 'from fa1', 'import fa2', 'from fa2',
                          'MirrorLM', 'R38'):
            self.assertNotIn(forbidden, source)

    def test_multiseed_report_cannot_claim_benchmark_success(self):
        report = compare(range(2026, 2028), sft_updates=4, rpr_updates=2)
        self.assertEqual(report['seeds'], 2)
        self.assertEqual(report['heldout_task_count_per_seed'], 200)
        self.assertFalse(report['benchmark_eligible'])
        self.assertFalse(report['candidate_promoted'])


if __name__ == '__main__':
    unittest.main()
