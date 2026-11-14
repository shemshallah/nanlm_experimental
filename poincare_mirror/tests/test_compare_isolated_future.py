"""Matched forecast gate cannot promote a NOW or +2/+3 regression."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_isolated_future import compare


def report(step=2865):
    return {'checkpoint_step':step,'test_sha256':'synthetic-doc-hash',
            'target_positions_sha256':'fixed-position-hash',
            'test_positions':2048,'context_bytes':256,'seed':2026,
            'results':{h:{'ce_nats_per_byte':ce,'bits_per_byte':ce/0.6931471805599453,
                           'top1':acc,'top5':top5}
                       for h,ce,acc,top5 in [('+1',2.72,.32,.65),
                                             ('+2',3.19,.21,.50),
                                             ('+3',3.37,.18,.47)]}}


class ForecastComparisonTests(unittest.TestCase):
    def test_accepts_future_gains_with_exact_now(self):
        before=report()
        after=report(2900)
        for key in ('+2','+3'):
            after['results'][key]['ce_nats_per_byte']-=.02
            after['results'][key]['top1']+=.01
            after['results'][key]['top5']+=.01
        self.assertTrue(compare(before,after)['eligible'])

    def test_rejects_any_now_top1_regression(self):
        before=report();after=copy.deepcopy(report(2900))
        after['results']['+1']['top1']-=.001
        for key in ('+2','+3'):
            after['results'][key]['ce_nats_per_byte']-=.02
        self.assertFalse(compare(before,after)['eligible'])

    def test_rejects_future_accuracy_regression(self):
        before=report();after=copy.deepcopy(report(2900))
        for key in ('+2','+3'):
            after['results'][key]['ce_nats_per_byte']-=.02
        after['results']['+3']['top5']-=.001
        self.assertFalse(compare(before,after)['eligible'])

    def test_requires_same_positions(self):
        before=report();after=copy.deepcopy(report(2900))
        after['target_positions_sha256']='another-sample'
        with self.assertRaisesRegex(ValueError,'identity'):
            compare(before,after)


if __name__=='__main__':
    unittest.main()
