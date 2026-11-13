"""Gate independent data and aligned contexts without consulting test for selection."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_r31_independent import compare


class TestIndependentGate(unittest.TestCase):
    def dev(self):
        return {'promotable_on_development':True,'checkpoint_before':2801,'checkpoint_after':2865}

    def evaluation(self,step=2801,ce=2.4,ce2=3.0,ce3=3.2,accuracy=.3):
        return {'split':'test','data_split_sha256':'a'*64,'evaluation_context':256,
                'evaluated_tokens':4096,'evaluation_seed':2026,'exhaustive':True,'eval_stride':256,
                'checkpoint_step':step,'results':{'base':{'ce_1':ce,'ce_2':ce2,
                                                         'ce_3':ce3,'acc_1':accuracy}}}

    def context(self,step=2801,short=3.5,long=3.8):
        return {'step':step,'split':'valid','data_split_sha256':'b'*64,'seed':2026,
                'windows':4,'short':256,'long':4096,'same_target_positions':1024,
                'context_comparison':{'256':{'bpb':short},'4096':{'bpb':long}}}

    def inputs(self):
        return (self.dev(),self.evaluation(),self.evaluation(2865,2.38,2.99,3.198,.305),
                self.context(),self.context(2865,3.4,3.7))

    def test_eligible_but_frontier_not_claimed(self):
        output=compare(*self.inputs())
        self.assertTrue(output['eligible_for_manual_promotion'])
        self.assertFalse(output['frontier_context_advantage_demonstrated'])

    def test_test_regression_rejects(self):
        values=list(self.inputs())
        values[2]['results']['base']['ce_1']=2.5
        self.assertFalse(compare(*values)['eligible_for_manual_promotion'])

    def test_long_context_regression_rejects(self):
        values=list(self.inputs())
        values[4]['context_comparison']['4096']['bpb']=4.0
        self.assertFalse(compare(*values)['long_context_nonregression_pass'])

    def test_mismatched_split_checksum_rejected(self):
        values=list(self.inputs())
        values[2]['data_split_sha256']='c'*64
        with self.assertRaises(ValueError):compare(*values)

    def test_mismatched_window_seed_rejected(self):
        values=list(self.inputs())
        values[4]['seed']=5
        with self.assertRaises(ValueError):compare(*values)

    def test_cannot_open_test_without_development_selection(self):
        values=list(self.inputs())
        values[0]['promotable_on_development']=False
        with self.assertRaises(ValueError):compare(*values)

    def test_step_mismatch_rejected(self):
        values=list(self.inputs())
        values[2]['checkpoint_step']=2864
        with self.assertRaises(ValueError):compare(*values)


if __name__=='__main__':unittest.main()
