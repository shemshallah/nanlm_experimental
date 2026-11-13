"""Development acceptance cannot silently promote mismatched evaluations."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_r31 import compare


class TestCompare(unittest.TestCase):
    def evaluation(self, ce1=2.1, ce2=2.8, ce3=3.0, acc=0.40):
        return {'split':'valid','evaluation_context':256,'evaluated_tokens':4096,
                'checkpoint_step':2801,'results':{'base':{
                    'ce_1':ce1,'ce_2':ce2,'ce_3':ce3,'acc_1':acc}}}

    def test_improved(self):
        self.assertTrue(compare(self.evaluation(),self.evaluation(2.0,2.7,3.0,.41),.0001)
                        ['promotable_on_development'])

    def test_regressions_and_mismatches(self):
        self.assertFalse(compare(self.evaluation(),self.evaluation(2.0,2.9,3.0,.41),.0001)
                         ['promotable_on_development'])
        test=self.evaluation(2.0,2.7,3.0,.41)
        test['split']='test'
        with self.assertRaises(ValueError):
            compare(self.evaluation(),test,.0001)


if __name__=='__main__':
    unittest.main()
