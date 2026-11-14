"""Book-unit sampling and paired bootstrap invariants for FA2 promotion gates."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.compare_geometry_books import book_starts, bootstrap_gain


class WholeBookComparisonTest(unittest.TestCase):
    def test_windows_remain_inside_independent_books(self):
        lengths=[80,100,75]
        starts,total=book_starts(lengths,context=32,windows=20,seed=9)
        self.assertEqual(total,sum(lengths))
        offset=0
        for positions,length in zip(starts,lengths):
            self.assertTrue(all(offset <= p and p+35 <= offset+length
                                for p in positions))
            offset+=length
        repeated,_=book_starts(lengths,32,20,9)
        self.assertTrue(all((a==b).all() for a,b in zip(starts,repeated)))

    def test_bootstrap_resamples_books_and_positive_means_better(self):
        result=bootstrap_gain([2.0]*12,[1.9]*12,resamples=1000)
        self.assertAlmostEqual(result['mean_gain'],.1)
        self.assertGreater(result['ci95'][0],0)
        self.assertEqual(result['books_improved'],12)
        acc=bootstrap_gain([.25]*12,[.30]*12,resamples=1000,larger_better=True)
        self.assertAlmostEqual(acc['mean_gain'],.05)
        self.assertGreater(acc['ci95'][0],0)

    def test_single_book_is_insufficient(self):
        with self.assertRaisesRegex(ValueError,'at least two'):
            bootstrap_gain([2.0],[1.9])


if __name__=='__main__':
    unittest.main()
