"""Tests do not open the independent TEST corpus."""
import unittest
from unittest.mock import patch

import numpy as np

from m3_r23.apex import M3R23Predictor
from m3_r24.locked_test import load_locked_candidate, score_locked_test


class FrozenSpecialistTests(unittest.TestCase):
    def setUp(self):
        self.prior = np.ones(256,dtype=np.float64)/256
        self.theta,self.lock = load_locked_candidate()

    def test_only_the_selected_specialist_is_active(self):
        self.assertEqual(self.lock["selection"],"copy_only")
        self.assertEqual(self.lock["locked_test_indices"],[220,244])
        self.assertGreater(self.theta[0],0)
        self.assertTrue(np.array_equal(self.theta[1:],np.zeros(4)))

    def test_no_match_parity_with_inherited_default(self):
        model = M3R23Predictor(self.prior,theta=self.theta)
        for byte in b"hello":
            model.observe(byte)
            base,features = model.base_and_features()
            actual = model.readout(base,features,self.theta)
            self.assertTrue(np.array_equal(actual,base))

    def test_locked_copy_matches_r23_train_dev_path(self):
        model=M3R23Predictor(self.prior,theta=self.theta)
        for byte in b"long prefix long prefix long prefix":
            model.observe(byte)
            base,full_features=model.base_and_features()
            expected=model.readout(base,full_features,self.theta)
            fast_features=np.vstack((model.suffix.residual(base),np.zeros((4,256))))
            observed=model.readout(base,fast_features,self.theta)
            self.assertTrue(np.array_equal(observed,expected))

    def test_wrong_successor_value_is_appropriately_negative(self):
        model=M3R23Predictor(self.prior,theta=self.theta)
        for byte in b"hello world hello world hello world":
            model.observe(byte)
        base=model.predict_logp("r20")
        right=model.suffix.residual(base)
        wrong=model.suffix.residual(base,wrong_values=True)
        self.assertFalse(np.array_equal(right,wrong))

    def test_requires_exactly_twenty_four_books(self):
        with self.assertRaisesRegex(ValueError,"24 predetermined"):
            score_locked_test([b"book"],["book"],self.prior,self.theta)

    def test_small_synthetic_sequential_score(self):
        books=[b"alpha beta gamma alpha beta gamma alpha beta gamma"*2 for _ in range(24)]
        with patch("m3_r24.locked_test.scoring_start",return_value=5), \
             patch("m3_r24.locked_test.WARMUP",5):
            got=score_locked_test(books,[f"synthetic-{i}" for i in range(24)],
                                  self.prior,self.theta)
        self.assertEqual(got["test_books"],24)
        self.assertEqual(got["test_positions"],sum(len(b)-5 for b in books))
        self.assertGreater(got["copy_hit_rate"],0)
        self.assertEqual(len(got["per_book"]),24)


if __name__=="__main__":
    unittest.main()
