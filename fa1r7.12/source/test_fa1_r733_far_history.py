"""FA1 R7.33 matched older-history causal negative and promotion tests."""
import copy
import unittest

import torch

from fa1_r733_far_history import HORIZONS, matched_far_history
from fa1_r728_geometric_memory import candidate_priority, passes_gate


class FarHistoryTests(unittest.TestCase):
    @staticmethod
    def data():
        batch=4
        def matrix(width):
            return torch.arange(batch*width,dtype=torch.long).reshape(batch,width)
        return {'target':torch.tensor([5,7,11,13]),
                'book_ids':torch.tensor([0,0,1,1]),
                'hidden':torch.randn(batch,96),
                'word_ids':matrix(32), 'byte_ids':matrix(128),
                'sentence_ids':matrix(8), 'vertex_ids':matrix(8),
                'diamond_memory_features':matrix(92).float(),
                'diamond_modal_ids':matrix(12),
                'diamond_modal_mask':matrix(12).remainder(2).bool(),
                'byte_target_sequence':matrix(5)}

    def test_foreign_old_history_preserves_recent_and_gold(self):
        data=self.data()
        snapshot=copy.deepcopy(data)
        swapped,donors=matched_far_history(data,733)
        self.assertTrue(torch.all(data['book_ids'][donors]!=data['book_ids']))
        self.assertTrue(torch.equal(swapped['target'],data['target']))
        self.assertTrue(torch.equal(swapped['hidden'],data['hidden']))
        self.assertTrue(torch.equal(swapped['vertex_ids'],data['vertex_ids']))
        self.assertTrue(torch.equal(swapped['byte_target_sequence'],data['byte_target_sequence']))
        for key,n in (('word_ids',8),('byte_ids',32),('sentence_ids',2)):
            self.assertTrue(torch.equal(swapped[key][:,-n:],data[key][:,-n:]))
            self.assertTrue(torch.equal(swapped[key][:,:-n],data[key][donors,:-n]))
        self.assertTrue(torch.equal(swapped['diamond_memory_features'][:,:60],
                                    data['diamond_memory_features'][:,:60]))
        self.assertTrue(torch.equal(swapped['diamond_memory_features'][:,60:],
                                    data['diamond_memory_features'][donors,60:]))
        for key in ('diamond_modal_ids','diamond_modal_mask'):
            self.assertTrue(torch.equal(swapped[key][:,:8],data[key][:,:8]))
            self.assertTrue(torch.equal(swapped[key][:,8:],data[key][donors,8:]))
        for key in data:
            self.assertTrue(torch.equal(data[key],snapshot[key]),f'mutated {key}')

    def test_matched_control_is_deterministic(self):
        data=self.data()
        a,ai=matched_far_history(data,112)
        b,bi=matched_far_history(data,112)
        self.assertTrue(torch.equal(ai,bi))
        for k in a:
            self.assertTrue(torch.equal(a[k],b[k]))

    def test_three_horizons_keep_progressively_more_local_evidence(self):
        data=self.data()
        self.assertEqual(HORIZONS,((8,32,2),(16,64,4),(24,96,6)))
        for horizon,(nw,nb,ns) in enumerate(HORIZONS):
            swapped,donors=matched_far_history(data,1700,horizon=horizon)
            for key,keep in (('word_ids',nw),('byte_ids',nb),('sentence_ids',ns)):
                self.assertTrue(torch.equal(swapped[key][:,-keep:],data[key][:,-keep:]))
                self.assertTrue(torch.equal(swapped[key][:,:-keep],data[key][donors,:-keep]))
            self.assertTrue(torch.equal(swapped['diamond_memory_features'][:,:60],
                                        data['diamond_memory_features'][:,:60]))
        with self.assertRaisesRegex(ValueError,'horizon'):
            matched_far_history(data,1700,horizon=3)

    def test_validates_different_books_and_temporal_shapes(self):
        data=self.data()
        data['book_ids'].zero_()
        with self.assertRaises(ValueError):matched_far_history(data)
        data=self.data()
        data['word_ids']=data['word_ids'][:,:8]
        with self.assertRaisesRegex(ValueError,'word_ids'):matched_far_history(data)
        data=self.data()
        data['diamond_memory_features']=data['diamond_memory_features'][:,:60]
        with self.assertRaisesRegex(ValueError,'60 near'):matched_far_history(data)

    @staticmethod
    def candidate():
        positive={'book_bootstrap_ci95':[.001,.002]}
        return dict(full_word_nll=5.2,geometry_word_gain=.01,
          branch_geometry_gain=.01,correct_book_advantage_nats_per_word=.005,
          apex_word_gain=.002,pair_word_gain=.002,
          book_geometry_gain=positive,book_correct_history=positive,
          conditional_byte_geometry_gain_nats_per_byte=.006,
          conditional_byte_apex_gain_nats_per_byte=.002,
          conditional_byte_pair_gain_nats_per_byte=.002,
          conditional_byte_geometry_gain_book_bootstrap=positive,
          conditional_spelling_bits_per_byte=3.9,
          conditional_spelling_top1=.4,conditional_spelling_top5=.7,
          matched_far_history_advantage_nats_per_word=.01,
          matched_far_geometry_specificity_nats_per_word=.005,
          matched_far_history_bootstrap=positive,
          matched_far_geometry_bootstrap=positive,
          matched_far_byte_history_advantage_nats_per_byte=.02,
          matched_far_byte_geometry_specificity_nats_per_byte=.01,
          matched_far_byte_history_bootstrap=positive,
          matched_far_byte_geometry_bootstrap=positive)

    def test_strict_gate_requires_far_history_specificity_and_book_ci(self):
        reference=self.candidate()
        reference['full_word_nll']=5.3
        reference['conditional_spelling_bits_per_byte']=4.2
        candidate=self.candidate()
        self.assertTrue(passes_gate(reference,candidate,.0001))
        wrong=copy.deepcopy(candidate)
        wrong['matched_far_geometry_specificity_nats_per_word']=-.001
        self.assertFalse(passes_gate(reference,wrong,.0001))
        self.assertLess(candidate_priority(reference,wrong,.0001),
                        candidate_priority(reference,candidate,.0001))
        wrong=copy.deepcopy(candidate)
        wrong['matched_far_byte_geometry_bootstrap']={'book_bootstrap_ci95':[-.001,.1]}
        self.assertFalse(passes_gate(reference,wrong,.0001))
        wrong=copy.deepcopy(candidate)
        wrong['matched_far_byte_history_advantage_nats_per_byte']=-.0001
        self.assertFalse(passes_gate(reference,wrong,.0001))


if __name__=='__main__':
    unittest.main()
