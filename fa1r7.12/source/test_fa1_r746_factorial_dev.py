"""R7.46 causal ablation safety and attribution controls."""
import unittest

import torch

from fa1_r746_factorial_dev import perturbed_context, same_points_new_order


class FactorialCausalControlTests(unittest.TestCase):
    def setUp(self):
        self.data={
            'book_ids':torch.tensor([0,0,1,1]),
            'target':torch.tensor([100,101,102,103]),
            'hidden':torch.arange(16).reshape(4,4),
            'diamond_backoff_recent_ids':torch.arange(16).reshape(4,4),
            'vertex_ids':torch.arange(32).reshape(4,8),
            'word_ids':torch.arange(128).reshape(4,32),
            'byte_ids':torch.arange(512).reshape(4,128),
            'sentence_ids':torch.arange(32).reshape(4,8),
        }

    def test_address_only_control_changes_only_vertices(self):
        new=perturbed_context(self.data,'addresses_only',746)
        self.assertFalse(torch.equal(new['vertex_ids'],self.data['vertex_ids']))
        for field in self.data.keys()-{'vertex_ids'}:
            self.assertTrue(torch.equal(new[field],self.data[field]),field)
        for row in range(4):
            self.assertNotEqual(int(self.data['book_ids'][row]),
                                int(self.data['book_ids'][new['vertex_ids'][row,0]//8]))

    def test_lexical_towers_never_swap_copy_candidates_or_labels(self):
        new=perturbed_context(self.data,'lexical_towers_only',747)
        for field in ('target','book_ids','hidden','vertex_ids',
                      'diamond_backoff_recent_ids'):
            self.assertTrue(torch.equal(new[field],self.data[field]),field)
        for field in ('word_ids','byte_ids','sentence_ids'):
            self.assertFalse(torch.equal(new[field],self.data[field]),field)

    def test_reorder_keeps_each_active_point_set(self):
        new=same_points_new_order(self.data)
        self.assertFalse(torch.equal(new['vertex_ids'],self.data['vertex_ids']))
        self.assertTrue(torch.equal(new['vertex_ids'].sort(1).values,
                                    self.data['vertex_ids'].sort(1).values))
        for field in self.data.keys()-{'vertex_ids'}:
            self.assertTrue(torch.equal(new[field],self.data[field]),field)

    def test_unsupported_control_is_rejected(self):
        with self.assertRaises(ValueError):
            perturbed_context(self.data,'swap_gold_targets',748)


if __name__=='__main__':
    unittest.main()
