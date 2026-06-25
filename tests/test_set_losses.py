import unittest

import torch

from sage_ssl.set_losses import negative_set_loss, partial_label_loss, singleton_or_set_loss


class SetLossTest(unittest.TestCase):
    def test_singleton_matches_ce_and_full_set_zero(self):
        logits = torch.tensor([[[[4.0]], [[1.0]], [[0.0]]]])
        singleton = torch.tensor([[[[True]], [[False]], [[False]]]])
        ce_like = singleton_or_set_loss(logits, singleton)
        self.assertLess(float(ce_like), 0.1)
        full = torch.ones_like(singleton).bool()
        self.assertEqual(float(partial_label_loss(torch.softmax(logits, 1), full)), 0.0)

    def test_negative_loss_penalizes_outside_probability(self):
        prob = torch.tensor([[[[0.1]], [[0.1]], [[0.8]]]])
        neg = torch.tensor([[[[False]], [[False]], [[True]]]])
        self.assertGreater(float(negative_set_loss(prob, neg)), 1.0)


if __name__ == "__main__":
    unittest.main()
