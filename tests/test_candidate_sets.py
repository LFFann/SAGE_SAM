import unittest

import torch

from sage_ssl.candidate_sets import build_candidate_sets


class CandidateSetsTest(unittest.TestCase):
    def test_union_core_negative_and_unknown(self):
        prob_a = torch.tensor([[[[0.8]], [[0.1]], [[0.1]]]])
        prob_b = torch.tensor([[[[0.1]], [[0.8]], [[0.1]]]])
        sets = build_candidate_sets(prob_a, prob_b, torch.tensor([0.3, 0.3, 0.3]), torch.tensor([0.3, 0.3, 0.3]))
        self.assertTrue(bool(sets.union[:, 0].item()))
        self.assertTrue(bool(sets.union[:, 1].item()))
        self.assertFalse(bool(sets.union[:, 2].item()))
        self.assertTrue(bool(sets.negative[:, 2].item()))
        self.assertTrue(bool(sets.unknown.item()))


if __name__ == "__main__":
    unittest.main()
