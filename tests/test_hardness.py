import unittest

import torch

from sage_ssl.hardness import estimate_hardness


class HardnessTest(unittest.TestCase):
    def test_disagreement_increases_hardness(self):
        a = torch.zeros(1, 3, 2, 2)
        a[:, 0] = 1
        b_same = a.clone()
        b_diff = torch.zeros_like(a)
        b_diff[:, 1] = 1
        self.assertLess(float(estimate_hardness(a, b_same)), float(estimate_hardness(a, b_diff)))


if __name__ == "__main__":
    unittest.main()
