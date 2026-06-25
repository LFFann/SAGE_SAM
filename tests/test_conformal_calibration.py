import unittest

import torch

from sage_ssl.conformal_calibration import classwise_conformal_q, reliability_from_q


class ConformalCalibrationTest(unittest.TestCase):
    def test_classwise_q_and_reliability(self):
        prob = torch.full((1, 3, 3, 3), 0.1)
        label = torch.tensor([[[0, 1, 2], [0, 1, 2], [0, 1, 2]]])
        prob.scatter_(1, label[:, None], 0.8)
        q = classwise_conformal_q(prob, label, 3, alpha=0.1)
        self.assertEqual(tuple(q.shape), (3,))
        rel = reliability_from_q(prob, q)
        self.assertTrue(((rel >= 0) & (rel <= 1)).all())
        with self.assertRaises(ValueError):
            classwise_conformal_q(prob, torch.zeros_like(label), 3)


if __name__ == "__main__":
    unittest.main()
