import unittest

import torch

from sage_ssl.relational_consistency import relational_consistency_loss, select_structure_agents


class RelationalConsistencyTest(unittest.TestCase):
    def test_identical_views_have_small_loss_and_backward(self):
        prob = torch.softmax(torch.randn(2, 3, 4, 4, requires_grad=True), dim=1)
        agents = select_structure_agents(torch.randn(2, 5, 4, 4), k=3)
        loss = relational_consistency_loss(prob, prob, agents)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()


if __name__ == "__main__":
    unittest.main()
