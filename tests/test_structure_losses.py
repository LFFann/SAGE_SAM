import unittest

import torch

from sage_ssl.structure_graph import build_local_structure_graph
from sage_ssl.structure_losses import boundary_agreement_penalty, same_edge_kl_loss


class StructureLossTest(unittest.TestCase):
    def test_same_loss_finite_and_identical_small(self):
        prob = torch.softmax(torch.randn(1, 3, 3, 3), dim=1)
        graph = build_local_structure_graph(torch.ones(1, 4, 3, 3), tau_same=0.5)
        loss = same_edge_kl_loss(prob, graph)
        self.assertTrue(torch.isfinite(loss))
        penalty = boundary_agreement_penalty(prob, graph)
        self.assertTrue(torch.isfinite(penalty))


if __name__ == "__main__":
    unittest.main()
