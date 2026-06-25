import unittest

import torch

from sage_ssl.graph_propagation import propagate_on_same_edges
from sage_ssl.structure_graph import build_local_structure_graph


class GraphPropagationTest(unittest.TestCase):
    def test_propagates_and_respects_union(self):
        prob = torch.softmax(torch.randn(1, 3, 3, 3), dim=1)
        graph = build_local_structure_graph(torch.ones(1, 4, 3, 3), tau_same=0.5)
        union = torch.ones_like(prob).bool()
        union[:, 2] = False
        out = propagate_on_same_edges(prob, graph, union)
        self.assertTrue(torch.allclose(out.sum(1), torch.ones(1, 3, 3), atol=1e-5))
        self.assertEqual(float(out[:, 2].sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
