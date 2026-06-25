import unittest

import torch

from sage_ssl.structure_graph import build_local_structure_graph


class StructureGraphTest(unittest.TestCase):
    def test_affinity_masks_are_local_and_exclusive(self):
        emb = torch.ones(1, 4, 4, 4)
        graph = build_local_structure_graph(emb, tau_same=0.8, tau_boundary=0.1)
        self.assertEqual(tuple(graph.affinity.shape), (1, 2, 4, 4))
        self.assertTrue((graph.affinity <= 1).all())
        self.assertTrue(graph.same[graph.valid].all())
        self.assertFalse(bool((graph.same & graph.boundary).any()))


if __name__ == "__main__":
    unittest.main()
