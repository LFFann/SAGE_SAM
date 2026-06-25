import tempfile
import unittest

import torch

from sage_ssl.structure_cache import load_structure_cache, save_structure_cache


class StructureCacheTest(unittest.TestCase):
    def test_save_load_and_shape_mismatch(self):
        emb = torch.randn(1, 4, 3, 3)
        with tempfile.TemporaryDirectory() as tmp:
            save_structure_cache(tmp, "a", emb, "imagehash", "ckpthash")
            loaded = load_structure_cache(tmp, "a", expected_shape=tuple(emb.shape))
            self.assertTrue(torch.isfinite(loaded).all())
            with self.assertRaises(ValueError):
                load_structure_cache(tmp, "a", expected_shape=(1, 4, 4, 4))


if __name__ == "__main__":
    unittest.main()
