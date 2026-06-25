import tempfile
import unittest
from pathlib import Path

import torch

from sage_ssl.structure_cache import StructureCacheReader
from sage_test_utils import create_synthetic_real_dataset, create_synthetic_structure_cache, sample_ids_for_cache


class StructureCacheTrainingTest(unittest.TestCase):
    def test_reader_validation_get_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = create_synthetic_real_dataset(root)
            cache = create_synthetic_structure_cache(dataset_root, root / "cache", grid_size=16)
            ids = set(sample_ids_for_cache(dataset_root))
            reader = StructureCacheReader(cache, dataset_root.name, 16, ids)
            info = reader.validate()
            self.assertEqual(info["required_count"], len(ids))
            emb = reader.get([next(iter(ids))])
            self.assertFalse(emb.requires_grad)
            self.assertEqual(tuple(emb.shape[-2:]), (16, 16))
            with self.assertRaises(FileNotFoundError):
                StructureCacheReader(cache, dataset_root.name, 16, ids | {"missing/image.png"}).validate()
            with self.assertRaises(ValueError):
                StructureCacheReader(cache, dataset_root.name, 32, ids).validate()


if __name__ == "__main__":
    unittest.main()
