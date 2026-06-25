import tempfile
import unittest
from pathlib import Path

from dataloader.builders import build_sage_dataloaders, resolve_dataset_root
from dataloader.calibration_split import compute_dataset_fingerprint, load_or_create_calibration_manifest
from sage_test_utils import create_synthetic_real_dataset


class RealDataPipelineTest(unittest.TestCase):
    def test_loader_construction_and_batch_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = create_synthetic_real_dataset(root)
            self.assertEqual(resolve_dataset_root(root, "/synthetic_3class"), dataset_root.resolve())
            manifest = load_or_create_calibration_manifest(
                dataset_root / "labeled" / "image",
                dataset_root / "labeled" / "mask",
                root / "manifest.json",
                0.34,
                7,
                3,
                None,
                compute_dataset_fingerprint(dataset_root),
            )
            loaders = build_sage_dataloaders(dataset_root, manifest, 3, 3, 32, 2, 1, 1, 0, 7)
            self.assertEqual(next(iter(loaders["supervised_loader"]))["image"].shape[0], 2)
            self.assertEqual(next(iter(loaders["unlabeled_loader"]))["image"].shape[0], 1)
            self.assertEqual(next(iter(loaders["calibration_loader"]))["mask"].shape[-2:], (32, 32))
            self.assertEqual(next(iter(loaders["val_loader"]))["image"].shape[1], 3)


if __name__ == "__main__":
    unittest.main()
