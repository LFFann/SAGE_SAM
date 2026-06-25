import tempfile
import unittest
from pathlib import Path

from dataloader.calibration_split import compute_dataset_fingerprint, load_or_create_calibration_manifest
from sage_test_utils import create_synthetic_real_dataset


class CalibrationSplitManifestTest(unittest.TestCase):
    def test_deterministic_complete_and_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = create_synthetic_real_dataset(root)
            fp = compute_dataset_fingerprint(dataset_root)
            path = root / "manifest.json"
            a = load_or_create_calibration_manifest(dataset_root / "labeled" / "image", dataset_root / "labeled" / "mask", path, 0.34, 123, 3, None, fp)
            b = load_or_create_calibration_manifest(dataset_root / "labeled" / "image", dataset_root / "labeled" / "mask", path, 0.34, 999, 3, None, fp)
            self.assertEqual(a, b)
            self.assertFalse(set(a["supervised_ids"]) & set(a["calibration_ids"]))
            self.assertEqual(len(set(a["supervised_ids"]) | set(a["calibration_ids"])), 6)
            self.assertFalse(any(sample_id.startswith("val/") or sample_id.startswith("test/") for sample_id in a["calibration_ids"]))

    def test_fingerprint_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = create_synthetic_real_dataset(root)
            path = root / "manifest.json"
            fp = compute_dataset_fingerprint(dataset_root)
            load_or_create_calibration_manifest(dataset_root / "labeled" / "image", dataset_root / "labeled" / "mask", path, 0.34, 123, 3, None, fp)
            with self.assertRaises(ValueError):
                load_or_create_calibration_manifest(dataset_root / "labeled" / "image", dataset_root / "labeled" / "mask", path, 0.34, 123, 3, None, "bad")


if __name__ == "__main__":
    unittest.main()
