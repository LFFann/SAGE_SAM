import tempfile
import unittest
from pathlib import Path

import torch

from sage_ssl.structure_calibration import StructureCalibration, calibrate_structure_thresholds


class StructureCalibrationTest(unittest.TestCase):
    def test_thresholds_are_ordered_and_serializable(self):
        emb = torch.randn(2, 4, 6, 6)
        mask = torch.zeros(2, 6, 6).long()
        mask[:, :, 3:] = 1
        cal = calibrate_structure_thresholds(emb, mask, target_precision=0.8)
        self.assertLess(cal.tau_boundary, cal.tau_same)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cal.json"
            cal.to_json(path)
            loaded = StructureCalibration.from_json(path)
            self.assertEqual(loaded.target_precision, cal.target_precision)


if __name__ == "__main__":
    unittest.main()
