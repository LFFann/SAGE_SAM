import tempfile
import unittest
from pathlib import Path

import torch

from Model.sage_model import DualSegmentor
from engine.checkpoint import convert_knowsam_state_dict, load_checkpoint, save_checkpoint


class CheckpointTest(unittest.TestCase):
    def test_save_restore_and_convert_skips_discriminator(self):
        model = DualSegmentor(in_channels=3, num_classes=3)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            save_checkpoint(path, model, opt, step=2, calibration={"q": [0.5]}, split_manifest={"train": ["a"]})
            payload = load_checkpoint(path, model, opt, strict=False)
            self.assertEqual(payload["step"], 2)
        converted = convert_knowsam_state_dict({"UNet.x": torch.tensor(1), "Discriminator.x": torch.tensor(2)})
        self.assertIn("UNet.x", converted["state_dict"])
        self.assertIn("Discriminator.x", converted["skipped"])


if __name__ == "__main__":
    unittest.main()
