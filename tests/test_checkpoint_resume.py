import tempfile
import unittest
from pathlib import Path

import torch

from Model.sage_model import DualSegmentor
from engine.checkpoint import CheckpointManager
from engine.sage_trainer import SAGETrainer


class CheckpointResumeTest(unittest.TestCase):
    def test_manager_round_trip_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = DualSegmentor(in_channels=3, num_classes=3)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            trainer = SAGETrainer(model, opt, 3)
            batch = {"image": torch.rand(1, 3, 32, 32), "mask": torch.randint(0, 3, (1, 32, 32))}
            trainer.train_step(batch, None, None, 0, torch.device("cpu"))
            manager = CheckpointManager(Path(tmp))
            payload = manager.build_payload(
                iteration=1,
                best_metric=0.2,
                model=model,
                optimizer=opt,
                dataset_root="dataset",
                dataset_fingerprint="fp",
                calibration_manifest={"calibration_ids": ["a"]},
                semantic_calibration_state={"q_A": [0.5] * 3, "q_B": [0.5] * 3},
                structure_calibration_state={"tau_same": 0.8, "tau_boundary": 0.2},
                class_weights=torch.ones(3),
            )
            path = manager.save_latest(payload)
            new_model = DualSegmentor(in_channels=3, num_classes=3)
            new_opt = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
            loaded = manager.load(path, new_model, optimizer=new_opt)
            self.assertEqual(loaded["iteration"], 1)
            self.assertEqual(loaded["calibration_manifest"]["calibration_ids"], ["a"])
            self.assertEqual(loaded["semantic_calibration_state"]["q_A"], [0.5] * 3)
            before = [p.detach().clone() for p in new_model.parameters()]
            SAGETrainer(new_model, new_opt, 3).train_step(batch, None, None, 2, torch.device("cpu"))
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, new_model.parameters())))


if __name__ == "__main__":
    unittest.main()
