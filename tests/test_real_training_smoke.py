import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sage_test_utils import create_synthetic_real_dataset, create_synthetic_structure_cache


class RealTrainingSmokeTest(unittest.TestCase):
    def test_minimal_real_training_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = create_synthetic_real_dataset(root)
            cache = create_synthetic_structure_cache(dataset_root, root / "cache", grid_size=32)
            output = root / "out"
            cmd = [
                sys.executable,
                "train_sage_sam.py",
                "--data_path",
                str(root),
                "--dataset",
                dataset_root.name,
                "--structure_cache",
                str(cache),
                "--output_dir",
                str(output),
                "--experiment_name",
                "mini",
                "--device",
                "cpu",
                "--num_classes",
                "3",
                "--image_size",
                "32",
                "--labeled_batch_size",
                "2",
                "--unlabeled_batch_size",
                "1",
                "--calibration_batch_size",
                "1",
                "--max_iterations",
                "3",
                "--warmup_iterations",
                "1",
                "--validation_interval",
                "1",
                "--checkpoint_interval",
                "1",
                "--no_amp",
            ]
            subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True)
            self.assertTrue((output / "mini" / "checkpoints" / "latest.pth").exists())
            self.assertTrue((output / "mini" / "checkpoints" / "best_fused_dice.pth").exists())
            self.assertTrue((output / "mini" / "checkpoints" / "final.pth").exists())
            resume = cmd[:]
            resume[resume.index("--max_iterations") + 1] = "5"
            resume.extend(["--resume", str(output / "mini" / "checkpoints" / "latest.pth")])
            subprocess.run(resume, cwd=Path(__file__).resolve().parents[1], check=True)


if __name__ == "__main__":
    unittest.main()
