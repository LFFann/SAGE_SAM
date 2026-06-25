import tempfile
import unittest

import torch

import train_sage_sam


class TrainEntrypointTest(unittest.TestCase):
    def test_data_path_aliases_are_equivalent(self):
        parser = train_sage_sam.build_parser()
        a = parser.parse_args(["--data_path", "A", "--dataset", "D"])
        b = parser.parse_args(["--data-path", "A", "--dataset", "D"])
        self.assertEqual(a.data_path, b.data_path)

    def test_smoke_cpu_runs(self):
        args = train_sage_sam.build_parser().parse_args(["--smoke", "--device", "cpu", "--num_classes", "3"])
        train_sage_sam.run_smoke(args)

    def test_dataset_missing_errors_outside_smoke(self):
        args = train_sage_sam.build_parser().parse_args(["--data_path", "x"])
        with self.assertRaises(ValueError):
            train_sage_sam.prepare_run(args)

    def test_resume_and_init_checkpoint_conflict(self):
        args = train_sage_sam.build_parser().parse_args(["--data_path", "x", "--dataset", "d", "--resume", "a", "--init_checkpoint", "b"])
        with self.assertRaises(ValueError):
            train_sage_sam.prepare_run(args)

    def test_warmup_greater_than_max_errors(self):
        args = train_sage_sam.build_parser().parse_args(["--data_path", "x", "--dataset", "d", "--warmup_iterations", "5", "--max_iterations", "3"])
        with self.assertRaises(ValueError):
            train_sage_sam.prepare_run(args)


if __name__ == "__main__":
    unittest.main()
