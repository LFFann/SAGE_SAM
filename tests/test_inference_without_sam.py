import unittest
from unittest import mock

import torch

import prediction_sage_sam
from engine.evaluator import inference_summary


class InferenceWithoutSAMTest(unittest.TestCase):
    def test_prediction_does_not_need_sam(self):
        with mock.patch.dict("sys.modules", {"Model.sam": None}):
            pred, summary = prediction_sage_sam.predict_tensor(torch.zeros(1, 3, 32, 32), num_classes=3)
        self.assertEqual(tuple(pred.shape), (1, 32, 32))
        self.assertFalse(summary["inference_uses_sam"])
        self.assertFalse(inference_summary()["inference_reads_precomputed_structure"])


if __name__ == "__main__":
    unittest.main()
