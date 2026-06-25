import unittest

import torch

from Model.sage_model import DualSegmentor


class DualSegmentorTest(unittest.TestCase):
    def test_shapes_branches_features_and_backward(self):
        model = DualSegmentor(in_channels=3, num_classes=3)
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        self.assertEqual(tuple(out["logits_a"].shape), (1, 3, 32, 32))
        self.assertEqual(tuple(out["logits_b"].shape), (1, 3, 32, 32))
        self.assertIsNotNone(out["feature_a"])
        self.assertIsNotNone(out["decoder_feature_b"])
        loss = out["logits_a"].mean() + out["logits_b"].mean()
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.UNet.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.VNet.parameters()))

    def test_a_only_b_only_and_no_forbidden_modules(self):
        model = DualSegmentor(in_channels=3, num_classes=3)
        x = torch.randn(1, 3, 32, 32)
        self.assertIsNone(model(x, branches=("A",))["logits_b"])
        self.assertIsNone(model(x, branches=("B",))["logits_a"])
        self.assertFalse(hasattr(model, "Discriminator"))


if __name__ == "__main__":
    unittest.main()
