import unittest

import torch

from sage_ssl.adaptive_augmentation import adaptive_intensity_augmentation, structure_safe_mask


class AdaptiveAugmentationTest(unittest.TestCase):
    def test_shapes_ranges_and_safe_mask(self):
        image = torch.full((2, 1, 8, 8), 0.5)
        aug = adaptive_intensity_augmentation(image, torch.tensor([0.0, 1.0]))
        self.assertEqual(tuple(aug.shape), tuple(image.shape))
        self.assertTrue(((aug >= 0) & (aug <= 1)).all())
        masked = structure_safe_mask(image, torch.ones(2, 1, 8, 8), torch.tensor([0.0, 1.0]))
        self.assertLess(float(masked[0].mean()), float(masked[1].mean()))


if __name__ == "__main__":
    unittest.main()
