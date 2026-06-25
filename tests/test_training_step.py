import unittest

import torch

from Model.sage_model import DualSegmentor
from engine.sage_trainer import SAGETrainer


class TrainingStepTest(unittest.TestCase):
    def test_supervised_and_ssl_steps_update_only_unet_vnet(self):
        model = DualSegmentor(in_channels=3, num_classes=3)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = SAGETrainer(model, opt, num_classes=3)
        labeled = (torch.rand(1, 3, 32, 32), torch.randint(0, 3, (1, 32, 32)))
        out = trainer.step(labeled=labeled)
        self.assertTrue(torch.isfinite(out.total))
        structure = torch.randn(1, 4, 8, 8)
        out = trainer.step(unlabeled=(labeled[0], structure), q=torch.tensor([0.5, 0.5, 0.5]))
        self.assertTrue(torch.isfinite(out.total))
        sources = trainer.optimizer_parameter_sources()
        self.assertGreater(sources["UNet"], 0)
        self.assertGreater(sources["VNet"], 0)


if __name__ == "__main__":
    unittest.main()
