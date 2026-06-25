"""SAGE-SAM training entrypoint."""

from __future__ import annotations

import argparse

import torch

from Model.sage_model import DualSegmentor
from engine.sage_trainer import SAGETrainer
from utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    seed_everything()
    model = DualSegmentor(in_channels=args.in_channels, num_classes=args.num_classes).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = SAGETrainer(model, optimizer, num_classes=args.num_classes)
    if args.smoke:
        image = torch.rand(2, args.in_channels, 64, 64, device=args.device)
        label = torch.randint(0, args.num_classes, (2, 64, 64), device=args.device)
        output = trainer.step(labeled=(image, label))
        print({"total_loss": float(output.total), "optimizer_sources": trainer.optimizer_parameter_sources()})
        return
    raise SystemExit("Dataset training is configured via scripts/train_v100.sh or custom loader integration.")


if __name__ == "__main__":
    main()
