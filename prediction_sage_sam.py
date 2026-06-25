"""SAGE-SAM inference entrypoint.

Inference is intentionally UNet + VNet only. It does not import SAM builders,
read structure caches, or instantiate the copied KnowSAM HAM/Discriminator.
"""

from __future__ import annotations

import argparse
import json

import torch

from Model.sage_model import DualSegmentor
from engine.evaluator import inference_summary, predict_logits


def build_model(num_classes: int = 3, in_channels: int = 3, checkpoint: str | None = None, device: str = "cpu") -> DualSegmentor:
    model = DualSegmentor(in_channels=in_channels, num_classes=num_classes).to(device)
    if checkpoint:
        payload = torch.load(checkpoint, map_location=device)
        state = payload.get("model", payload)
        model.load_state_dict(state, strict=False)
    return model


def predict_tensor(image: torch.Tensor, checkpoint: str | None = None, num_classes: int = 3, device: str = "cpu") -> tuple[torch.Tensor, dict]:
    model = build_model(num_classes=num_classes, in_channels=image.shape[1], checkpoint=checkpoint, device=device)
    logits = predict_logits(model, image.to(device))
    return logits.argmax(dim=1).cpu(), inference_summary()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    image = torch.zeros(1, 3, 64, 64)
    pred, summary = predict_tensor(image, checkpoint=args.checkpoint, num_classes=args.num_classes, device=args.device)
    print(json.dumps({**summary, "prediction_shape": list(pred.shape)}, sort_keys=True))


if __name__ == "__main__":
    main()
