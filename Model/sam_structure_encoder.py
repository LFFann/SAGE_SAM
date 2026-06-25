"""SAM image-encoder wrapper for structure precomputation only."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class SAMStructureEncoder(nn.Module):
    """Frozen image-encoder-only SAM wrapper.

    The prompt encoder and mask decoder are never called. When no checkpoint is
    supplied, the encoder can also return a deterministic synthetic embedding
    for CPU smoke tests.
    """

    def __init__(
        self,
        checkpoint: str | None = None,
        model_type: str = "vit_b",
        device: str | torch.device = "cpu",
        image_size: int = 256,
        in_channels: int = 3,
        num_classes: int = 3,
        point_nums: int = 5,
        box_nums: int = 1,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.model_type = model_type
        self.device = torch.device(device)
        self.sam_args = SimpleNamespace(
            image_size=int(image_size),
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            point_nums=int(point_nums),
            box_nums=int(box_nums),
        )
        self.sam = None
        self.image_encoder = None
        if checkpoint:
            self._load_sam_image_encoder(checkpoint)

    def _load_sam_image_encoder(self, checkpoint: str) -> None:
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
        from Model.sam import sam_model_registry

        sam = sam_model_registry[self.model_type](self.sam_args, checkpoint=str(path))
        sam.eval().to(self.device)
        for parameter in sam.parameters():
            parameter.requires_grad_(False)
        self.sam = sam
        self.image_encoder = sam.image_encoder

    @torch.no_grad()
    def forward(self, image: torch.Tensor, output_size: tuple[int, int] = (16, 16)) -> torch.Tensor:
        image = image.to(self.device)
        if self.image_encoder is None:
            pooled = F.interpolate(image.float(), size=output_size, mode="bilinear", align_corners=False)
            channels = [pooled]
            channels.append(torch.sin(pooled[:, :1] * 3.14159265))
            channels.append(torch.cos(pooled[:, 1:2] * 3.14159265 if pooled.shape[1] > 1 else pooled[:, :1]))
            return torch.cat(channels, dim=1)
        embedding = self.image_encoder(image)
        return F.interpolate(embedding.float(), size=output_size, mode="bilinear", align_corners=False)
