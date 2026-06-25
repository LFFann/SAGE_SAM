"""Hardness-controlled, structure-safe perturbations."""

from __future__ import annotations

import torch


def adaptive_intensity_augmentation(image: torch.Tensor, hardness: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    strength = (0.35 * (1.0 - hardness).view(-1, 1, 1, 1) + 0.03).to(image.device)
    noise = torch.randn(image.shape, device=image.device, dtype=image.dtype, generator=generator) * strength
    return (image + noise).clamp(0.0, 1.0)


def structure_safe_mask(image: torch.Tensor, safe_mask: torch.Tensor, hardness: torch.Tensor) -> torch.Tensor:
    out = image.clone()
    strength = (0.5 * (1.0 - hardness).view(-1, 1, 1, 1)).to(image.device)
    mask = safe_mask.to(device=image.device, dtype=image.dtype)
    out = out * (1.0 - mask * strength)
    return out.clamp(0.0, 1.0)
