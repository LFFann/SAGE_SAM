"""Tensor helpers for SAGE-SAM."""

import torch


def finite_or_zero(value: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(value), value, torch.zeros_like(value))
