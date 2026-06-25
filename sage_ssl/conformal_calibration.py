"""Class-conditional conformal calibration for each branch."""

from __future__ import annotations

import math

import torch


def classwise_conformal_q(prob: torch.Tensor, label: torch.Tensor, num_classes: int, alpha: float = 0.1) -> torch.Tensor:
    if prob.ndim != 4:
        raise ValueError("prob must be [B,C,H,W]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    q = torch.empty(num_classes, dtype=prob.dtype, device=prob.device)
    for cls in range(num_classes):
        cls_mask = label == cls
        if not bool(cls_mask.any()):
            raise ValueError(f"Missing class {cls} in calibration labels")
        scores = 1.0 - prob[:, cls][cls_mask].detach()
        scores = scores.sort().values
        rank = min(scores.numel(), max(1, math.ceil((scores.numel() + 1) * (1.0 - alpha)))) - 1
        q[cls] = scores[rank].clamp(1e-6, 1.0)
    return q


def reliability_from_q(prob: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    threshold = (1.0 - q).view(1, -1, 1, 1).to(prob.device)
    return ((prob - threshold) / q.view(1, -1, 1, 1).to(prob.device).clamp_min(1e-6)).clamp(0.0, 1.0)
