"""Target-domain calibration for class-agnostic SAM structure edges."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass
class StructureCalibration:
    tau_same: float
    tau_boundary: float
    target_precision: float
    num_same_edges: int
    num_boundary_edges: int

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "StructureCalibration":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def _edge_samples(embedding: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    emb = F.normalize(embedding.float(), dim=1)
    if mask.ndim == 4:
        mask = mask.squeeze(1)
    mask = F.interpolate(mask[:, None].float(), size=emb.shape[-2:], mode="nearest").squeeze(1).long()
    aff_right = (emb[:, :, :, :-1] * emb[:, :, :, 1:]).sum(dim=1).reshape(-1)
    same_right = (mask[:, :, :-1] == mask[:, :, 1:]).reshape(-1)
    aff_down = (emb[:, :, :-1, :] * emb[:, :, 1:, :]).sum(dim=1).reshape(-1)
    same_down = (mask[:, :-1, :] == mask[:, 1:, :]).reshape(-1)
    aff = torch.cat([aff_right, aff_down])
    same = torch.cat([same_right, same_down])
    return aff[same], aff[~same]


def calibrate_structure_thresholds(
    embeddings: torch.Tensor,
    masks: torch.Tensor,
    target_precision: float = 0.9,
    fallback_same: float = 0.75,
    fallback_boundary: float = 0.25,
) -> StructureCalibration:
    if not 0.5 <= target_precision < 1.0:
        raise ValueError("target_precision should be in [0.5, 1.0)")
    same_aff, boundary_aff = _edge_samples(embeddings, masks)
    if same_aff.numel() < 2 or boundary_aff.numel() < 2:
        return StructureCalibration(fallback_same, fallback_boundary, target_precision, int(same_aff.numel()), int(boundary_aff.numel()))
    tau_same = float(torch.quantile(same_aff, 1.0 - target_precision).item())
    tau_boundary = float(torch.quantile(boundary_aff, target_precision).item())
    if tau_boundary >= tau_same:
        midpoint = float((tau_same + tau_boundary) / 2.0)
        tau_same = min(0.99, midpoint + 0.05)
        tau_boundary = max(-0.99, midpoint - 0.05)
    return StructureCalibration(tau_same, tau_boundary, target_precision, int(same_aff.numel()), int(boundary_aff.numel()))
