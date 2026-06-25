"""Shared lightweight data containers for SAGE-SAM."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class StructureGraph:
    affinity: torch.Tensor
    valid: torch.Tensor
    same: torch.Tensor
    boundary: torch.Tensor
    unknown: torch.Tensor


@dataclass
class CandidateSets:
    set_a: torch.Tensor
    set_b: torch.Tensor
    union: torch.Tensor
    core: torch.Tensor
    negative: torch.Tensor
    unknown: torch.Tensor
    reliability_a: torch.Tensor
    reliability_b: torch.Tensor
