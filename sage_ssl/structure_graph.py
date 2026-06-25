"""Non-parametric local structure graph from SAM image embeddings."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .structures import StructureGraph


def _neighbor_affinity(embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    emb = F.normalize(embedding.float(), dim=1)
    right = (emb[:, :, :, :-1] * emb[:, :, :, 1:]).sum(dim=1)
    down = (emb[:, :, :-1, :] * emb[:, :, 1:, :]).sum(dim=1)
    b, _, h, w = emb.shape
    affinity = emb.new_zeros((b, 2, h, w))
    valid = torch.zeros((b, 2, h, w), dtype=torch.bool, device=emb.device)
    affinity[:, 0, :, :-1] = right
    affinity[:, 1, :-1, :] = down
    valid[:, 0, :, :-1] = True
    valid[:, 1, :-1, :] = True
    return affinity.clamp(-1.0, 1.0), valid


def build_local_structure_graph(
    embedding: torch.Tensor,
    tau_same: float = 0.75,
    tau_boundary: float = 0.25,
) -> StructureGraph:
    if embedding.ndim != 4:
        raise ValueError("embedding must have shape [B,D,H,W]")
    if tau_boundary >= tau_same:
        raise ValueError("tau_boundary must be lower than tau_same")
    affinity, valid = _neighbor_affinity(embedding)
    same = valid & (affinity >= tau_same)
    boundary = valid & (affinity <= tau_boundary)
    unknown = valid & ~(same | boundary)
    return StructureGraph(affinity=affinity, valid=valid, same=same, boundary=boundary, unknown=unknown)
