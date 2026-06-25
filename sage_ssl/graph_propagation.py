"""Parameter-free probability propagation over reliable same-structure edges."""

from __future__ import annotations

import torch

from .structures import StructureGraph


def propagate_on_same_edges(prob: torch.Tensor, graph: StructureGraph, candidate_union: torch.Tensor | None = None) -> torch.Tensor:
    out = prob.clone()
    count = torch.ones_like(prob[:, :1])
    same = graph.same
    right = same[:, 0]
    if bool(right.any()):
        out[:, :, :, :-1] += prob[:, :, :, 1:] * right[:, None, :, :-1]
        out[:, :, :, 1:] += prob[:, :, :, :-1] * right[:, None, :, :-1]
        count[:, :, :, :-1] += right[:, None, :, :-1]
        count[:, :, :, 1:] += right[:, None, :, :-1]
    down = same[:, 1]
    if bool(down.any()):
        out[:, :, :-1, :] += prob[:, :, 1:, :] * down[:, None, :-1, :]
        out[:, :, 1:, :] += prob[:, :, :-1, :] * down[:, None, :-1, :]
        count[:, :, :-1, :] += down[:, None, :-1, :]
        count[:, :, 1:, :] += down[:, None, :-1, :]
    out = out / count.clamp_min(1.0)
    if candidate_union is not None:
        out = out * candidate_union.to(out.dtype)
    return out / out.sum(dim=1, keepdim=True).clamp_min(1e-6)
