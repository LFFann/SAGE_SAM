"""Structure consistency losses on a calibrated local graph."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .structures import StructureGraph


def same_edge_kl_loss(prob: torch.Tensor, graph: StructureGraph) -> torch.Tensor:
    losses = []
    same = graph.same
    if bool(same[:, 0].any()):
        p = prob[:, :, :, :-1]
        q = prob[:, :, :, 1:]
        m = same[:, 0, :, :-1]
        losses.append((F.kl_div(p.clamp_min(1e-6).log(), q.detach(), reduction="none").sum(dim=1)[m]).mean())
    if bool(same[:, 1].any()):
        p = prob[:, :, :-1, :]
        q = prob[:, :, 1:, :]
        m = same[:, 1, :-1, :]
        losses.append((F.kl_div(p.clamp_min(1e-6).log(), q.detach(), reduction="none").sum(dim=1)[m]).mean())
    if not losses:
        return prob.sum() * 0.0
    return torch.stack(losses).mean()


def boundary_agreement_penalty(prob: torch.Tensor, graph: StructureGraph) -> torch.Tensor:
    penalties = []
    boundary = graph.boundary
    if bool(boundary[:, 0].any()):
        sim = (prob[:, :, :, :-1] * prob[:, :, :, 1:]).sum(dim=1)
        penalties.append(sim[boundary[:, 0, :, :-1]].mean())
    if bool(boundary[:, 1].any()):
        sim = (prob[:, :, :-1, :] * prob[:, :, 1:, :]).sum(dim=1)
        penalties.append(sim[boundary[:, 1, :-1, :]].mean())
    if not penalties:
        return prob.sum() * 0.0
    return torch.stack(penalties).mean()
