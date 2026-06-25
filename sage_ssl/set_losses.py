"""Set-valued supervision losses for ambiguous pseudo labels."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def partial_label_loss(prob: torch.Tensor, candidate_set: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    candidate_set = candidate_set.to(device=prob.device, dtype=prob.dtype)
    set_size = candidate_set.sum(dim=1)
    active = set_size < prob.shape[1]
    if not bool(active.any()):
        return prob.sum() * 0.0
    set_prob = (prob * candidate_set).sum(dim=1).clamp_min(1e-6)
    loss = -torch.log(set_prob)
    if weight is not None:
        loss = loss * weight.to(loss.device)
    return loss[active].mean()


def negative_set_loss(prob: torch.Tensor, negative_set: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    negative_set = negative_set.to(device=prob.device, dtype=prob.dtype)
    if not bool(negative_set.any()):
        return prob.sum() * 0.0
    outside_prob = (prob * negative_set).sum(dim=1).clamp(0.0, 1.0 - 1e-6)
    loss = -torch.log1p(-outside_prob)
    if weight is not None:
        loss = loss * weight.to(loss.device)
    return loss.mean()


def singleton_or_set_loss(logits: torch.Tensor, candidate_set: torch.Tensor, class_weight: torch.Tensor | None = None) -> torch.Tensor:
    singleton = candidate_set.sum(dim=1) == 1
    if bool(singleton.all()):
        target = candidate_set.float().argmax(dim=1)
        return F.cross_entropy(logits, target, weight=class_weight)
    return partial_label_loss(torch.softmax(logits, dim=1), candidate_set)
