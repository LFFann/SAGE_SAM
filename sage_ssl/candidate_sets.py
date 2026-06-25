"""Conflict-to-set pseudo supervision for two calibrated branches."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .conformal_calibration import reliability_from_q
from .structures import CandidateSets


def _branch_set(prob: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q = q.to(prob.device, dtype=prob.dtype)
    threshold = (1.0 - q).view(1, -1, 1, 1)
    selected = prob >= threshold
    top = F.one_hot(prob.argmax(dim=1), prob.shape[1]).permute(0, 3, 1, 2).bool()
    selected = torch.where(selected.any(dim=1, keepdim=True), selected, top)
    return selected, reliability_from_q(prob, q)


def build_candidate_sets(prob_a: torch.Tensor, prob_b: torch.Tensor, q_a: torch.Tensor, q_b: torch.Tensor) -> CandidateSets:
    set_a, rel_a = _branch_set(prob_a, q_a)
    set_b, rel_b = _branch_set(prob_b, q_b)
    union = set_a | set_b
    core = set_a & set_b
    negative = ~union
    unknown = union.sum(dim=1, keepdim=True) > 1
    return CandidateSets(set_a=set_a, set_b=set_b, union=union, core=core, negative=negative, unknown=unknown, reliability_a=rel_a, reliability_b=rel_b)
