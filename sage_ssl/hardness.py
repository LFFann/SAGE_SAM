"""Instance hardness from branch disagreement and set ambiguity."""

from __future__ import annotations

import torch


def estimate_hardness(prob_a: torch.Tensor, prob_b: torch.Tensor, candidate_union: torch.Tensor | None = None) -> torch.Tensor:
    disagreement = (prob_a - prob_b).abs().mean(dim=(1, 2, 3))
    if candidate_union is None:
        ambiguity = torch.zeros_like(disagreement)
    else:
        ambiguity = (candidate_union.float().sum(dim=1) - 1.0).clamp_min(0.0).mean(dim=(1, 2))
        ambiguity = ambiguity / max(1, prob_a.shape[1] - 1)
    return (0.7 * disagreement + 0.3 * ambiguity).detach().clamp(0.0, 1.0)
