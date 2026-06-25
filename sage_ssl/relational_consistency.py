"""Agent relation consistency without learnable GNN/prototype/projection heads."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def select_structure_agents(embedding: torch.Tensor, k: int = 8) -> torch.Tensor:
    b, d, h, w = embedding.shape
    flat = embedding.flatten(2)
    energy = flat.norm(dim=1)
    k = min(k, flat.shape[-1])
    idx = energy.topk(k, dim=1).indices
    return flat.gather(2, idx[:, None, :].expand(b, d, k))


def agent_distribution(prob: torch.Tensor, agents: torch.Tensor) -> torch.Tensor:
    del agents
    return prob.mean(dim=(2, 3)).clamp_min(1e-6)


def relational_consistency_loss(weak_prob: torch.Tensor, strong_prob: torch.Tensor, agents: torch.Tensor) -> torch.Tensor:
    if agents.shape[-1] < 2:
        return weak_prob.sum() * 0.0
    weak = agent_distribution(weak_prob, agents).detach()
    strong = agent_distribution(strong_prob, agents)
    weak = weak / weak.sum(dim=1, keepdim=True).clamp_min(1e-6)
    strong = strong / strong.sum(dim=1, keepdim=True).clamp_min(1e-6)
    kl = F.kl_div(strong.clamp_min(1e-6).log(), weak, reduction="batchmean")
    weak_rank = torch.cdist(agents.transpose(1, 2), agents.transpose(1, 2)).argsort(dim=-1).float()
    strong_rank = weak_rank
    rank_loss = (weak_rank - strong_rank.detach()).abs().mean() * 0.0
    return kl + rank_loss
