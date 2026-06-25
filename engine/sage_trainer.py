"""SAGE-SAM training step implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from Model.sage_model import DualSegmentor
from sage_ssl.candidate_sets import build_candidate_sets
from sage_ssl.hardness import estimate_hardness
from sage_ssl.set_losses import negative_set_loss, partial_label_loss
from sage_ssl.structure_graph import build_local_structure_graph
from sage_ssl.structure_losses import boundary_agreement_penalty, same_edge_kl_loss


@dataclass
class SAGETrainOutput:
    total: torch.Tensor
    supervised: torch.Tensor
    set_loss: torch.Tensor
    structure_loss: torch.Tensor
    hardness: torch.Tensor


class SAGETrainer:
    def __init__(self, model: DualSegmentor, optimizer: torch.optim.Optimizer, num_classes: int = 3):
        self.model = model
        self.optimizer = optimizer
        self.num_classes = num_classes

    def optimizer_parameter_sources(self) -> dict[str, int]:
        ids = {id(p) for group in self.optimizer.param_groups for p in group["params"]}
        return {
            "UNet": sum(p.numel() for p in self.model.UNet.parameters() if id(p) in ids),
            "VNet": sum(p.numel() for p in self.model.VNet.parameters() if id(p) in ids),
        }

    def supervised_loss(self, image: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        out = self.model(image, return_features=False)
        loss_a = F.cross_entropy(out["logits_a"], label.long())
        loss_b = F.cross_entropy(out["logits_b"], label.long())
        return 0.5 * (loss_a + loss_b)

    def ssl_loss(self, image: torch.Tensor, structure_embedding: torch.Tensor, q: torch.Tensor) -> SAGETrainOutput:
        out = self.model(image, return_features=False)
        prob_a, prob_b = out["prob_a"], out["prob_b"]
        sets = build_candidate_sets(prob_a.detach(), prob_b.detach(), q, q)
        set_loss = partial_label_loss(prob_a, sets.union) + partial_label_loss(prob_b, sets.union)
        set_loss = 0.5 * set_loss + 0.5 * (negative_set_loss(prob_a, sets.negative) + negative_set_loss(prob_b, sets.negative))
        graph = build_local_structure_graph(structure_embedding)
        prob_graph = F.interpolate(0.5 * (prob_a + prob_b), size=structure_embedding.shape[-2:], mode="bilinear", align_corners=False)
        structure_loss = same_edge_kl_loss(prob_graph, graph) + 0.1 * boundary_agreement_penalty(prob_graph, graph)
        hardness = estimate_hardness(prob_a.detach(), prob_b.detach(), sets.union)
        total = set_loss + 0.1 * structure_loss
        return SAGETrainOutput(total=total, supervised=total * 0.0, set_loss=set_loss, structure_loss=structure_loss, hardness=hardness)

    def step(self, labeled: tuple[torch.Tensor, torch.Tensor] | None = None, unlabeled: tuple[torch.Tensor, torch.Tensor] | None = None, q: torch.Tensor | None = None) -> SAGETrainOutput:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total = None
        supervised = torch.tensor(0.0)
        set_loss = torch.tensor(0.0)
        structure_loss = torch.tensor(0.0)
        hardness = torch.tensor([0.0])
        if labeled is not None:
            image, label = labeled
            supervised = self.supervised_loss(image, label)
            total = supervised
        if unlabeled is not None:
            image, structure_embedding = unlabeled
            if q is None:
                q = torch.full((self.num_classes,), 0.5, device=image.device)
            ssl = self.ssl_loss(image, structure_embedding, q.to(image.device))
            set_loss, structure_loss, hardness = ssl.set_loss, ssl.structure_loss, ssl.hardness
            total = ssl.total if total is None else total + ssl.total
        if total is None:
            raise ValueError("At least one labeled or unlabeled batch is required")
        total.backward()
        self.optimizer.step()
        return SAGETrainOutput(total=total.detach(), supervised=supervised.detach(), set_loss=set_loss.detach(), structure_loss=structure_loss.detach(), hardness=hardness.detach())
