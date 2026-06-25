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
from sage_ssl.conformal_calibration import classwise_conformal_q


@dataclass
class SAGETrainOutput:
    total: torch.Tensor
    supervised: torch.Tensor
    set_loss: torch.Tensor
    structure_loss: torch.Tensor
    hardness: torch.Tensor
    relation_loss: torch.Tensor | None = None
    singleton_ratio: float = 0.0
    ambiguous_ratio: float = 0.0
    unknown_ratio: float = 0.0
    mean_set_size: float = 0.0
    mean_instance_weight: float = 1.0


class SAGETrainer:
    def __init__(
        self,
        model: DualSegmentor,
        optimizer: torch.optim.Optimizer,
        num_classes: int = 3,
        warmup_iterations: int = 0,
        gradient_clip_norm: float | None = None,
        class_weights: torch.Tensor | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.warmup_iterations = warmup_iterations
        self.gradient_clip_norm = gradient_clip_norm
        self.class_weights = class_weights
        self.semantic_calibration_state: dict | None = None

    def optimizer_parameter_sources(self) -> dict[str, int]:
        ids = {id(p) for group in self.optimizer.param_groups for p in group["params"]}
        return {
            "UNet": sum(p.numel() for p in self.model.UNet.parameters() if id(p) in ids),
            "VNet": sum(p.numel() for p in self.model.VNet.parameters() if id(p) in ids),
        }

    def supervised_loss(self, image: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        out = self.model(image, return_features=False)
        weight = self.class_weights.to(image.device) if self.class_weights is not None else None
        loss_a = F.cross_entropy(out["logits_a"], label.long(), weight=weight)
        loss_b = F.cross_entropy(out["logits_b"], label.long(), weight=weight)
        return 0.5 * (loss_a + loss_b)

    def ssl_loss(self, image: torch.Tensor, structure_embedding: torch.Tensor, q_a: torch.Tensor, q_b: torch.Tensor | None = None) -> SAGETrainOutput:
        if q_b is None:
            q_b = q_a
        out = self.model(image, return_features=False)
        prob_a, prob_b = out["prob_a"], out["prob_b"]
        sets = build_candidate_sets(prob_a.detach(), prob_b.detach(), q_a, q_b)
        set_loss = partial_label_loss(prob_a, sets.union) + partial_label_loss(prob_b, sets.union)
        set_loss = 0.5 * set_loss + 0.5 * (negative_set_loss(prob_a, sets.negative) + negative_set_loss(prob_b, sets.negative))
        graph = build_local_structure_graph(structure_embedding)
        prob_graph = F.interpolate(0.5 * (prob_a + prob_b), size=structure_embedding.shape[-2:], mode="bilinear", align_corners=False)
        structure_loss = same_edge_kl_loss(prob_graph, graph) + 0.1 * boundary_agreement_penalty(prob_graph, graph)
        hardness = estimate_hardness(prob_a.detach(), prob_b.detach(), sets.union)
        total = set_loss + 0.1 * structure_loss
        set_size = sets.union.float().sum(dim=1)
        return SAGETrainOutput(
            total=total,
            supervised=total * 0.0,
            set_loss=set_loss,
            structure_loss=structure_loss,
            hardness=hardness,
            relation_loss=total * 0.0,
            singleton_ratio=float((set_size == 1).float().mean().item()),
            ambiguous_ratio=float((set_size > 1).float().mean().item()),
            unknown_ratio=float(sets.unknown.float().mean().item()),
            mean_set_size=float(set_size.mean().item()),
            mean_instance_weight=float((1.0 - hardness).mean().item()),
        )

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
        if self.gradient_clip_norm is not None and self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
        self.optimizer.step()
        return SAGETrainOutput(total=total.detach(), supervised=supervised.detach(), set_loss=set_loss.detach(), structure_loss=structure_loss.detach(), hardness=hardness.detach(), relation_loss=total.detach() * 0.0)

    def train_step(
        self,
        labeled_batch: dict,
        unlabeled_batch: dict | None,
        structure_batch: torch.Tensor | None,
        iteration: int,
        device: torch.device,
        scaler=None,
        amp_enabled: bool = False,
    ) -> SAGETrainOutput:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        labeled = (labeled_batch["image"].to(device), labeled_batch["mask"].to(device))
        total = self.supervised_loss(*labeled)
        supervised = total
        set_loss = total * 0.0
        structure_loss = total * 0.0
        hardness = torch.zeros(labeled[0].shape[0], device=device)
        metrics = {}
        if unlabeled_batch is not None:
            if structure_batch is None:
                raise ValueError("SSL step requires structure_batch when unlabeled_batch is provided")
            if self.semantic_calibration_state is None:
                q_a = torch.full((self.num_classes,), 0.5, device=device)
                q_b = q_a
            else:
                q_a = torch.tensor(self.semantic_calibration_state["q_A"], device=device, dtype=torch.float32)
                q_b = torch.tensor(self.semantic_calibration_state["q_B"], device=device, dtype=torch.float32)
            ssl = self.ssl_loss(unlabeled_batch["image"].to(device), structure_batch.to(device), q_a, q_b)
            total = total + ssl.total
            set_loss = ssl.set_loss
            structure_loss = ssl.structure_loss
            hardness = ssl.hardness
            metrics = {
                "singleton_ratio": ssl.singleton_ratio,
                "ambiguous_ratio": ssl.ambiguous_ratio,
                "unknown_ratio": ssl.unknown_ratio,
                "mean_set_size": ssl.mean_set_size,
                "mean_instance_weight": ssl.mean_instance_weight,
            }

        if not torch.isfinite(total):
            raise FloatingPointError(f"Non-finite total loss at iteration={iteration}: {float(total.detach().cpu())}")
        total.backward()
        if self.gradient_clip_norm is not None and self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
        self.optimizer.step()
        return SAGETrainOutput(
            total=total.detach(),
            supervised=supervised.detach(),
            set_loss=set_loss.detach(),
            structure_loss=structure_loss.detach(),
            hardness=hardness.detach(),
            relation_loss=total.detach() * 0.0,
            **metrics,
        )

    @torch.no_grad()
    def calibrate_semantics(self, calibration_loader, iteration: int, alpha: float = 0.1, device: torch.device | str = "cpu") -> dict:
        was_training = self.model.training
        self.model.eval()
        prob_a_chunks = []
        prob_b_chunks = []
        label_chunks = []
        for batch in calibration_loader:
            image = batch["image"].to(device)
            label = batch["mask"].to(device)
            out = self.model(image, return_features=False)
            prob_a_chunks.append(out["prob_a"].detach().cpu())
            prob_b_chunks.append(out["prob_b"].detach().cpu())
            label_chunks.append(label.detach().cpu())
        if not prob_a_chunks:
            raise ValueError("calibration_loader is empty; cannot calibrate semantics")
        prob_a = torch.cat(prob_a_chunks, dim=0)
        prob_b = torch.cat(prob_b_chunks, dim=0)
        labels = torch.cat(label_chunks, dim=0)
        q_a = classwise_conformal_q(prob_a, labels, self.num_classes, alpha=alpha)
        q_b = classwise_conformal_q(prob_b, labels, self.num_classes, alpha=alpha)
        state = {
            "iteration": iteration,
            "alpha": alpha,
            "q_A": [float(v) for v in q_a.tolist()],
            "q_B": [float(v) for v in q_b.tolist()],
            "reliability_A": [float(max(0.0, min(1.0, 1.0 - q))) for q in q_a.tolist()],
            "reliability_B": [float(max(0.0, min(1.0, 1.0 - q))) for q in q_b.tolist()],
        }
        self.semantic_calibration_state = state
        if was_training:
            self.model.train()
        return state
