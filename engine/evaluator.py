"""Evaluation helpers. Inference never loads SAM or the structure cache."""

from __future__ import annotations

import torch

from Model.sage_model import DualSegmentor


@torch.no_grad()
def predict_logits(model: DualSegmentor, image: torch.Tensor, branch_fusion: str = "mean") -> torch.Tensor:
    model.eval()
    out = model(image, return_features=False)
    logits_a = out["logits_a"]
    logits_b = out["logits_b"]
    if logits_a is None or logits_b is None:
        raise RuntimeError("DualSegmentor inference requires both branches")
    if branch_fusion == "a":
        return logits_a
    if branch_fusion == "b":
        return logits_b
    return 0.5 * (logits_a + logits_b)


def inference_summary() -> dict:
    return {
        "inference_uses_sam": False,
        "inference_reads_precomputed_structure": False,
        "active_model": "DualSegmentor",
    }
